"""Visualize per-frame BA pointing drift from a frame-rotation export JSON.

Local diagnostic tool, not a test or stable application entrypoint. Reads the
JSON produced by ``BundleFrameRotationExportSaveOp`` (see
hoshicore/ops/report_ops.py) and plots each frame's optical-axis direction,
expressed in the bundle-adjustment reference frame's coordinate system, as a
point on a unit sphere. Frame index maps to color, so drift/rotation across
the sequence is visible as a colored trail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-frame BA pointing directions on a unit sphere.")
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("frame_rotations.json"),
        help="JSON produced by BundleFrameRotationExportSaveOp")
    parser.add_argument(
        "--output", type=Path,
        help="Save the figure to this path (in addition to showing it)")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip the interactive window (use with --output on headless runs)")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument(
        "--event-zscore", type=float, default=6.0,
        help=("Robust (MAD) z-score threshold on frame-to-frame step size "
              "above which a step is flagged as a disturbance (tripod "
              "nudge, gust) rather than normal drift"))
    parser.add_argument(
        "--min-stable-segment", type=int, default=50,
        help=("Minimum length (frames) of a contiguous stable run to be "
              "trusted for fitting the rotation axis. Short windows make "
              "the PCA axis fit ill-conditioned (see module docstring of "
              "_fit_small_circle)."))
    parser.add_argument(
        "--min-eig-ratio", type=float, default=100.0,
        help=("Minimum ratio between the two smallest covariance "
              "eigenvalues for a candidate segment's axis fit to be "
              "trusted, on top of the length requirement."))
    parser.add_argument(
        "--axis-consistency-deg", type=float, default=3.0,
        help=("Max angle (degrees) between axes independently fitted on "
              "different stable segments to treat them as the same "
              "physical rotation axis rather than a real baseline change."))
    return parser.parse_args()


def _frame_pointing(rotation_ref_to_src: np.ndarray) -> np.ndarray:
    """This frame's own optical axis, expressed in reference-frame coordinates.

    ``rotation_ref_to_src`` maps reference-frame rays to this frame's rays
    (OpenCV convention: Z forward), so the transpose maps this frame's own
    forward axis back into the reference frame's coordinate system.
    """
    optical_axis = np.array([0.0, 0.0, 1.0])
    return np.asarray(rotation_ref_to_src, dtype=np.float64).T @ optical_axis


def _consecutive_steps_arcsec(indices: np.ndarray,
                              pointings: np.ndarray
                              ) -> np.ndarray:
    """Angular distance between each frame and the previous *available* one.

    Normalized by the index gap so a run of skipped frames doesn't read as
    one large step; a real tripod nudge shows as a spike regardless. Result
    aligns with ``indices[1:]``.
    """
    cos_step = np.clip(
        np.sum(pointings[1:] * pointings[:-1], axis=1), -1.0, 1.0)
    step_deg = np.rad2deg(np.arccos(cos_step))
    gaps = np.diff(indices)
    return step_deg * 3600.0 / np.maximum(gaps, 1)


def _detect_unstable_frames(step_arcsec: np.ndarray, zscore: float) -> np.ndarray:
    """Flag frames adjacent to a step-size outlier via a robust (MAD) z-score.

    A tripod nudge shows up as one or two step sizes far above the steady
    per-frame rate (sidereal drift, in this project's use case); regular
    variation in that rate does not produce outliers at this threshold.
    Marking both frames straddling each flagged step (not just one) keeps
    the flagged run from splitting the event's rising and falling edge
    across the stable/unstable boundary.

    Returns a boolean mask aligned with the full ``indices`` array (one
    longer than ``step_arcsec``).
    """
    median = np.median(step_arcsec)
    mad = np.median(np.abs(step_arcsec - median))
    if mad <= 1e-12:
        return np.zeros(len(step_arcsec) + 1, dtype=bool)
    z = (step_arcsec - median) / (1.4826 * mad)
    outlier_steps = np.where(np.abs(z) > zscore)[0]

    unstable = np.zeros(len(step_arcsec) + 1, dtype=bool)
    for pos in outlier_steps:
        unstable[max(0, pos - 1):min(len(unstable), pos + 3)] = True
    return unstable


def _unstable_windows(indices: np.ndarray,
                      unstable: np.ndarray) -> list[tuple[int, int]]:
    """Group a per-frame unstable mask into contiguous (start, end) index runs."""
    windows = []
    positions = np.where(unstable)[0]
    if len(positions) == 0:
        return windows
    run_start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos != prev + 1:
            windows.append((int(indices[run_start]), int(indices[prev])))
            run_start = pos
        prev = pos
    windows.append((int(indices[run_start]), int(indices[prev])))
    return windows


def _find_stable_segments(indices: np.ndarray, unstable: np.ndarray,
                          min_length: int) -> list[tuple[int, int]]:
    """Array-position (start, end) ranges of contiguous stable runs at least
    ``min_length`` frames long.

    Mirrors ``_unstable_windows``'s grouping but on the stable complement,
    and drops runs too short to trust for axis fitting -- see the
    ill-conditioning note in ``_fit_axis_model``.
    """
    positions = np.where(~unstable)[0]
    if len(positions) == 0:
        return []
    segments = []
    run_start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos != prev + 1:
            if prev - run_start + 1 >= min_length:
                segments.append((int(run_start), int(prev)))
            run_start = pos
        prev = pos
    if prev - run_start + 1 >= min_length:
        segments.append((int(run_start), int(prev)))
    return segments


def _axis_eig_ratio(pointings: np.ndarray) -> tuple[np.ndarray, float]:
    """Smallest-variance eigenvector and its separation from the next one.

    A large ratio means PCA has a well-defined single smallest-variance
    direction (a trustworthy axis); a small ratio means the two smallest
    eigenvalues are close enough that noise -- not geometry -- decides
    which direction PCA calls "the axis". This is the actual failure mode
    behind the short-window axis flips this whole segment search exists to
    route around.
    """
    mean_pointing = pointings.mean(axis=0)
    covariance = np.cov((pointings - mean_pointing).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, 0]
    if axis[2] < 0:
        axis = -axis
    ratio = (float(eigenvalues[1] / eigenvalues[0])
             if eigenvalues[0] > 1e-18 else float("inf"))
    return axis, ratio


def _max_pairwise_axis_angle_deg(axes: list[np.ndarray]) -> float:
    """Largest angle (degrees) between any pair of fitted axis vectors."""
    if len(axes) < 2:
        return 0.0
    max_angle = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cos_angle = np.clip(axes[i] @ axes[j], -1.0, 1.0)
            max_angle = max(max_angle, float(np.degrees(np.arccos(cos_angle))))
    return max_angle


def _fit_axis_theta0(pointings: np.ndarray
                     ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fixed-axis PCA fit: rotation axis, angular radius, and an orthonormal
    in-plane basis. See ``_fit_small_circle`` for the physical model.

    Split out from the phase/rate fit so a global axis can be estimated by
    pooling many well-conditioned frames (PCA over the covariance is stable
    under pooling) while phase/rate stays fit per-segment -- see
    ``_fit_world_frame_model``.
    """
    mean_pointing = pointings.mean(axis=0)
    covariance = np.cov((pointings - mean_pointing).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, 0]  # smallest-variance direction = rotation axis
    if axis[2] < 0:
        axis = -axis  # orient consistently across events/runs
    theta0 = float(np.arccos(np.clip(axis @ mean_pointing, -1.0, 1.0)))

    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array(
        [0.0, 1.0, 0.0])
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return axis, theta0, e1, e2


def _fit_phase_rate(pointings: np.ndarray, indices: np.ndarray,
                    axis: np.ndarray, e1: np.ndarray, e2: np.ndarray
                    ) -> tuple[float, float]:
    """Phase-vs-index line (rad/frame, rad) for a fixed axis/basis.

    Deliberately kept local to whatever ``pointings``/``indices`` are
    passed in (one stable segment, not the whole sequence): ``np.unwrap``
    across a large *unobserved* index gap between two segments has no data
    to anchor the number of half-turns it inserts, so a tiny per-segment
    rate difference (well within noise) gets amplified by the gap length
    into a large, spurious phase offset. Keeping this per-segment and
    letting each frame be scored against its own nearest segment (see
    ``_fit_world_frame_model``) avoids that failure mode entirely.
    """
    in_plane = pointings - (pointings @ axis)[:, None] * axis[None, :]
    phase = np.unwrap(np.arctan2(in_plane @ e2, in_plane @ e1))
    design = np.stack([indices, np.ones_like(indices)], axis=1)
    (omega, phi0), *_ = np.linalg.lstsq(design, phase, rcond=None)
    return float(omega), float(phi0)


def _fit_small_circle(pointings: np.ndarray, indices: np.ndarray
                      ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray,
                                float, float]:
    """Fit a fixed-axis, constant-angular-rate rotation to a set of pointings.

    This is the "world frame" the user asked for: sequence drift here comes
    from Earth's rotation, which sweeps every fixed camera pointing along a
    small circle around the celestial pole at a constant rate — not a curve
    that needs a polynomial to approximate over hundreds of frames. Fitting
    that physical model directly (axis via PCA on the unit vectors, phase
    via linear regression against frame index) gives an extrapolation basis
    with no start/end-point dependency and no assumption that the rate
    varies smoothly (which a polynomial trend silently makes).

    Intended for a single contiguous, well-conditioned window (one PCA fit,
    one phase unwrap) -- see ``_fit_axis_theta0``/``_fit_phase_rate`` for the
    split version used to combine multiple disjoint segments.

    Returns (axis, theta0, e1, e2, omega, phi0): ``axis`` is the fitted
    rotation axis, ``theta0`` the fixed angular distance from it, ``e1``/
    ``e2`` an orthonormal basis for the plane perpendicular to ``axis``, and
    ``omega``/``phi0`` the phase-vs-index line (rad/frame, rad).
    """
    axis, theta0, e1, e2 = _fit_axis_theta0(pointings)
    omega, phi0 = _fit_phase_rate(pointings, indices, axis, e1, e2)
    return axis, theta0, e1, e2, omega, phi0


def _predict_small_circle(axis: np.ndarray, theta0: float, e1: np.ndarray,
                          e2: np.ndarray, omega: float, phi0: float,
                          indices: np.ndarray) -> np.ndarray:
    """Predicted pointing at each index under the fitted small-circle model."""
    phase = omega * indices + phi0
    return (np.cos(theta0) * axis[None, :] + np.sin(theta0) *
            (np.cos(phase)[:, None] * e1[None, :] +
             np.sin(phase)[:, None] * e2[None, :]))


def _decompose_deviation(pointings: np.ndarray, indices: np.ndarray,
                         axis: np.ndarray, theta0: float, e1: np.ndarray,
                         e2: np.ndarray, omega: float, phi0: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along-track/cross-track/total deviation of ``pointings`` from a fitted
    small-circle model, evaluated at ``indices`` (see ``_fit_small_circle``).
    """
    predicted = _predict_small_circle(axis, theta0, e1, e2, omega, phi0,
                                      indices)
    phase = omega * indices + phi0
    along = -np.sin(phase)[:, None] * e1[None, :] + np.cos(phase)[:, None] * e2[None, :]
    cross = (-np.sin(theta0) * axis[None, :] + np.cos(theta0) *
            (np.cos(phase)[:, None] * e1[None, :] +
             np.sin(phase)[:, None] * e2[None, :]))
    deviation = pointings - predicted
    along_arcsec = np.degrees(np.sum(deviation * along, axis=1)) * 3600.0
    cross_arcsec = np.degrees(np.sum(deviation * cross, axis=1)) * 3600.0
    cos_dev = np.clip(np.sum(predicted * pointings, axis=1), -1.0, 1.0)
    total_arcsec = np.degrees(np.arccos(cos_dev)) * 3600.0
    return along_arcsec, cross_arcsec, total_arcsec


def _fit_world_frame_model(pointings: np.ndarray, indices: np.ndarray,
                           unstable: np.ndarray, min_length: int,
                           min_eig_ratio: float, axis_consistency_deg: float
                           ) -> dict:
    """Fit a sidereal-rate (fixed-axis, constant-rate) model and decompose
    every frame's deviation from it into along-track / cross-track components.

    Axis fitting is decoupled from "which window happens to precede the
    first flagged event": any short window can make the PCA axis fit in
    ``_fit_small_circle`` ill-conditioned (near-degenerate two smallest
    eigenvalues), which lets the axis -- and the omega/phi0 phase fit that
    depends on it -- flip or drift with the window's position rather than
    the physics. So this instead scans the *whole* sequence for long,
    well-conditioned stable segments (``_find_stable_segments`` +
    ``_axis_eig_ratio``), fits each independently, and only then decides
    whether those independent fits agree.

    If the trusted segments' axes agree within ``axis_consistency_deg``,
    they almost certainly are the same physical rotation axis (Earth's
    pole), so all their frames are pooled into one final global fit. If
    they disagree, that is itself evidence of a real baseline change (the
    camera's mount was reset or genuinely re-pointed, not just nudged) --
    forcing one global axis on such data would produce a virtual axis with
    no physical meaning, so each segment gets its own model instead and
    every other frame is evaluated against whichever trusted segment is
    nearest to it in time.

    Returns a dict with a ``"status"`` key: ``"ok"`` (single global model),
    ``"segmented"`` (baseline change detected, per-segment models), or
    ``"insufficient"`` (no segment was long/well-conditioned enough to
    trust; always includes a human-readable ``"message"``).
    """
    stable_segments = _find_stable_segments(indices, unstable, min_length)
    trusted = []
    for start, end in stable_segments:
        axis, ratio = _axis_eig_ratio(pointings[start:end + 1])
        if ratio >= min_eig_ratio:
            trusted.append((start, end, axis, ratio))

    if not trusted:
        # Report the true longest stable run for diagnostics, not just the
        # (possibly empty) length-filtered list above.
        all_stable_runs = _find_stable_segments(indices, unstable, 1)
        longest = max((end - start + 1 for start, end in all_stable_runs),
                      default=0)
        return {
            "status": "insufficient",
            "message": (
                f"no stable run of >= {min_length} frames had a "
                f"well-conditioned axis fit (eig_ratio >= {min_eig_ratio:g}); "
                f"longest stable run found was {longest} frame(s). Skipping "
                f"world-frame model rather than plot an ill-conditioned axis."),
        }

    max_angle = _max_pairwise_axis_angle_deg([t[2] for t in trusted])

    if max_angle <= axis_consistency_deg:
        # Axis/theta0 are pooled across all trusted segments: PCA over many
        # well-conditioned points is stable, and this is the "single
        # physical rotation axis" the agreement check just confirmed. But
        # omega/phi0 stay per-segment (see _fit_phase_rate) -- pooling the
        # phase across the large unobserved gaps between segments amplifies
        # each segment's own tiny rate noise into a large phantom offset.
        sel = np.zeros(len(indices), dtype=bool)
        for start, end, _, _ in trusted:
            sel[start:end + 1] = True
        axis, theta0, e1, e2 = _fit_axis_theta0(pointings[sel])

        segment_mid = np.array([(start + end) / 2.0 for start, end, _, _ in trusted])
        assigned = np.array([int(np.argmin(np.abs(segment_mid - pos)))
                             for pos in range(len(indices))])
        along_arcsec = np.empty(len(indices))
        cross_arcsec = np.empty(len(indices))
        total_arcsec = np.empty(len(indices))
        rates = []
        for seg_i, (start, end, _, _) in enumerate(trusted):
            omega, phi0 = _fit_phase_rate(
                pointings[start:end + 1], indices[start:end + 1], axis, e1, e2)
            rates.append(omega)
            mask = assigned == seg_i
            a, c, t = _decompose_deviation(
                pointings[mask], indices[mask], axis, theta0, e1, e2, omega, phi0)
            along_arcsec[mask], cross_arcsec[mask], total_arcsec[mask] = a, c, t

        return {
            "status": "ok",
            "along_arcsec": along_arcsec,
            "cross_arcsec": cross_arcsec,
            "total_arcsec": total_arcsec,
            "omega_arcsec_per_frame": float(np.degrees(np.mean(rates))) * 3600.0,
            "trusted_frame_count": int(sel.sum()),
            "segment_count": len(trusted),
            "max_axis_angle_deg": max_angle,
        }

    # Axes disagree beyond noise: build one model per trusted segment and
    # assign every frame (including unstable/untrusted ones) to whichever
    # trusted segment's array position it is temporally closest to.
    models = []
    for start, end, _, _ in trusted:
        axis, theta0, e1, e2, omega, phi0 = _fit_small_circle(
            pointings[start:end + 1], indices[start:end + 1])
        models.append({
            "axis": axis, "theta0": theta0, "e1": e1, "e2": e2,
            "omega": omega, "phi0": phi0,
            "index_range": (int(indices[start]), int(indices[end])),
            "mid_pos": (start + end) / 2.0,
        })

    segment_mid = np.array([m["mid_pos"] for m in models])
    assigned = np.array(
        [int(np.argmin(np.abs(segment_mid - pos))) for pos in range(len(indices))])

    along_arcsec = np.empty(len(indices))
    cross_arcsec = np.empty(len(indices))
    total_arcsec = np.empty(len(indices))
    for seg_i, model in enumerate(models):
        mask = assigned == seg_i
        if not mask.any():
            continue
        a, c, t = _decompose_deviation(
            pointings[mask], indices[mask], model["axis"], model["theta0"],
            model["e1"], model["e2"], model["omega"], model["phi0"])
        along_arcsec[mask], cross_arcsec[mask], total_arcsec[mask] = a, c, t

    return {
        "status": "segmented",
        "along_arcsec": along_arcsec,
        "cross_arcsec": cross_arcsec,
        "total_arcsec": total_arcsec,
        "segment_models": models,
        "assigned_segment": assigned,
        "max_axis_angle_deg": max_angle,
        "message": (
            f"camera pointing baseline changed: {len(models)} trusted "
            f"segments' axes differ by up to {max_angle:.1f} deg "
            f"(> {axis_consistency_deg:g} deg threshold) -- evaluating each "
            f"frame against its nearest segment's own model instead of one "
            f"global axis"),
    }


def main() -> int:
    args = _arguments()
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    indices, pointings = [], []
    skipped = 0
    for frame in payload["frames"]:
        rotation = frame["rotation_ref_to_src"]
        if rotation is None:
            skipped += 1
            continue
        indices.append(frame["index"])
        pointings.append(_frame_pointing(np.array(rotation)))

    if not pointings:
        raise ValueError("no frame has a resolved rotation_ref_to_src")

    indices = np.array(indices)
    pointings = np.array(pointings)
    reference_index = payload["camera"]["reference_frame_index"]

    fig = plt.figure(figsize=(15, 13))
    ax_sphere = fig.add_subplot(221, projection="3d")

    # Faint reference sphere for orientation.
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax_sphere.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="lightgray", linewidth=0.3, alpha=0.4, rstride=4, cstride=4)

    # Drift path in frame-index order (gaps at skipped frames are fine).
    ax_sphere.plot(pointings[:, 0], pointings[:, 1], pointings[:, 2],
                  color="tab:gray", linewidth=0.8, alpha=0.6)

    scatter = ax_sphere.scatter(
        pointings[:, 0], pointings[:, 1], pointings[:, 2],
        c=indices, cmap="viridis", s=18, depthshade=False)
    fig.colorbar(scatter, ax=ax_sphere, shrink=0.6, pad=0.1,
                label="frame index")

    ax_sphere.scatter([0], [0], [1], marker="*", s=200, color="red",
                      label=f"reference frame (index {reference_index})")

    ax_sphere.set_xlabel("X")
    ax_sphere.set_ylabel("Y")
    ax_sphere.set_zlabel("Z (reference optical axis)")
    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.view_init(elev=args.elev, azim=args.azim)
    ax_sphere.set_title("Full-sky view")
    ax_sphere.legend(loc="upper left")

    # Detect disturbance windows from frame-to-frame step-size outliers, then
    # fit a fixed-axis, constant-rate rotation (Earth's sidereal drift) to
    # the stable frames before the first one. Every other panel below reads
    # off this single physical model instead of a local curve fit, so
    # extrapolation quality doesn't degrade with distance from any
    # particular window of frames.
    step_arcsec = _consecutive_steps_arcsec(indices, pointings)
    unstable = _detect_unstable_frames(step_arcsec, args.event_zscore)
    windows = _unstable_windows(indices, unstable)
    world_frame = _fit_world_frame_model(
        pointings, indices, unstable, args.min_stable_segment,
        args.min_eig_ratio, args.axis_consistency_deg)
    if world_frame["status"] == "insufficient":
        print(f"world-frame model: {world_frame['message']}")
    elif world_frame["status"] == "segmented":
        print(f"world-frame model: {world_frame['message']}")

    step_median = float(np.median(step_arcsec))
    step_mad = float(np.median(np.abs(step_arcsec - step_median)))
    flagged_steps = (indices[1:][np.abs(step_arcsec - step_median) >
                                 args.event_zscore * 1.4826 * step_mad]
                     if step_mad > 1e-12 else np.array([]))
    step_detail = (f"step median={step_median:.1f} arcsec/frame, "
                  f"z-score threshold={args.event_zscore:g}, "
                  f"{len(flagged_steps)} step(s) flagged")

    # Spatial view of the deviation from the world-frame model: along-track
    # (tangent to the fitted drift direction — a timing/phase error) vs.
    # cross-track (perpendicular — an axis/declination error). The dominant
    # sidereal drift is already subtracted out by the model, so a real
    # disturbance's shape (one-off jump, swing-and-settle, oscillation) is
    # visible at its own natural scale without an arbitrary zoom factor.
    ax_spatial = fig.add_subplot(222)
    if world_frame["status"] == "insufficient":
        ax_spatial.set_title(
            "Spatial deviation: no well-conditioned stable segment found "
            "to fit a rotation axis (see console output)")
    else:
        along = world_frame["along_arcsec"]
        cross = world_frame["cross_arcsec"]
        ax_spatial.plot(along, cross, color="tab:gray", linewidth=0.6,
                        alpha=0.5)
        ax_spatial.scatter(along[~unstable], cross[~unstable],
                          c=indices[~unstable], cmap="viridis", s=16,
                          label="stable")
        if unstable.any():
            ax_spatial.scatter(along[unstable], cross[unstable],
                              facecolors="none", edgecolors="red", s=40,
                              linewidths=1.2, label="flagged unstable")
        ax_spatial.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax_spatial.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        ax_spatial.set_xlabel("along-track deviation (arcsec)")
        ax_spatial.set_ylabel("cross-track deviation (arcsec)")
        if world_frame["status"] == "ok":
            title = (
                f"Spatial deviation from sidereal-rate model "
                f"({world_frame['trusted_frame_count']} trusted frames from "
                f"{world_frame['segment_count']} segment(s), "
                f"rate={world_frame['omega_arcsec_per_frame']:.1f} "
                f"arcsec/frame)")
        else:
            title = (
                f"Spatial deviation (per-segment models, baseline changed: "
                f"axes differ by up to {world_frame['max_axis_angle_deg']:.1f}"
                f" deg)")
        ax_spatial.set_title(title)
        ax_spatial.set_aspect("equal")
        ax_spatial.legend(loc="best", fontsize=8)
        ax_spatial.grid(True, alpha=0.3)

    # Same deviation vs. frame index: answers *when* it happened and whether
    # it settled, which the spatial view above doesn't show on its own.
    ax_world = fig.add_subplot(212)
    if world_frame["status"] == "insufficient":
        ax_world.set_title(
            "World-frame deviation: no well-conditioned stable segment "
            "found to fit a rotation axis (see console output)")
    else:
        total = world_frame["total_arcsec"]
        ax_world.plot(indices, total, color="tab:purple", linewidth=1.0,
                     marker=".", markersize=3)
        for start, end in windows:
            ax_world.axvspan(start, end, color="red", alpha=0.15)
        if world_frame["status"] == "segmented":
            for model in world_frame["segment_models"]:
                ax_world.axvline(model["index_range"][0], color="tab:orange",
                                 linewidth=0.8, linestyle="--", alpha=0.6)
        if len(windows):
            settled = total[indices > windows[-1][1]]
            settled_desc = (f", settled level after last event ≈ "
                            f"{np.median(settled):.1f} arcsec"
                            if len(settled) else "")
        else:
            settled_desc = ""
        ax_world.set_xlabel("frame index")
        ax_world.set_ylabel("total deviation from sidereal-rate model (arcsec)")
        status_note = (
            f", {world_frame['message']}" if world_frame["status"] == "segmented"
            else "")
        ax_world.set_title(
            f"World-frame deviation over time ({step_detail}){settled_desc}"
            f"{status_note}", fontsize=9 if status_note else 10)
        ax_world.grid(True, alpha=0.3)

    title = f"Frame pointing drift ({len(pointings)} plotted"
    title += f", {skipped} skipped)" if skipped else ")"
    fig.suptitle(title)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"saved: {args.output}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
