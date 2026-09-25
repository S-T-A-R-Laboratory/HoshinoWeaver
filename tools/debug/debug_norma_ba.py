"""Run Norma bundle adjustment on every supported image in one folder.

This is a local diagnostic tool, not a test or a stable application entrypoint.
It compares the BA result with an adjacent-pair chain and, optionally, a
single-reference star of independent two-frame solves. Image-domain residual
shift is the primary accuracy signal; match residuals and graph consistency
explain failures.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from hoshicore.component.exif import read_exif_data
from hoshicore.component.image_io import load_img, save_img
from hoshicore.component.norma.bundle import (BundleFrame,
                                               FrameAlignmentStatus,
                                               build_bundle_plan)
from hoshicore.component.norma.detection import detect_star_points
from hoshicore.component.norma.frame_align import (
    AlignmentCameraCandidate, CameraInitializationPolicy,
    build_camera_candidate, solve_star_alignment)
from hoshicore.component.norma.geometry_view import to_gray_f64
from hoshicore.component.norma.intrinsics_from_exif import lens_type_from_exif
from hoshicore.component.norma.optimization import CameraOptimizationPolicy
from hoshicore.component.norma.types import BaseCameraModel
from hoshicore.component.utils import (ASTRO_SUFFIX, COMMON_SUFFIX,
                                       NOT_RECOM_SUFFIX, RAW_SUFFIX)


IMAGE_SUFFIXES = frozenset(COMMON_SUFFIX + NOT_RECOM_SUFFIX + RAW_SUFFIX +
                           ASTRO_SUFFIX)


def _csv_numbers(value: str, cast=float) -> tuple:
    try:
        result = tuple(cast(item.strip()) for item in value.split(",")
                       if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Norma BA against an adjacent-pair chain.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--exclude-glob", action="append", default=[],
                        help="Repeatable filename/path glob to exclude")
    parser.add_argument("--evaluation-mask", type=Path)
    parser.add_argument(
        "--detection-mask", type=Path,
        help="Binary mask applied during star detection and matching")
    parser.add_argument("--no-exif", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--uniform-sample-fraction", type=float,
        help="Uniformly retain this fraction of the discovered sequence")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--lens-type", choices=("auto", "perspective", "fisheye"),
                        default="auto")
    parser.add_argument("--focal-length-mm", type=float)
    parser.add_argument("--crop-factor", type=float, default=1.0)
    parser.add_argument("--fallback-focal-equiv-mm", type=float, default=20.0)
    parser.add_argument("--distortion", type=lambda value: _csv_numbers(value, float))
    parser.add_argument("--optimize-focal", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--optimize-distortion",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimize-principal-point",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--allow-large-principal-point-offset", action="store_true",
        help="Allow each principal-point axis to move by up to 50%% instead of 5%%")
    parser.add_argument("--pair-offsets",
                        type=lambda value: _csv_numbers(value, int), default=(1, 2, 4))
    parser.add_argument(
        "--diagnostic-offsets", type=lambda value: _csv_numbers(value, int),
        default=(),
        help=("Additional pair offsets to evaluate without adding them to the "
              "BA graph; offset N evaluates every available i->i+N pair"))
    parser.add_argument(
        "--baseline-camera", choices=("initial", "ba"), default="initial",
        help="Fixed camera used by the rotation-only diagnostic pair solves")
    parser.add_argument(
        "--star-reference-baseline", action="store_true",
        help=("Also solve every reference-to-frame pair independently, "
              "including shared camera parameters for each pair"))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument(
        "--max-pairs-per-edge", type=int, default=512,
        help="Cap matched pairs sampled per BA graph edge")
    parser.add_argument(
        "--camera-solve-frames", type=int, default=None,
        help="Solve the shared camera on an evenly spaced subset of this many "
             "frames first, then refine poses over the whole sequence")
    parser.add_argument(
        "--edge-topology", choices=("dense", "multiscale"), default="multiscale",
        help=("dense: every (pos, pos+offset) pair per --pair-offsets entry. "
              "multiscale: sparse interleaved-start dyadic topology, "
              "~2N edges regardless of scale count (--pair-offsets is ignored)"))
    parser.add_argument(
        "--max-pair-offset", type=int, default=None,
        help="Cap the maximum frame-index gap an edge may span (multiscale only)")
    parser.add_argument("--map-scale", type=float, default=0.5)
    parser.add_argument(
        "--image-scale", type=float, default=1.0,
        help=("Resize source images and camera intrinsics before remap; "
              "use 0.25 for a low-memory full-field diagnostic"))
    parser.add_argument("--write-max-stack", action="store_true",
                        help="Write BA and chain maximum-value composites")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--min-tile-mask-ratio", type=float, default=0.90)
    parser.add_argument("--min-phase-response", type=float, default=0.05)
    parser.add_argument("--no-image-metrics", action="store_true",
                        help="Skip image reload, remap, and phase metrics")
    parser.add_argument("--no-write-images", action="store_true")
    parser.add_argument("--output-format", choices=("png", "tif"), default="png")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def _image_paths(folder: Path, recursive: bool, output_dir: Path,
                 exclude_globs: list[str]) -> list[Path]:
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    output_resolved = output_dir.resolve()
    paths = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower().lstrip(".") not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(folder)
        if any(relative.match(pattern) for pattern in exclude_globs):
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            paths.append(path)
    return sorted(paths, key=lambda item: str(item.relative_to(folder)).lower())


def _scaled_camera(camera: BaseCameraModel,
                   scale: float) -> BaseCameraModel:
    """Express the same physical camera in resized pixel coordinates."""
    if not 0.0 < scale <= 1.0:
        raise ValueError("image_scale must be in (0, 1]")
    if scale == 1.0:
        return camera
    intrinsics = camera.intrinsics
    scaled = dataclasses.replace(
        intrinsics,
        image_width_px=max(2, round(intrinsics.image_width_px * scale)),
        image_height_px=max(2, round(intrinsics.image_height_px * scale)),
        cx_px=(None if intrinsics.cx_px is None else intrinsics.cx_px * scale),
        cy_px=(None if intrinsics.cy_px is None else intrinsics.cy_px * scale),
    )
    return camera.with_intrinsics(scaled)


def _resize_for_evaluation(image: np.ndarray,
                           size: tuple[int, int]) -> np.ndarray:
    if image.shape[1::-1] == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _read_exif(path: Path) -> dict[str, str] | None:
    try:
        data = read_exif_data(str(path))
    except Exception as exc:
        print(f"warning: EXIF unavailable for {path.name}: {exc}")
        return None
    return data.exif if data is not None else None


def _camera_payload(camera: BaseCameraModel) -> dict[str, Any]:
    intrinsics = camera.intrinsics
    return {
        "projection": type(camera).__name__,
        "focal_length_mm": float(intrinsics.focal_length_mm),
        "principal_point_px": [float(value) for value in
                               intrinsics.principal_point_px],
        "distortion": [float(value) for value in
                       camera.distortion.to_cv2().reshape(-1)],
    }


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _pixel_residual_p90(match: Any, rotation: np.ndarray,
                        first_camera: BaseCameraModel,
                        second_camera: BaseCameraModel) -> float | None:
    first_rays = first_camera.unproject(match.ref_pts)
    predicted = second_camera.project((rotation @ first_rays.T).T)
    finite = np.all(np.isfinite(predicted), axis=1)
    if not np.any(finite):
        return None
    residuals = np.linalg.norm(predicted[finite] - match.src_pts[finite], axis=1)
    return float(np.percentile(residuals, 90))


def _local_normalize(image: np.ndarray) -> np.ndarray | None:
    gray = to_gray_f64(image).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=8.0, sigmaY=8.0)
    centered = gray - mean
    variance = cv2.GaussianBlur(centered * centered, (0, 0),
                                sigmaX=8.0, sigmaY=8.0)
    normalized = centered / np.sqrt(np.maximum(variance, 0.0) + 1e-6)
    if not np.all(np.isfinite(normalized)) or np.std(normalized) <= 1e-6:
        return None
    return np.clip(normalized, -6.0, 6.0)


def _tile_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def _image_residual(reference: np.ndarray, aligned: np.ndarray,
                    valid_mask: np.ndarray, args: argparse.Namespace
                    ) -> dict[str, Any]:
    height, width = valid_mask.shape
    size = min(args.tile_size, height, width)
    if size < 32:
        return {"status": "insufficient_support", "tiles": 0,
                "coverage": 0.0, "shift_p90_px": None}
    stride = min(args.tile_stride, size)
    window = cv2.createHanningWindow((size, size), cv2.CV_32F)
    shifts: list[float] = []
    coverage = np.zeros_like(valid_mask, dtype=bool)
    for y0 in _tile_starts(height, size, stride):
        for x0 in _tile_starts(width, size, stride):
            ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
            tile_mask = valid_mask[ys, xs]
            if float(np.mean(tile_mask)) < args.min_tile_mask_ratio:
                continue
            ref_tile = _local_normalize(reference[ys, xs])
            src_tile = _local_normalize(aligned[ys, xs])
            if ref_tile is None or src_tile is None:
                continue
            (dx, dy), response = cv2.phaseCorrelate(ref_tile, src_tile, window)
            shift = float(np.hypot(dx, dy))
            if (np.isfinite(response) and response >= args.min_phase_response
                    and np.isfinite(shift)):
                shifts.append(shift)
                coverage[ys, xs] |= tile_mask
    valid_count = max(int(np.count_nonzero(valid_mask)), 1)
    return {
        "status": "ok" if shifts else "insufficient_support",
        "tiles": len(shifts),
        "coverage": float(np.count_nonzero(coverage) / valid_count),
        "shift_p90_px": float(np.percentile(shifts, 90)) if shifts else None,
    }


def _summarize(values: list[float]) -> dict[str, Any]:
    return ({"count": 0, "median": None, "p90": None} if not values else {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    })


def _build_chain(pair_results: dict[tuple[int, int], dict[str, Any]],
                 reference: int) -> dict[int, np.ndarray]:
    neighbours: dict[int, list[tuple[int, np.ndarray]]] = {}
    for (first, second), result in pair_results.items():
        if second != first + 1 or result.get("error"):
            continue
        alignment = result["alignment"]
        neighbours.setdefault(first, []).append(
            (second, alignment.rotation_ref_to_src))
        neighbours.setdefault(second, []).append(
            (first, alignment.rotation_ref_to_src.T))
    rotations = {reference: np.eye(3, dtype=np.float64)}
    pending = [reference]
    while pending:
        first = pending.pop(0)
        for second, rotation in neighbours.get(first, []):
            if second in rotations:
                continue
            rotations[second] = rotation @ rotations[first]
            pending.append(second)
    return rotations


def main() -> int:
    args = _arguments()
    if not 0.0 < args.image_scale <= 1.0:
        raise ValueError("image-scale must be in (0, 1]")
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "debug_norma_ba_results").resolve()
    paths = _image_paths(input_dir, args.recursive, output_dir,
                         args.exclude_glob)
    if args.uniform_sample_fraction is not None:
        fraction = float(args.uniform_sample_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("uniform-sample-fraction must be in (0, 1]")
        sample_count = max(2, int(round(len(paths) * fraction)))
        sample_indices = np.linspace(
            0, len(paths) - 1, min(sample_count, len(paths)),
            dtype=np.int64)
        paths = [paths[int(index)] for index in sample_indices]
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if len(paths) < 2:
        raise ValueError("at least two supported images are required")
    if not 0 <= args.reference_index < len(paths):
        raise ValueError("reference-index is outside the discovered image list")
    output_dir.mkdir(parents=True, exist_ok=True)

    focal_equiv = (args.focal_length_mm * args.crop_factor
                   if args.focal_length_mm is not None else None)
    detection_mask_source = None
    if args.detection_mask is not None:
        detection_mask_source = load_img(str(args.detection_mask))
        if detection_mask_source is None:
            raise ValueError("detection mask is missing")
        if detection_mask_source.ndim == 3:
            detection_mask_source = np.max(detection_mask_source, axis=2)
    frames, candidates, star_counts = [], [], []
    started = time.perf_counter()
    stage_started = started
    for index, path in enumerate(paths):
        print(f"detect [{index + 1}/{len(paths)}] {path.name}")
        image = load_img(str(path))
        if image is None:
            raise FileNotFoundError(f"failed to load {path}")
        tags = None if args.no_exif else _read_exif(path)
        configured_lens = None if args.lens_type == "auto" else args.lens_type
        lens_type = configured_lens or lens_type_from_exif(tags)
        policy = CameraInitializationPolicy(
            lens_type=lens_type,
            fallback_focal_equiv_mm=args.fallback_focal_equiv_mm,
            optimize_focal=args.optimize_focal,
            optimize_distortion=args.optimize_distortion,
            optimize_principal_point=args.optimize_principal_point,
            allow_large_principal_point_offset=(
                args.allow_large_principal_point_offset),
        )
        candidate = build_camera_candidate(
            tags, image.shape, "distortion",
            init_distortion=(list(args.distortion) if args.distortion else None),
            focal_equiv_mm=focal_equiv, init_policy=policy)
        detection_mask = None
        if detection_mask_source is not None:
            detection_mask = cv2.resize(
                detection_mask_source, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST) > 0
        stars = detect_star_points(to_gray_f64(image), mask=detection_mask)
        candidates.append(candidate)
        star_counts.append(len(stars.positions))
        frames.append(BundleFrame(index, stars, candidate))

    detection_seconds = time.perf_counter() - stage_started
    print("solve BA")
    stage_started = time.perf_counter()
    plan = build_bundle_plan(
        frames, args.reference_index, pair_offsets=args.pair_offsets,
        random_seed=args.random_seed, max_nfev=args.max_nfev,
        max_pairs_per_edge=args.max_pairs_per_edge,
        camera_solve_frames=args.camera_solve_frames,
        edge_topology=args.edge_topology,
        max_pair_offset=args.max_pair_offset)
    ba_seconds = time.perf_counter() - stage_started

    # Solve the diagnostic graph independently. Offset-1 edges form the chain;
    # skip edges measure consistency of both chain and BA pose estimates.
    fixed_policy = CameraOptimizationPolicy(False, False, False, 0)
    fixed_camera = (plan.shared_camera if args.baseline_camera == "ba" else
                    candidates[args.reference_index].camera)
    fixed_candidate = AlignmentCameraCandidate(
        fixed_camera, fixed_policy, f"fixed_{args.baseline_camera}_baseline")
    pair_results: dict[tuple[int, int], dict[str, Any]] = {}
    diagnostic_offsets = tuple(sorted(
        set(args.pair_offsets) | set(args.diagnostic_offsets) | {1}))
    stage_started = time.perf_counter()
    for first in range(len(frames)):
        for offset in diagnostic_offsets:
            second = first + offset
            if second >= len(frames):
                continue
            print(f"pair {first}->{second}")
            try:
                alignment, match = solve_star_alignment(
                    frames[first].stars, frames[second].stars,
                    fixed_candidate, fixed_candidate,
                    bootstrap_scales=(1.0,), same_camera=True,
                    use_asterism_bootstrap=True, random_seed=args.random_seed)
                pair_results[(first, second)] = {
                    "alignment": alignment, "match": match, "error": None}
            except Exception as exc:
                pair_results[(first, second)] = {"error": str(exc)}
    baseline_seconds = time.perf_counter() - stage_started

    star_results: dict[int, dict[str, Any]] = {}
    stage_started = time.perf_counter()
    if args.star_reference_baseline:
        reference_frame = frames[args.reference_index]
        for index, frame in enumerate(frames):
            if index == args.reference_index:
                continue
            print(f"star {args.reference_index}->{index}")
            try:
                alignment, match = solve_star_alignment(
                    reference_frame.stars, frame.stars,
                    candidates[args.reference_index], candidates[index],
                    bootstrap_scales=(0.7, 1.0, 1.3), same_camera=True,
                    use_asterism_bootstrap=True,
                    random_seed=args.random_seed)
                star_results[index] = {
                    "alignment": alignment, "match": match, "error": None}
            except Exception as exc:
                star_results[index] = {"error": str(exc)}
    star_seconds = time.perf_counter() - stage_started

    chain_rotations = _build_chain(pair_results, args.reference_index)
    ba_rotations = {
        item.index: item.rotation_ref_to_src for item in plan.frames
        if item.status == FrameAlignmentStatus.SOLVED
        and item.rotation_ref_to_src is not None
    }

    pair_rows = []
    ba_pair_p90, direct_pair_p90 = [], []
    ba_skip_consistency, chain_skip_consistency = [], []
    for (first, second), result in pair_results.items():
        row: dict[str, Any] = {"first": first, "second": second,
                               "error": result.get("error")}
        if not result.get("error"):
            alignment, match = result["alignment"], result["match"]
            row["match_count"] = len(match.ref_pts)
            row["direct_match_p90_px"] = _pixel_residual_p90(
                match, alignment.rotation_ref_to_src,
                alignment.ref_camera, alignment.src_camera)
            if row["direct_match_p90_px"] is not None:
                direct_pair_p90.append(row["direct_match_p90_px"])
            if first in ba_rotations and second in ba_rotations:
                relative = ba_rotations[second] @ ba_rotations[first].T
                row["ba_match_p90_px"] = _pixel_residual_p90(
                    match, relative, plan.shared_camera, plan.shared_camera)
                row["ba_rotation_consistency_deg"] = _rotation_error_deg(
                    relative, alignment.rotation_ref_to_src)
                if row["ba_match_p90_px"] is not None:
                    ba_pair_p90.append(row["ba_match_p90_px"])
                if second - first > 1:
                    ba_skip_consistency.append(
                        row["ba_rotation_consistency_deg"])
            if first in chain_rotations and second in chain_rotations:
                relative = chain_rotations[second] @ chain_rotations[first].T
                row["chain_rotation_consistency_deg"] = _rotation_error_deg(
                    relative, alignment.rotation_ref_to_src)
                if second - first > 1:
                    chain_skip_consistency.append(
                        row["chain_rotation_consistency_deg"])
        pair_rows.append(row)

    ba_camera = _scaled_camera(plan.shared_camera, args.image_scale)
    baseline_camera = _scaled_camera(fixed_candidate.camera, args.image_scale)
    width = ba_camera.intrinsics.image_width_px
    height = ba_camera.intrinsics.image_height_px
    evaluation_size = (width, height)
    reference = (None if args.no_image_metrics else
                 load_img(str(paths[args.reference_index])))
    if not args.no_image_metrics and reference is None:
        raise FileNotFoundError(paths[args.reference_index])
    if reference is not None:
        reference = _resize_for_evaluation(reference, evaluation_size)
    evaluation_mask = np.full((height, width), 255, dtype=np.uint8)
    if args.evaluation_mask is not None:
        loaded_mask = load_img(str(args.evaluation_mask))
        if loaded_mask is None:
            raise ValueError("evaluation mask is missing")
        if loaded_mask.ndim == 3:
            loaded_mask = np.max(loaded_mask, axis=2)
        loaded_mask = cv2.resize(loaded_mask, evaluation_size,
                                 interpolation=cv2.INTER_NEAREST)
        evaluation_mask = np.where(loaded_mask > 0, 255, 0).astype(np.uint8)
    frame_rows = []
    ba_shifts, chain_shifts, star_shifts = [], [], []
    ba_max = None
    chain_max = None
    stage_started = time.perf_counter()
    for index, path in enumerate(paths):
        print(f"remap [{index + 1}/{len(paths)}] {path.name}")
        plan_frame = plan.frame(index)
        row = {
            "index": index, "file": str(path), "stars": star_counts[index],
            "status": plan_frame.status.value,
            "pose_source": plan_frame.pose_source,
            "incident_edges": plan_frame.incident_edge_count,
            "ba_edge_residual_p90_deg": (
                math.degrees(plan_frame.residual_p90_rad)
                if plan_frame.residual_p90_rad is not None else None),
        }
        if args.no_image_metrics:
            disabled = {"status": "disabled", "tiles": 0,
                        "coverage": 0.0, "shift_p90_px": None}
            row["ba_image"] = disabled
            row["chain_image"] = disabled.copy()
            row["star_image"] = disabled.copy()
            frame_rows.append(row)
            continue
        source = load_img(str(path))
        if source is None:
            raise FileNotFoundError(path)
        source = _resize_for_evaluation(source, evaluation_size)
        source_mask = evaluation_mask
        if plan_frame.rotation_ref_to_src is not None:
            ba_aligned = (source.copy() if index == args.reference_index else
                          ba_camera.project_image_from_camera(
                              ba_camera, source, evaluation_size,
                              rotation_dst_to_src=plan_frame.rotation_ref_to_src,
                              map_scale=args.map_scale))
            ba_mask = (source_mask if index == args.reference_index else
                       ba_camera.project_image_from_camera(
                           ba_camera, source_mask, evaluation_size,
                           interpolation=cv2.INTER_NEAREST,
                           rotation_dst_to_src=plan_frame.rotation_ref_to_src,
                           map_scale=args.map_scale))
            row["ba_image"] = _image_residual(
                reference, ba_aligned,
                (ba_mask > 250) & (evaluation_mask > 0), args)
            if row["ba_image"]["shift_p90_px"] is not None and index != args.reference_index:
                ba_shifts.append(row["ba_image"]["shift_p90_px"])
        else:
            ba_aligned = None
            row["ba_image"] = {"status": "no_pose", "tiles": 0,
                               "coverage": 0.0, "shift_p90_px": None}

        if index in chain_rotations:
            chain_aligned = (source.copy() if index == args.reference_index else
                             baseline_camera.project_image_from_camera(
                                 baseline_camera, source, evaluation_size,
                                 rotation_dst_to_src=chain_rotations[index],
                                 map_scale=args.map_scale))
            chain_mask = (source_mask > 250 if index == args.reference_index else
                          baseline_camera.project_image_from_camera(
                              baseline_camera, source_mask, evaluation_size,
                              interpolation=cv2.INTER_NEAREST,
                              rotation_dst_to_src=chain_rotations[index],
                              map_scale=args.map_scale) > 250)
            row["chain_image"] = _image_residual(
                reference, chain_aligned, chain_mask, args)
            if row["chain_image"]["shift_p90_px"] is not None and index != args.reference_index:
                chain_shifts.append(row["chain_image"]["shift_p90_px"])
        else:
            chain_aligned = None
            row["chain_image"] = {"status": "no_pose", "tiles": 0,
                                  "coverage": 0.0, "shift_p90_px": None}

        if index == args.reference_index and args.star_reference_baseline:
            star_aligned = source.copy()
            row["star_image"] = {
                "status": "reference", "tiles": 0,
                "coverage": 1.0, "shift_p90_px": 0.0}
        elif index in star_results and not star_results[index].get("error"):
            star_alignment = star_results[index]["alignment"]
            star_ref_camera = _scaled_camera(
                star_alignment.ref_camera, args.image_scale)
            star_src_camera = _scaled_camera(
                star_alignment.src_camera, args.image_scale)
            star_aligned = star_ref_camera.project_image_from_camera(
                star_src_camera, source, evaluation_size,
                rotation_dst_to_src=star_alignment.rotation_ref_to_src,
                map_scale=args.map_scale)
            star_mask = star_ref_camera.project_image_from_camera(
                star_src_camera, source_mask, evaluation_size,
                interpolation=cv2.INTER_NEAREST,
                rotation_dst_to_src=star_alignment.rotation_ref_to_src,
                map_scale=args.map_scale) > 250
            row["star_image"] = _image_residual(
                reference, star_aligned, star_mask & (evaluation_mask > 0),
                args)
            if row["star_image"]["shift_p90_px"] is not None:
                star_shifts.append(row["star_image"]["shift_p90_px"])
        else:
            star_aligned = None
            row["star_image"] = {
                "status": "disabled" if not args.star_reference_baseline
                else "no_pose",
                "tiles": 0, "coverage": 0.0, "shift_p90_px": None}

        if args.write_max_stack:
            if ba_aligned is not None:
                ba_max = (ba_aligned.copy() if ba_max is None else
                          np.maximum(ba_max, ba_aligned))
            if chain_aligned is not None:
                chain_max = (chain_aligned.copy() if chain_max is None else
                             np.maximum(chain_max, chain_aligned))

        if not args.no_write_images:
            stem = f"{index:04d}_{path.stem}.{args.output_format}"
            for name, image in (("src", source), ("ba", ba_aligned),
                                ("chain_shared", chain_aligned),
                                ("star_reference", star_aligned)):
                if image is not None:
                    folder = output_dir / name
                    folder.mkdir(exist_ok=True)
                    save_img(str(folder / stem), image)
        frame_rows.append(row)
    if args.write_max_stack:
        if ba_max is not None:
            save_img(str(output_dir / f"ba_max.{args.output_format}"), ba_max)
        if chain_max is not None:
            save_img(str(output_dir /
                         f"chain_shared_max.{args.output_format}"), chain_max)
    remap_seconds = time.perf_counter() - stage_started

    star_match_p90 = [
        value for result in star_results.values()
        if not result.get("error")
        for value in [_pixel_residual_p90(
            result["match"], result["alignment"].rotation_ref_to_src,
            result["alignment"].ref_camera,
            result["alignment"].src_camera)]
        if value is not None
    ]
    paired = [(row["ba_image"]["shift_p90_px"],
               row["chain_image"]["shift_p90_px"])
              for row in frame_rows if row["index"] != args.reference_index
              and row["ba_image"]["shift_p90_px"] is not None
              and row["chain_image"]["shift_p90_px"] is not None]
    benefit = {
        "primary_accuracy_metric": "reference remap local phase residual shift",
        "ba_image_shift_p90_px": _summarize(ba_shifts),
        "chain_image_shift_p90_px": _summarize(chain_shifts),
        "star_image_shift_p90_px": _summarize(star_shifts),
        "star_match_p90_px": _summarize(star_match_p90),
        "paired_image_frames": len(paired),
        "image_shift_median_improvement_px": (
            float(np.median([chain for _, chain in paired])
                  - np.median([ba for ba, _ in paired])) if paired else None),
        "ba_match_p90_px": _summarize(ba_pair_p90),
        "direct_pair_match_p90_px": _summarize(direct_pair_p90),
        "ba_skip_edge_consistency_deg": _summarize(ba_skip_consistency),
        "chain_skip_edge_consistency_deg": _summarize(chain_skip_consistency),
        "skip_edge_consistency_median_improvement_deg": (
            float(np.median(chain_skip_consistency)
                  - np.median(ba_skip_consistency))
            if ba_skip_consistency and chain_skip_consistency else None),
    }
    report = {
        "baseline": "fixed shared-camera adjacent-pair rotation chain",
        "config": {key: value for key, value in vars(args).items()
                   if key not in {"input_dir", "output_dir"}},
        "input_dir": str(input_dir), "files": [str(path) for path in paths],
        "reference_index": args.reference_index,
        "initial_reference_camera": _camera_payload(
            candidates[args.reference_index].camera),
        "ba_shared_camera": _camera_payload(plan.shared_camera),
        "accepted_edge_count": plan.accepted_edge_count,
        "rejected_edge_count": plan.rejected_edge_count,
        "observability_condition": plan.observability_condition,
        "rotations_ref_to_src": {
            str(frame.index): (
                frame.rotation_ref_to_src.tolist()
                if frame.rotation_ref_to_src is not None else None
            )
            for frame in plan.frames
        },
        "timings_seconds": {
            "load_detect": detection_seconds,
            "build_bundle_plan": ba_seconds,
            "baseline_pairs": baseline_seconds,
            "star_reference_pairs": star_seconds,
            "remap_and_metrics": remap_seconds,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "benefit": benefit, "frames": frame_rows, "pairs": pair_rows,
        "star_reference_pairs": {
            str(index): ({"error": result.get("error")}
                         if result.get("error") else {
                "error": None,
                "match_count": len(result["match"].ref_pts),
                "match_p90_px": _pixel_residual_p90(
                    result["match"], result["alignment"].rotation_ref_to_src,
                    result["alignment"].ref_camera,
                    result["alignment"].src_camera),
                "reference_camera": _camera_payload(
                    result["alignment"].ref_camera),
                "source_camera": _camera_payload(
                    result["alignment"].src_camera),
            }) for index, result in star_results.items()
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["index", "file", "stars", "status", "pose_source",
                  "incident_edges", "ba_edge_residual_p90_deg",
                  "ba_shift_p90_px", "chain_shift_p90_px",
                  "star_shift_p90_px"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow({**{key: row.get(key) for key in fields},
                             "ba_shift_p90_px": row["ba_image"]["shift_p90_px"],
                             "chain_shift_p90_px": row["chain_image"]["shift_p90_px"],
                             "star_shift_p90_px": row["star_image"]["shift_p90_px"]})
    print(json.dumps(benefit, indent=2, ensure_ascii=False))
    print(f"report: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
