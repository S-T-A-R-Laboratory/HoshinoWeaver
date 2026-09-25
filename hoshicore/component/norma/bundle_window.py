"""Window geometry derived from a bundle-adjustment alignment plan."""
from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from .bundle import BAAlignmentPlan, FrameAlignmentStatus


@dataclasses.dataclass(frozen=True)
class BundleWindowSource:
    """One solved source frame expressed in a window center's coordinates."""

    index: int
    rotation_center_to_source: NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class BundleWindowSpec:
    """Geometry needed to produce one center-frame window result."""

    center_index: int
    sources: tuple[BundleWindowSource, ...]


@dataclasses.dataclass(frozen=True)
class BundleWindowSchedule:
    """Immutable, input-ordered windows derived from a BA plan."""

    frame_count: int
    window_size: int
    windows: tuple[BundleWindowSpec, ...]


def _readonly_rotation(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(rotation, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def build_bundle_window_schedule(
    plan: BAAlignmentPlan,
    window_size: int,
    *,
    min_contributors: int = 1,
) -> BundleWindowSchedule:
    """Build shrinking-edge windows while omitting excluded BA frames.

    Excluded frames are never centers or contributors. A solved center is
    emitted only when at least ``min_contributors`` solved frames, including
    the center frame itself, remain in its truncated window.
    """
    if not isinstance(plan, BAAlignmentPlan):
        raise TypeError("plan must be a BAAlignmentPlan")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not 1 <= min_contributors <= window_size:
        raise ValueError("min_contributors must be in [1, window_size]")

    rotations: dict[int, NDArray[np.float64]] = {}
    for position, frame in enumerate(plan.frames):
        if frame.index != position:
            raise ValueError(
                "AlignmentPlan frames must use contiguous stream indices")
        if frame.status == FrameAlignmentStatus.EXCLUDED:
            continue
        if frame.rotation_ref_to_src is None:
            raise ValueError(
                f"solved frame {frame.index} has no reference-relative pose")
        rotation = np.asarray(frame.rotation_ref_to_src, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(
                f"frame {frame.index} rotation must be a finite 3x3 matrix")
        rotations[frame.index] = rotation

    radius = window_size // 2
    frame_count = len(plan.frames)
    windows: list[BundleWindowSpec] = []
    for center_index in range(frame_count):
        center_rotation = rotations.get(center_index)
        if center_rotation is None:
            continue
        start = max(0, center_index - radius)
        stop = min(frame_count, center_index + radius + 1)
        source_indices = [
            index for index in range(start, stop) if index in rotations
        ]
        if len(source_indices) < min_contributors:
            continue
        sources = []
        for source_index in source_indices:
            if source_index == center_index:
                relative = np.eye(3, dtype=np.float64)
            else:
                relative = rotations[source_index] @ center_rotation.T
            sources.append(BundleWindowSource(
                index=source_index,
                rotation_center_to_source=_readonly_rotation(relative),
            ))
        windows.append(BundleWindowSpec(
            center_index=center_index,
            sources=tuple(sources),
        ))

    return BundleWindowSchedule(
        frame_count=frame_count,
        window_size=window_size,
        windows=tuple(windows),
    )


def build_identity_window_schedule(
    frame_count: int,
    window_size: int,
    *,
    min_contributors: int = 1,
) -> BundleWindowSchedule:
    """Build shrinking-edge windows without any geometric transform.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not 1 <= min_contributors <= window_size:
        raise ValueError("min_contributors must be in [1, window_size]")

    identity = np.eye(3, dtype=np.float64)
    identity.setflags(write=False)
    windows: list[BundleWindowSpec] = []
    for center_index in range(frame_count):
        spec = build_identity_window_spec(
            frame_count, center_index, window_size,
            min_contributors=min_contributors, identity=identity)
        if spec is not None:
            windows.append(spec)
    return BundleWindowSchedule(
        frame_count=frame_count,
        window_size=window_size,
        windows=tuple(windows),
    )


def build_identity_window_spec(
    frame_count: int,
    center_index: int,
    window_size: int,
    *,
    min_contributors: int = 1,
    identity: NDArray[np.float64] | None = None,
) -> BundleWindowSpec | None:
    """Build one shrinking-edge identity window for streaming execution."""
    if not 0 <= center_index < frame_count:
        raise ValueError("center_index must be inside frame_count")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not 1 <= min_contributors <= window_size:
        raise ValueError("min_contributors must be in [1, window_size]")
    radius = window_size // 2
    start = max(0, center_index - radius)
    stop = min(frame_count, center_index + radius + 1)
    source_indices = tuple(range(start, stop))
    if len(source_indices) < min_contributors:
        return None
    if identity is None:
        identity = np.eye(3, dtype=np.float64)
        identity.setflags(write=False)
    return BundleWindowSpec(
        center_index=center_index,
        sources=tuple(BundleWindowSource(
            index=source_index,
            rotation_center_to_source=identity,
        ) for source_index in source_indices),
    )
