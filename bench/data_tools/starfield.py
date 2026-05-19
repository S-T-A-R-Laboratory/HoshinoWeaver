"""Synthetic starfield helpers for alignment benchmarks."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _add_gaussian_star(
    canvas: np.ndarray,
    *,
    cx: float,
    cy: float,
    sigma: float,
    amplitude: float,
) -> None:
    radius = max(2, int(math.ceil(3 * sigma)))
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(canvas.shape[1], int(math.ceil(cx + radius + 1)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(canvas.shape[0], int(math.ceil(cy + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = amplitude * np.exp(-dist2 / (2.0 * sigma * sigma))
    canvas[y0:y1, x0:x1] += patch.astype(canvas.dtype, copy=False)


def make_starfield_base(
    *,
    height: int,
    width: int,
    stars: int,
    seed: int,
    dtype: np.dtype,
    channels: int = 3,
    background_level: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), background_level, dtype=np.float32)

    for _ in range(stars):
        cx = rng.uniform(0, width - 1)
        cy = rng.uniform(0, height - 1)
        sigma = rng.uniform(0.6, 1.8)
        amplitude = rng.uniform(0.35, 1.0)
        _add_gaussian_star(
            canvas,
            cx=cx,
            cy=cy,
            sigma=sigma,
            amplitude=amplitude,
        )

    canvas = np.clip(canvas, 0.0, 1.0)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        scale = float(np.iinfo(dtype).max)
        base_img = np.round(canvas * scale).astype(dtype)
    else:
        base_img = canvas.astype(dtype)
    if channels == 1:
        return base_img
    return np.repeat(base_img[..., None], channels, axis=2)


def apply_starfield_transform(
    frame: np.ndarray,
    *,
    shift_x: float,
    shift_y: float,
    rotation_deg: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def generate_starfield_frames(
    *,
    frames: int,
    height: int,
    width: int,
    stars: int,
    seed: int,
    dtype: np.dtype = np.uint8,
    channels: int = 3,
    max_shift: float = 12.0,
    max_rotation_deg: float = 0.8,
    noise_sigma: float = 1.5,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    base = make_starfield_base(
        height=height,
        width=width,
        stars=stars,
        seed=seed,
        dtype=dtype,
        channels=channels,
    )

    result: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for idx in range(frames):
        if idx == 0:
            shift_x = 0.0
            shift_y = 0.0
            rotation_deg = 0.0
        else:
            shift_x = float(rng.uniform(-max_shift, max_shift))
            shift_y = float(rng.uniform(-max_shift, max_shift))
            rotation_deg = float(
                rng.uniform(-max_rotation_deg, max_rotation_deg))

        frame = apply_starfield_transform(
            base,
            shift_x=shift_x,
            shift_y=shift_y,
            rotation_deg=rotation_deg,
        )
        if noise_sigma > 0:
            noise = rng.normal(0.0, noise_sigma, size=frame.shape)
            if np.issubdtype(dtype, np.integer):
                max_value = np.iinfo(dtype).max
            else:
                max_value = 1.0
            frame = np.clip(frame.astype(np.float32) + noise, 0,
                            max_value).astype(dtype)

        result.append(frame)
        meta.append(
            {
                "index": idx,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "rotation_deg": rotation_deg,
            })

    return result, meta
