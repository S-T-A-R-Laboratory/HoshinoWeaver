"""Alignment pipeline benchmark with stage-level profiling."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from bench.common import (
    collect_env_info,
    print_or_save_report,
    run_benchmark,
)
from bench.data_tools.starfield import generate_starfield_frames
from hoshicore.component.norma.alignment import match_star_pairs, optimize_alignment
from hoshicore.component.norma.cache import GeometryView, StarDetectionCache
from hoshicore.component.norma.detection import _wavelet_dec_rec
from hoshicore.component.norma.frame_align import (
    align_frame_camera_model,
    align_frame_homography,
    make_geometry,
    to_gray_f64,
)
from hoshicore.component.norma.types import CameraModel, Distortion, Intrinsics


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CASE_NAMES = [
    "detect_stream",
    "detect_prepare_stream",
    "detect_wavelet_stream",
    "detect_extract_stream",
    "features_stream",
    "geometry_stream",
    "match_stream",
    "warp_stream",
    "homography_pipeline",
    "optimization_stream",
    "remap_stream",
    "camera_model_pipeline",
]


@dataclasses.dataclass
class DetectPayload:
    img_gray: np.ndarray
    img_blr: np.ndarray
    mask: np.ndarray
    resize_factor: float


def _prepare_detect_payload(frame: np.ndarray,
                            *,
                            resize_length: int = 10000,
                            gaussian_ksize: int = 9,
                            sigma: float = 2.0) -> DetectPayload:
    img_gray = to_gray_f64(frame)
    img_shape = img_gray.shape

    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize),
                               sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr = (img_blr - img_blr_mean) / img_blr_range

    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    tmp_mask = cv2.resize(img_gray, None, fx=resize_factor, fy=resize_factor)
    tmp_mask_10percent = np.percentile(tmp_mask, 10)
    tmp_mask = (tmp_mask < min(tmp_mask_10percent, 0.15)).astype(
        np.uint8) * 255

    dilate_size = max(1, int(max(img_shape) * 0.003 * resize_factor))
    tmp_mask = 255 - cv2.dilate(
        tmp_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_size, dilate_size)))
    tmp_mask = cv2.resize(tmp_mask, (img_shape[1], img_shape[0]))
    mask = tmp_mask > 127
    if np.sum(mask) * 100.0 / np.prod(mask.shape) < 50:
        mask = np.ones(tmp_mask.shape, dtype=bool)

    return DetectPayload(
        img_gray=img_gray,
        img_blr=img_blr,
        mask=mask,
        resize_factor=resize_factor,
    )


def _extract_detect_features(payload: DetectPayload) -> None:
    img_rec = _wavelet_dec_rec(
        payload.img_blr, resize_factor=payload.resize_factor) * payload.mask
    bw = ((img_rec > np.percentile(img_rec[payload.mask], 99.5)) *
          payload.mask).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) > 5]
    if not contours:
        return

    elps = [cv2.fitEllipse(contour) for contour in contours]
    areas = np.array([
        cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours
    ])
    eccentricities = np.sqrt(
        np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))
    _ = eccentricities

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for i, contour in enumerate(contours):
        cv2.drawContours(mask_img, contours, i, 255, -1)
        rect = cv2.boundingRect(contour)
        val = cv2.mean(
            img_rec[rect[1]:rect[1] + rect[3] + 1,
                    rect[0]:rect[0] + rect[2] + 1],
            mask_img[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
        mask_img[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
        intensities[i] = val[0]
    _ = intensities


def load_frames_from_dir(input_dir: str, frames: int) -> list[np.ndarray]:
    root = Path(input_dir)
    files = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if len(files) < frames:
        raise ValueError(
            f"Not enough images in {input_dir}: need {frames}, got {len(files)}")

    result: list[np.ndarray] = []
    for path in files[:frames]:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        result.append(frame)
    return result


def prepare_alignment_frames(
    *,
    frames: int,
    height: int,
    width: int,
    channels: int,
    stars: int,
    seed: int,
    max_shift: float,
    max_rotation_deg: float,
    noise_sigma: float,
    input_dir: str | None,
    input_mode: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if input_mode == "images":
        if input_dir is None:
            raise ValueError("--input-mode images requires --input-dir")
        loaded = load_frames_from_dir(input_dir, frames)
        return loaded, {
            "mode": "images",
            "input_dir": input_dir,
            "resolved_frames": len(loaded),
            "resolved_shape": list(loaded[0].shape),
            "resolved_dtype": str(loaded[0].dtype),
        }

    generated, meta = generate_starfield_frames(
        frames=frames,
        height=height,
        width=width,
        stars=stars,
        seed=seed,
        channels=channels,
        max_shift=max_shift,
        max_rotation_deg=max_rotation_deg,
        noise_sigma=noise_sigma,
    )
    return generated, {
        "mode": "synthetic_starfield",
        "resolved_frames": len(generated),
        "resolved_shape": list(generated[0].shape),
        "resolved_dtype": str(generated[0].dtype),
        "stars": stars,
        "max_shift": max_shift,
        "max_rotation_deg": max_rotation_deg,
        "noise_sigma": noise_sigma,
        "transform_preview": meta[:min(3, len(meta))],
    }


def bench_geometry_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        geo = make_geometry(frame)
        _ = geo.unit_vectors


def bench_detect_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        gray = to_gray_f64(frame)
        cache = StarDetectionCache(gray)
        _ = cache.detected_stars


def bench_detect_prepare_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        _ = _prepare_detect_payload(frame)


def bench_detect_wavelet_stream(payloads: list[DetectPayload]) -> None:
    for payload in payloads:
        _ = _wavelet_dec_rec(
            payload.img_blr, resize_factor=payload.resize_factor)


def bench_detect_extract_stream(payloads: list[DetectPayload]) -> None:
    for payload in payloads:
        _extract_detect_features(payload)


def bench_features_stream(frames: list[np.ndarray]) -> None:
    for frame in frames:
        gray = to_gray_f64(frame)
        cache = StarDetectionCache(gray)
        _ = GeometryView.from_flat_projection(cache).features


def precompute_geometries(frames: list[np.ndarray]):
    ref = make_geometry(frames[0])
    rest = [make_geometry(frame) for frame in frames[1:]]
    return ref, rest


def build_synthetic_camera(frame: np.ndarray) -> CameraModel:
    height, width = frame.shape[:2]
    intrinsics = Intrinsics(
        focal_length_mm=35.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        image_width_px=width,
        image_height_px=height,
    )
    return CameraModel(intrinsics=intrinsics, distortion=Distortion())


def bench_match_stream(frames: list[np.ndarray]) -> None:
    ref_geo, src_geos = precompute_geometries(frames)
    for src_geo in src_geos:
        _ = match_star_pairs(
            ref_geo.unit_vectors,
            src_geo.unit_vectors,
            ref_geo.volumes,
            src_geo.volumes,
            ref_geo.positions,
            src_geo.positions,
        )


def bench_warp_stream(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_geo, src_geos = precompute_geometries(frames)
    h, w = ref_arr.shape[:2]
    for frame, src_geo in zip(frames[1:], src_geos):
        match = match_star_pairs(
            ref_geo.unit_vectors,
            src_geo.unit_vectors,
            ref_geo.volumes,
            src_geo.volumes,
            ref_geo.positions,
            src_geo.positions,
        )
        H = np.linalg.inv(match.init_homography)
        _ = cv2.warpPerspective(frame, H, (w, h))


def bench_optimization_stream(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_geo = make_geometry(ref_arr)
    ref_camera = build_synthetic_camera(ref_arr)
    for frame in frames[1:]:
        src_geo = make_geometry(frame)
        src_camera = build_synthetic_camera(frame)
        match = match_star_pairs(
            ref_geo.unit_vectors,
            src_geo.unit_vectors,
            ref_geo.volumes,
            src_geo.volumes,
            ref_geo.positions,
            src_geo.positions,
        )
        _ = optimize_alignment(
            match,
            ref_camera,
            src_camera,
            same_camera=True,
        )


def bench_homography_pipeline(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_geo = make_geometry(ref_arr)
    for frame in frames[1:]:
        _ = align_frame_homography(frame, ref_geo, ref_arr)


def bench_camera_model_pipeline(frames: list[np.ndarray]) -> None:
    ref_arr = frames[0]
    ref_geo = make_geometry(ref_arr)
    ref_camera = build_synthetic_camera(ref_arr)
    for frame in frames[1:]:
        src_camera = build_synthetic_camera(frame)
        _ = align_frame_camera_model(
            frame,
            ref_geo,
            ref_arr,
            ref_camera,
            src_camera,
            same_camera=True,
        )


def _prepare_remap_payloads(frames: list[np.ndarray]):
    ref_arr = frames[0]
    ref_geo = make_geometry(ref_arr)
    ref_camera = build_synthetic_camera(ref_arr)
    payloads = []
    for frame in frames[1:]:
        src_geo = make_geometry(frame)
        src_camera = build_synthetic_camera(frame)
        match = match_star_pairs(
            ref_geo.unit_vectors,
            src_geo.unit_vectors,
            ref_geo.volumes,
            src_geo.volumes,
            ref_geo.positions,
            src_geo.positions,
        )
        result = optimize_alignment(
            match,
            ref_camera,
            src_camera,
            same_camera=True,
        )
        payloads.append((frame, result.camera2_refined, result.camera1_refined,
                         (ref_arr.shape[1], ref_arr.shape[0])))
    return payloads


def bench_remap_stream(payloads) -> None:
    for frame, src_camera, dst_camera, output_size in payloads:
        _ = dst_camera.project_image_from_camera(src_camera, frame, output_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--stars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-shift", type=float, default=12.0)
    parser.add_argument("--max-rotation-deg", type=float, default=0.8)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-mode",
                        choices=["synthetic", "images"],
                        default="synthetic")
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=list(CASE_NAMES))
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    frames, input_source = prepare_alignment_frames(
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
        stars=args.stars,
        seed=args.seed,
        max_shift=args.max_shift,
        max_rotation_deg=args.max_rotation_deg,
        noise_sigma=args.noise_sigma,
        input_dir=args.input_dir,
        input_mode=args.input_mode,
    )

    unknown_cases = [case for case in args.cases if case not in CASE_NAMES]
    if unknown_cases:
        raise ValueError(
            f"Unknown alignment benchmark case(s): {unknown_cases}. "
            f"Available: {list(CASE_NAMES)}")

    detect_payloads = None
    if any(case in args.cases
           for case in ("detect_wavelet_stream", "detect_extract_stream")):
        detect_payloads = [_prepare_detect_payload(frame) for frame in frames]

    remap_payloads = None
    if "remap_stream" in args.cases:
        remap_payloads = _prepare_remap_payloads(frames)

    runners: dict[str, Any] = {
        "detect_stream": lambda: bench_detect_stream(frames),
        "detect_prepare_stream": lambda: bench_detect_prepare_stream(frames),
        "detect_wavelet_stream": lambda: bench_detect_wavelet_stream(
            detect_payloads),
        "detect_extract_stream": lambda: bench_detect_extract_stream(
            detect_payloads),
        "features_stream": lambda: bench_features_stream(frames),
        "geometry_stream": lambda: bench_geometry_stream(frames),
        "match_stream": lambda: bench_match_stream(frames),
        "warp_stream": lambda: bench_warp_stream(frames),
        "homography_pipeline": lambda: bench_homography_pipeline(frames),
        "optimization_stream": lambda: bench_optimization_stream(frames),
        "remap_stream": lambda: bench_remap_stream(remap_payloads),
        "camera_model_pipeline": lambda: bench_camera_model_pipeline(frames),
    }

    cases = {
        case_name: run_benchmark(
            runners[case_name],
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for case_name in args.cases
    }

    report = {
        "suite": "alignment",
        "env": collect_env_info(),
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "channels": args.channels,
            "stars": args.stars,
            "seed": args.seed,
            "max_shift": args.max_shift,
            "max_rotation_deg": args.max_rotation_deg,
            "noise_sigma": args.noise_sigma,
            "input_dir": args.input_dir,
            "input_mode": args.input_mode,
            "cases": args.cases,
            "log_level": args.log_level,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "input_source": input_source,
        "results": cases,
    }
    print_or_save_report(report, args.output_json)


if __name__ == "__main__":
    main()
