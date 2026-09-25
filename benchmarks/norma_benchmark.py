"""Minimal single-case runner for the Norma alignment benchmark."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from hoshicore.component.exif import read_exif_data
from hoshicore.component.image_io import load_img, save_img
from hoshicore.component.norma import (
    CameraInitializationPolicy,
    FisheyeCameraModel,
    StarDetectionCache,
    build_camera_candidate,
    lens_type_from_exif,
    to_gray_f64,
)
from hoshicore.component.norma.frame_align import (
    BOOTSTRAP_MATCHING_PATHS,
    DEFAULT_BOOTSTRAP_SCALES,
    DEFAULT_MATCHING_PATH,
    MATCHING_PATH_ASTERISM,
    solve_staged_alignment,
)
from hoshicore.component.norma.matching import evaluate_rotation


LENS_TYPES = {"perspective", "fisheye", "ideal"}
ALIGNMENT_OPTIONS = {
    "matching_path", "same_camera", "bootstrap_scales", "guided_refine",
    "guided_refine_radius_px", "median_threshold_ratio", "remap_map_scale",
    "lens_type", "ref_lens_type", "src_lens_type", "focal_length_mm",
    "crop_factor", "ref_focal_equiv_mm", "src_focal_equiv_mm",
    "fallback_focal_equiv_mm", "ref_fallback_focal_equiv_mm",
    "src_fallback_focal_equiv_mm", "optimize_focal", "ref_optimize_focal",
    "src_optimize_focal", "optimize_distortion",
    "ref_optimize_distortion", "src_optimize_distortion",
    "optimize_principal_point", "ref_optimize_principal_point",
    "src_optimize_principal_point", "distortion", "ref_init_distortion",
    "src_init_distortion", "allow_large_principal_point_offset",
    "ref_allow_large_principal_point_offset",
    "src_allow_large_principal_point_offset", "ref_exif_json", "src_exif_json",
}
EVALUATION_OPTIONS = {
    "mask", "reference_mask", "source_mask", "mask_erode_px", "tile_size",
    "tile_stride", "min_tile_mask_ratio", "min_phase_response",
}


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _read_exif(path: Path, override: Path | None) -> dict[str, str] | None:
    if override is not None:
        payload = json.loads(override.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"EXIF override must be an object: {override}")
        return {str(key): str(value) for key, value in payload.items()}
    try:
        data = read_exif_data(str(path))
    except Exception as exc:
        logger.warning("Failed to read EXIF from {}: {}", path, exc)
        return None
    return data.exif if data is not None else None


def _lens_policy(lens_type: str | None) -> str | None:
    if lens_type is None:
        return None
    if lens_type not in LENS_TYPES:
        raise ValueError(f"Unsupported lens type: {lens_type!r}")
    return None if lens_type == "perspective" else lens_type


def _camera_policy(config: dict[str, Any], prefix: str,
                   lens_type: str | None,
                   exif_tags: dict[str, str] | None
                   ) -> CameraInitializationPolicy:
    def setting(name: str) -> Any:
        return config.get(f"{prefix}_{name}", config.get(name))

    fallback = setting("fallback_focal_equiv_mm")
    return CameraInitializationPolicy(
        lens_type=_lens_policy(lens_type or lens_type_from_exif(exif_tags)),
        fallback_focal_equiv_mm=(20.0 if fallback is None else float(fallback)),
        optimize_focal=setting("optimize_focal"),
        optimize_distortion=setting("optimize_distortion"),
        optimize_principal_point=setting("optimize_principal_point"),
        allow_large_principal_point_offset=bool(
            setting("allow_large_principal_point_offset") or False),
    )


def _match_residuals(match: Any, alignment: Any) -> dict[str, Any]:
    ref_vectors = alignment.ref_camera.unproject(match.ref_pts)
    predicted = alignment.src_camera.project(
        (alignment.rotation_ref_to_src @ ref_vectors.T).T)
    finite = np.all(np.isfinite(predicted), axis=1)
    values = np.linalg.norm(predicted[finite] - match.src_pts[finite], axis=1)
    if not len(values):
        return {"count": 0, "median_px": None, "p90_px": None}
    return {
        "count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
    }


def _load_mask(path: Path | None, shape: tuple[int, ...]) -> np.ndarray:
    if path is None:
        return np.full(shape[:2], 255, dtype=np.uint8)
    mask = load_img(str(path))
    if mask is None:
        raise FileNotFoundError(f"Failed to load evaluation mask: {path}")
    if mask.shape[:2] != shape[:2]:
        raise ValueError(
            f"Evaluation mask shape mismatch: {mask.shape[:2]} != {shape[:2]}")
    if mask.ndim == 3:
        mask = np.max(mask, axis=2)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _local_normalize(image: np.ndarray, sigma: float = 8.0
                     ) -> np.ndarray | None:
    gray = to_gray_f64(image).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    centered = gray - mean
    variance = cv2.GaussianBlur(centered * centered, (0, 0),
                                sigmaX=sigma, sigmaY=sigma)
    scale = np.sqrt(np.maximum(variance, 0.0) + 1e-6)
    normalized = np.clip(centered / scale, -6.0, 6.0)
    if not np.all(np.isfinite(normalized)) or np.std(normalized) <= 1e-6:
        return None
    return normalized.astype(np.float32, copy=False)


def _evaluate_remap(reference: np.ndarray, aligned: np.ndarray,
                    valid_mask: np.ndarray,
                    config: dict[str, Any]) -> dict[str, Any]:
    height, width = valid_mask.shape
    tile_size = int(config.get("tile_size", 512))
    tile_stride = int(config.get("tile_stride", tile_size))
    min_mask_ratio = float(config.get("min_tile_mask_ratio", 0.95))
    min_phase_response = float(config.get("min_phase_response", 0.05))
    if tile_size < 32 or tile_stride < 1:
        raise ValueError("tile_size must be >= 32 and tile_stride must be >= 1")
    if not 0.0 <= min_mask_ratio <= 1.0:
        raise ValueError("min_tile_mask_ratio must be between 0 and 1")

    shifts: list[float] = []
    coverage = np.zeros((height, width), dtype=bool)
    window = cv2.createHanningWindow((tile_size, tile_size), cv2.CV_32F)
    for y0 in range(0, max(height - tile_size + 1, 0), tile_stride):
        for x0 in range(0, max(width - tile_size + 1, 0), tile_stride):
            ys = slice(y0, y0 + tile_size)
            xs = slice(x0, x0 + tile_size)
            tile_mask = valid_mask[ys, xs] > 0
            if float(np.mean(tile_mask)) < min_mask_ratio:
                continue
            ref_tile = _local_normalize(reference[ys, xs])
            src_tile = _local_normalize(aligned[ys, xs])
            if ref_tile is None or src_tile is None:
                continue
            (shift_x, shift_y), response = cv2.phaseCorrelate(
                ref_tile, src_tile, window)
            shift = float(np.hypot(shift_x, shift_y))
            if (np.isfinite(response) and response >= min_phase_response
                    and np.isfinite(shift)):
                shifts.append(shift)
                coverage[ys, xs] |= tile_mask

    valid_pixels = int(np.count_nonzero(valid_mask))
    evaluated_coverage = float(
        np.count_nonzero(coverage) / max(valid_pixels, 1))
    return {
        "status": "ok" if shifts else "insufficient_support",
        "evaluated_tiles": len(shifts),
        "evaluated_tile_coverage": evaluated_coverage,
        "residual_shift_p90_px": (
            float(np.percentile(shifts, 90)) if shifts else None),
    }


def _expectation_checks(expected: dict[str, Any] | None,
                        result: dict[str, Any]) -> list[dict[str, Any]]:
    if not expected:
        return []
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "passed": bool(passed)})

    if "success" in expected:
        add("success", result["success"],
            result["success"] == bool(expected["success"]))
    if not result["success"]:
        return checks

    matching = result["matching"]
    remap = result["remap_validation"]
    limits = {
        "min_final_pairs": (matching["final_pairs"], lambda a, e: a >= e),
        "min_coverage_ratio": (
            matching["coverage_ratio"], lambda a, e: a >= e),
        "min_outer_pairs": (matching["outer_pairs"], lambda a, e: a >= e),
        "max_p90_px": (matching["p90_px"], lambda a, e: a <= e),
        "max_residual_shift_p90_px": (
            remap.get("residual_shift_p90_px"), lambda a, e: a <= e),
        "min_evaluated_tiles": (
            remap.get("evaluated_tiles"), lambda a, e: a >= e),
        "min_evaluated_tile_coverage": (
            remap.get("evaluated_tile_coverage"), lambda a, e: a >= e),
    }
    remap_limits = {
        "max_residual_shift_p90_px",
        "min_evaluated_tiles",
        "min_evaluated_tile_coverage",
    }
    if remap_limits.intersection(expected):
        add("remap_status", remap.get("status"), remap.get("status") == "ok")
    for name, (actual, compare) in limits.items():
        if name in expected:
            add(name, actual, actual is not None and
                compare(float(actual), float(expected[name])))
    return checks


def _case_config(case: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    config = {**defaults, **case.get("alignment", {})}
    unknown = set(config) - ALIGNMENT_OPTIONS
    if unknown:
        raise ValueError(f"Unsupported alignment options: {sorted(unknown)}")
    config.setdefault("matching_path", DEFAULT_MATCHING_PATH)
    config.setdefault("guided_refine", False)
    config.setdefault("bootstrap_scales", DEFAULT_BOOTSTRAP_SCALES)
    matching_path = str(config["matching_path"])
    if matching_path not in BOOTSTRAP_MATCHING_PATHS:
        raise ValueError(
            "Norma benchmark supports only solver bootstrap paths: "
            f"{BOOTSTRAP_MATCHING_PATHS}; got {matching_path!r}")
    config["matching_path"] = matching_path
    return config


def run_case(case: dict[str, Any], defaults: dict[str, Any], base_dir: Path,
             output_dir: Path, write_remap: bool, random_seed: int = 0
             ) -> dict[str, Any]:
    started = time.perf_counter()
    case_id = str(case.get("id", "<missing-id>"))
    result: dict[str, Any] = {
        "id": case_id,
        "labels": case.get("labels", []),
        "random_seed": random_seed,
        "expected": case.get("expected"),
        "success": False,
    }
    try:
        config = _case_config(case, defaults)
        reference_path = _resolve_path(str(case["reference"]), base_dir)
        source_path = _resolve_path(str(case["source"]), base_dir)
        assert reference_path is not None and source_path is not None
        result.update({
            "reference": str(reference_path),
            "source": str(source_path),
            "alignment": config,
        })

        t0 = time.perf_counter()
        reference = load_img(str(reference_path))
        source = load_img(str(source_path))
        if reference is None or source is None:
            raise FileNotFoundError("Failed to load one or both images")
        load_seconds = time.perf_counter() - t0

        shared_lens = config.get("lens_type")
        ref_lens = config.get("ref_lens_type", shared_lens)
        src_lens = config.get("src_lens_type", shared_lens)
        crop_factor = float(config.get("crop_factor") or 1.0)
        focal = config.get("focal_length_mm")
        focal_equiv = float(focal) * crop_factor if focal is not None else None
        ref_focal = config.get("ref_focal_equiv_mm", focal_equiv)
        src_focal = config.get("src_focal_equiv_mm", focal_equiv)
        ref_exif = _resolve_path(config.get("ref_exif_json"), base_dir)
        src_exif = _resolve_path(config.get("src_exif_json"), base_dir)
        ref_tags = _read_exif(reference_path, ref_exif)
        src_tags = _read_exif(source_path, src_exif)

        t0 = time.perf_counter()
        ref_candidate = build_camera_candidate(
            ref_tags, reference.shape, "distortion",
            init_distortion=config.get(
                "ref_init_distortion", config.get("distortion")),
            focal_equiv_mm=ref_focal,
            init_policy=_camera_policy(config, "ref", ref_lens, ref_tags))
        src_candidate = build_camera_candidate(
            src_tags, source.shape, "distortion",
            init_distortion=config.get(
                "src_init_distortion", config.get("distortion")),
            focal_equiv_mm=src_focal,
            init_policy=_camera_policy(config, "src", src_lens, src_tags))
        same_camera = bool(config.get("same_camera", False))
        if (same_camera and isinstance(ref_candidate.camera, FisheyeCameraModel)
                != isinstance(src_candidate.camera, FisheyeCameraModel)):
            logger.warning(
                "Benchmark case {} has mixed projections; forcing "
                "same_camera=False", case_id)
            same_camera = False
            config["same_camera"] = False
        camera_init_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref_detection = StarDetectionCache.from_image(
            reference,
            median_threshold_ratio=float(
                config.get("median_threshold_ratio", 1.0)))
        src_detection = StarDetectionCache.from_image(
            source,
            median_threshold_ratio=float(
                config.get("median_threshold_ratio", 1.0)))
        ref_bootstrap_stars = ref_detection.pywt_stars
        src_bootstrap_stars = src_detection.pywt_stars
        bootstrap_detection_seconds = time.perf_counter() - t0
        ref_refine_stars = src_refine_stars = None
        refine_detection_seconds = None
        if bool(config["guided_refine"]):
            t0 = time.perf_counter()
            ref_refine_stars = ref_detection.median_stars
            src_refine_stars = src_detection.median_stars
            refine_detection_seconds = time.perf_counter() - t0

        solved = solve_staged_alignment(
            ref_bootstrap_stars,
            src_bootstrap_stars,
            ref_candidate,
            src_candidate,
            bootstrap_scales=tuple(float(value) for value in
                                   config["bootstrap_scales"]),
            same_camera=same_camera,
            guided_refine=bool(config["guided_refine"]),
            guided_refine_radius_px=float(
                config.get("guided_refine_radius_px", 8.0)),
            use_asterism_bootstrap=(
                config["matching_path"] == MATCHING_PATH_ASTERISM),
            ref_refine_stars=ref_refine_stars,
            src_refine_stars=src_refine_stars,
            random_seed=random_seed,
        )

        final_match = solved.final_match
        final_alignment = solved.final_alignment
        if (solved.refine_status == "applied"
                and solved.refine_ref is not None
                and solved.refine_src is not None):
            ref_geo = solved.refine_ref
            src_geo = solved.refine_src
        else:
            ref_geo = solved.bootstrap_ref
            src_geo = solved.bootstrap_src
        diagnostics = evaluate_rotation(
            ref_geo.positions,
            src_geo.positions,
            final_match.pair_idx,
            final_alignment.ref_camera.unproject(ref_geo.positions),
            final_alignment.src_camera.unproject(src_geo.positions),
            final_alignment.rotation_ref_to_src,
        )
        residuals = _match_residuals(final_match, final_alignment)

        evaluation = dict(case.get("evaluation", {}))
        for key in ("mask", "reference_mask", "source_mask"):
            if key in case and key not in evaluation:
                evaluation[key] = case[key]
        unknown_evaluation = set(evaluation) - EVALUATION_OPTIONS
        if unknown_evaluation:
            raise ValueError(
                "Unsupported evaluation options: "
                f"{sorted(unknown_evaluation)}")
        common_mask = evaluation.get("mask")
        ref_mask_path = _resolve_path(
            evaluation.get("reference_mask", common_mask), base_dir)
        src_mask_path = _resolve_path(
            evaluation.get("source_mask", common_mask), base_dir)
        evaluate_image = ref_mask_path is not None or src_mask_path is not None
        aligned = None
        remap_seconds = None
        evaluation_seconds = None
        remap_validation: dict[str, Any] = {
            "status": "not_evaluated_no_mask"
        }
        if write_remap or evaluate_image:
            t0 = time.perf_counter()
            aligned = final_alignment.ref_camera.project_image_from_camera(
                final_alignment.src_camera,
                source,
                (reference.shape[1], reference.shape[0]),
                rotation_dst_to_src=final_alignment.rotation_ref_to_src,
                map_scale=float(config.get("remap_map_scale", 0.5)),
            )
            remap_seconds = time.perf_counter() - t0

        valid_mask = None
        if evaluate_image:
            assert aligned is not None
            t0 = time.perf_counter()
            reference_mask = _load_mask(ref_mask_path, reference.shape)
            source_mask = _load_mask(src_mask_path, source.shape)
            remapped_source_mask = (
                final_alignment.ref_camera.project_image_from_camera(
                    final_alignment.src_camera,
                    source_mask,
                    (reference.shape[1], reference.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                    rotation_dst_to_src=final_alignment.rotation_ref_to_src,
                    map_scale=float(config.get("remap_map_scale", 0.5)),
                ))
            valid_mask = np.where(
                (reference_mask > 0) & (remapped_source_mask >= 254),
                255, 0).astype(np.uint8)
            erode_px = int(evaluation.get("mask_erode_px", 2))
            if erode_px < 0:
                raise ValueError("mask_erode_px must be >= 0")
            if erode_px:
                size = erode_px * 2 + 1
                valid_mask = cv2.erode(
                    valid_mask, np.ones((size, size), dtype=np.uint8))
            remap_validation = _evaluate_remap(
                reference, aligned, valid_mask, evaluation)
            evaluation_seconds = time.perf_counter() - t0

        if write_remap:
            assert aligned is not None
            case_dir = output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            save_img(str(case_dir / "src_aligned.tif"), aligned)
            save_img(str(case_dir / "tgt_reference.tif"), reference)
            if valid_mask is not None:
                save_img(str(case_dir / "evaluation_mask.png"), valid_mask)

        result.update({
            "success": True,
            "matching_path": config["matching_path"],
            "refine_status": solved.refine_status,
            "star_counts": {
                "reference": int(len(ref_geo.positions)),
                "source": int(len(src_geo.positions)),
            },
            "matching": {
                "final_pairs": int(len(final_match.pair_idx)),
                "coverage_ratio": diagnostics.coverage_ratio,
                "outer_pairs": diagnostics.outer_inlier_count,
                **residuals,
            },
            "remap_validation": remap_validation,
            "optimized_focal_mm": {
                "reference": float(
                    final_alignment.ref_camera.intrinsics.focal_length_mm),
                "source": float(
                    final_alignment.src_camera.intrinsics.focal_length_mm),
            },
            "optimized_camera": {
                "reference": {
                    "principal_point_px": list(
                        final_alignment.ref_camera.intrinsics.principal_point_px),
                    "distortion": (
                        final_alignment.ref_camera.distortion.to_cv2()
                        .astype(float).tolist()),
                },
                "source": {
                    "principal_point_px": list(
                        final_alignment.src_camera.intrinsics.principal_point_px),
                    "distortion": (
                        final_alignment.src_camera.distortion.to_cv2()
                        .astype(float).tolist()),
                },
            },
            "timing_seconds": {
                "load": load_seconds,
                "camera_init": camera_init_seconds,
                "bootstrap_detection": bootstrap_detection_seconds,
                "refine_detection": refine_detection_seconds,
                **solved.timings,
                "remap": remap_seconds,
                "remap_evaluation": evaluation_seconds,
            },
        })
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    result.setdefault("timing_seconds", {})["total"] = (
        time.perf_counter() - started)
    result["expectation_checks"] = _expectation_checks(
        case.get("expected"), result)
    result["expectations_passed"] = (
        all(check["passed"] for check in result["expectation_checks"])
        if result["expectation_checks"] else None)
    return result
