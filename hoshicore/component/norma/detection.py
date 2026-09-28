"""Star point detection from images."""
import dataclasses
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from hoshicore._custom_op import median_star_mask, wavelet_dec_rec
from hoshicore._custom_op._dispatch import (CustomOpResourceExhaustedError,
                                            CustomOpUnavailableError)
from hoshicore._custom_op.ops.detection import (
    GraySource, StarDetectCapacityError, star_detect_fused_pixel_components)

MIN_STAR_AREA = 10
STAR_FILTER_PERCENTILE = 10
FULL_GPU_COMPONENT_FILTER_PERCENTILE = 22.5
NATIVE_CONTOUR_MISMATCH_RATIO = 0.001
NATIVE_CONTOUR_MIN_MISMATCH_BUDGET = 5
NATIVE_CONTOUR_MAX_DISTANCE_PX = 1.0


@dataclasses.dataclass
class DetectedStars:
    positions: NDArray[np.float64]
    volumes: NDArray[np.float64]
    intensities: Optional[NDArray[np.float64]] = None


@dataclasses.dataclass
class MedianDetectionDetails:
    detected_stars: DetectedStars
    response: NDArray[np.float32]
    threshold: float
    star_mask: NDArray[np.uint8]
    candidate_positions: NDArray[np.float64]
    areas: NDArray[np.float64]
    intensities: NDArray[np.float64]
    eccentricities: NDArray[np.float64]
    strict_valid_stars: NDArray[np.bool_]
    rescued_stars: NDArray[np.bool_]
    valid_stars: NDArray[np.bool_]
    area_percentile: float
    intensity_percentile: float


def _candidate_volumes_with_rescue(
    areas: NDArray[np.float64],
    intensities: NDArray[np.float64],
    eccentricities: NDArray[np.float64],
    rescued_stars: NDArray[np.bool_],
    strict_max_eccentricity: float,
) -> NDArray[np.float64]:
    volumes = areas * intensities
    if np.any(rescued_stars):
        quality = np.clip(
            (1.0 - eccentricities[rescued_stars]) /
            (1.0 - strict_max_eccentricity),
            0.15,
            1.0,
        )
        quality = np.nan_to_num(quality, nan=0.15, posinf=0.15, neginf=1.0)
        volumes = volumes.copy()
        volumes[rescued_stars] *= quality
    return volumes


class _NativeHybridGeometryMismatch(RuntimeError):
    """Native component summaries cannot be mapped to exact host contours."""

    pass


def _empty_detected_stars() -> DetectedStars:
    return DetectedStars(
        positions=np.empty((0, 2), dtype=np.float64),
        volumes=np.empty((0,), dtype=np.float64),
    )


def _normalize_gray(img_gray: NDArray) -> NDArray[np.float64]:
    if img_gray.dtype == np.float64:
        return img_gray
    if np.issubdtype(img_gray.dtype, np.integer):
        return img_gray.astype(np.float64) / np.iinfo(img_gray.dtype).max
    return img_gray.astype(np.float64)


def _wavelet_dec_rec(img_blr, resize_factor=0.25):
    return wavelet_dec_rec(img_blr, resize_factor=resize_factor)


def _bandpass_dog(img_blr: np.ndarray, resize_factor: float = 0.25) -> np.ndarray:
    h, w = img_blr.shape
    small = cv2.resize(img_blr,
                       None,
                       fx=resize_factor,
                       fy=resize_factor,
                       interpolation=cv2.INTER_AREA)
    diag_len = (h**2 + w**2)**(1 / 2)
    # 假设原图最大星点 20px in 7000px
    fine_width = diag_len * 0.0001
    coarse_width = diag_len * 0.001
    # 粗尺度去背景，细尺度去噪，差值保留星点
    coarse = cv2.GaussianBlur(small, (0, 0), sigmaX=coarse_width)
    fine = cv2.GaussianBlur(small, (0, 0), sigmaX=fine_width)
    dog = coarse - fine
    return cv2.resize(dog, (w, h), interpolation=cv2.INTER_LINEAR)


def _gray_u16_for_detection(img: np.ndarray) -> np.ndarray:
    """Convert a detection input to contiguous uint16 grayscale.

    Floating-point inputs must already use normalized intensity units. Values
    outside [0, 1] are rejected instead of being silently clipped.
    """
    image = np.asarray(img)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] != 3):
        raise ValueError("img must be a grayscale or three-channel image")

    if image.dtype == np.uint16:
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else image)
        return np.ascontiguousarray(gray)

    if image.dtype == np.uint8:
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else image)
        return np.ascontiguousarray(gray.astype(np.uint16) * np.uint16(257))

    if image.dtype.kind != "f":
        raise TypeError("img must have dtype uint8, uint16, or floating point")

    gray = image.astype(np.float32)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if not np.all(np.isfinite(gray)):
        raise ValueError("floating-point img must contain only finite values")
    if gray.size and (float(gray.min()) < 0.0 or float(gray.max()) > 1.0):
        raise ValueError("floating-point img must be normalized to [0, 1]")
    gray_u16 = np.rint(gray * np.float32(65535.0)).astype(np.uint16)
    return np.ascontiguousarray(gray_u16)


def _detect_starmask_by_threshold_details(
    img: np.ndarray,
    ksize: int = 13,
    med_algo: str = "median",
    threshold_ratio: int | float = 5,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the star mask, signed residual, and threshold used for it.

    Integer inputs are converted to a uint16 working image. Floating-point
    inputs must contain finite values in [0, 1] and are quantized to that same
    working representation. The returned residual is scaled back to normalized
    intensity units and may contain negative values.
    """
    started_at = perf_counter()
    if med_algo == "median" and ksize > 5:
        gray_u16 = _gray_u16_for_detection(img)
        star_mask, response, threshold = median_star_mask(
            gray_u16,
            median_ksize=ksize,
            threshold_ratio=threshold_ratio,
            open_ksize=open_ksize,
            dilate_ksize=dilate_ksize,
            mask=mask,
        )
        finished_at = perf_counter()
        logger.debug(
            "Star-mask timing: backend=median_star_mask ksize={} total={:.3f}s "
            "threshold={:.8f} foreground={}",
            ksize,
            finished_at - started_at,
            threshold,
            int(np.count_nonzero(star_mask)),
        )
        return star_mask, response, float(threshold)

    gray_u16 = _gray_u16_for_detection(img)
    gray = gray_u16.astype(np.float32) / np.float32(65535.0)
    gray_at = perf_counter()

    if mask is None:
        valid_mask = np.ones(gray.shape, dtype=bool)
    else:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != gray.shape:
            raise ValueError("mask shape must match the grayscale image")
        valid_mask = mask_arr > 0
        if not np.any(valid_mask):
            raise ValueError("mask must select at least one pixel")
    mask_at = perf_counter()

    background_backend = med_algo
    if med_algo == "median":
        if ksize <= 5:
            background_backend = "cv2.medianBlur"
            bg_u16 = cv2.medianBlur(gray_u16, ksize=ksize)
        else:
            raise AssertionError("large median kernels must use median_star_mask")
        # Promote both operands before subtraction. Subtracting uint16 arrays
        # directly would wrap negative residuals around to large positives.
        diff = (gray_u16.astype(np.float32) - bg_u16.astype(np.float32))
        diff /= np.float32(65535.0)
    elif med_algo == "mean":
        background_backend = "cv2.blur"
        bg = cv2.blur(gray, ksize=(ksize, ksize))
        diff = gray - bg
    else:
        raise NotImplementedError(f"Unknown med algo: {med_algo}.")
    background_at = perf_counter()
    logger.debug(
        "Star-mask background: backend={} ksize={} shape={} elapsed={:.3f}s",
        background_backend,
        ksize,
        gray.shape,
        background_at - mask_at,
    )

    diff_at = perf_counter()
    threshold = np.std(diff[valid_mask]) * threshold_ratio
    threshold_at = perf_counter()
    star_mask = np.logical_and(diff > threshold, valid_mask).astype(np.uint8)
    binary_at = perf_counter()

    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                      (open_ksize, open_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_OPEN, k)
    open_at = perf_counter()
    if dilate_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                      (dilate_ksize, dilate_ksize))
        star_mask = cv2.morphologyEx(star_mask, cv2.MORPH_DILATE, k)
    finished_at = perf_counter()
    logger.debug(
        "Star-mask timing: gray={:.3f}s mask={:.3f}s background={:.3f}s "
        "diff={:.3f}s threshold={:.3f}s binary={:.3f}s open={:.3f}s "
        "dilate={:.3f}s total={:.3f}s foreground={}",
        gray_at - started_at,
        mask_at - gray_at,
        background_at - mask_at,
        diff_at - background_at,
        threshold_at - diff_at,
        binary_at - threshold_at,
        open_at - binary_at,
        finished_at - open_at,
        finished_at - started_at,
        int(np.count_nonzero(star_mask)),
    )
    return star_mask, diff, float(threshold)


def detect_starmask_by_threshold_with_response(
    img: np.ndarray,
    ksize: int = 13,
    med_algo: str = "median",
    threshold_ratio: int | float = 5,
    open_ksize: int = 3,
    dilate_ksize: int = 0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a uint8 star mask and normalized float32 signed residual."""
    star_mask, response, _ = _detect_starmask_by_threshold_details(
        img,
        ksize=ksize,
        med_algo=med_algo,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        dilate_ksize=dilate_ksize,
        mask=mask,
    )
    return star_mask, response


def _find_star_contours(bw: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    contours, _ = cv2.findContours(
        bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if len(contour) > 5]


def _filter_star_candidates(
    positions: NDArray[np.float64],
    areas: NDArray[np.float64],
    intensities: NDArray[np.float64],
    eccentricities: NDArray[np.float64],
) -> DetectedStars:
    if len(positions) == 0:
        return _empty_detected_stars()

    area_percentile = np.percentile(areas, STAR_FILTER_PERCENTILE)
    intensity_percentile = np.percentile(
        intensities, STAR_FILTER_PERCENTILE)
    valid_stars = np.logical_and(
        areas > MIN_STAR_AREA,
        eccentricities < .8,
    )
    valid_stars = np.logical_and(
        valid_stars,
        np.logical_and(
            areas > area_percentile,
            intensities > intensity_percentile,
        ),
    )
    return DetectedStars(
        positions=positions[valid_stars],
        volumes=areas[valid_stars] * intensities[valid_stars],
    )


def _measure_contour_candidates(
    contours: list[NDArray[np.int32]],
    img_rec: NDArray[np.float64],
    bw: NDArray[np.uint8],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    positions = np.asarray([ellipse[0] for ellipse in ellipses])
    areas = np.asarray([
        cv2.contourArea(contour) + 0.5 * len(contour)
        for contour in contours
    ])
    eccentricities = np.sqrt(
        np.asarray([
            1 - (ellipse[1][0] / ellipse[1][1])**2
            for ellipse in ellipses
        ]))

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for index, contour in enumerate(contours):
        cv2.drawContours(mask_img, contours, index, 255, -1)
        x, y, width, height = cv2.boundingRect(contour)
        roi = np.s_[y:y + height + 1, x:x + width + 1]
        intensities[index] = cv2.mean(img_rec[roi], mask_img[roi])[0]
        mask_img[roi] = 0
    return positions, areas, intensities, eccentricities


def _measure_native_hybrid_contour_candidates(
    component_positions: NDArray[np.float64],
    component_intensities: NDArray[np.float64],
    binary_mask: NDArray[np.uint8],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Recover exact contour geometry and attach native component intensities."""
    contours = _find_star_contours(binary_mask)
    if not contours:
        empty = np.empty((0,), dtype=np.float64)
        return np.empty((0, 2), dtype=np.float64), empty, empty, empty

    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    positions = np.asarray([ellipse[0] for ellipse in ellipses])
    if len(component_positions) == 0:
        raise _NativeHybridGeometryMismatch(
            "Native detector returned no components for detected contours")

    nearest_distances, component_indices = cKDTree(component_positions).query(
        positions, k=1)
    component_counts = np.bincount(
        component_indices, minlength=len(component_positions))
    count_difference = abs(len(component_positions) - len(positions))
    duplicate_count = int(np.maximum(component_counts - 1, 0).sum())
    distant = nearest_distances > NATIVE_CONTOUR_MAX_DISTANCE_PX
    distant_count = int(np.count_nonzero(distant))
    mismatch_budget = max(
        NATIVE_CONTOUR_MIN_MISMATCH_BUDGET,
        int(np.ceil(NATIVE_CONTOUR_MISMATCH_RATIO * len(positions))),
    )
    if (count_difference > mismatch_budget
            or duplicate_count > mismatch_budget
            or distant_count > mismatch_budget):
        raise _NativeHybridGeometryMismatch(
            "Native/contour mapping exceeds tolerance "
            f"(count_difference={count_difference}, "
            f"duplicate_count={duplicate_count}, "
            f"distant_count={distant_count}, budget={mismatch_budget})")

    if count_difference or duplicate_count or distant_count:
        logger.debug(
            "Native/contour mapping accepted within tolerance: "
            "count_difference={} duplicate_count={} distant_count={} budget={}",
            count_difference,
            duplicate_count,
            distant_count,
            mismatch_budget,
        )

    # Within the small mismatch budget, preserve the nearest component's
    # intensity. Dropping or down-weighting these rare candidates changes the
    # percentile filter, while the full contour path produces the same values
    # for candidates that survive it. Larger mismatches still reject the whole
    # native result above.

    areas = np.asarray([
        cv2.contourArea(contour) + 0.5 * len(contour)
        for contour in contours
    ])
    eccentricities = np.sqrt(
        np.asarray([
            1 - (ellipse[1][0] / ellipse[1][1])**2
            for ellipse in ellipses
        ]))
    intensities = component_intensities[component_indices]
    return positions, areas, intensities, eccentricities


def _detect_star_points_contour(
    img_gray: NDArray,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    img_shape = img_gray.shape
    if img_gray.dtype != np.float64:
        if np.issubdtype(img_gray.dtype, np.integer):
            img_gray = img_gray.astype(np.float64) / np.iinfo(img_gray.dtype).max
        else:
            img_gray = img_gray.astype(np.float64)

    img_blr = cv2.GaussianBlur(img_gray, (gaussian_ksize, gaussian_ksize),
                               sigma)
    img_blr_mean = np.mean(img_blr)
    img_blr_range = np.max(img_blr) - np.min(img_blr)
    img_blr = (img_blr - img_blr_mean) / img_blr_range

    resize_factor = 1
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2

    logger.debug("Mask logical selection")
    if mask is None:
        mask = np.ones(img_shape, dtype=bool)
        logger.debug("Mask calculation Complete (auto mask disabled; using full image)")
    else:
        mask = mask > 0
        logger.debug("Mask calculation Complete (using external mask only)")
    mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
    logger.debug(f"mask rate: {mask_rate:.2f}")

    while True:
        img_rec = _wavelet_dec_rec(img_blr, resize_factor=resize_factor) * mask
        bw = ((img_rec > np.percentile(img_rec[mask], 99.5)) * mask).astype(
            np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours = _find_star_contours(bw)
        logger.debug(f"{len(contours)} star points detected")
        if len(contours) < min_star_points and resize_factor < 1:
            logger.debug(
                "Not enough points, resize factor is now increasing by 2")
            resize_factor *= 2
            continue
        else:
            break

    if len(contours) < min_star_points:
        logger.warning(
            f"Not enough points: expected {min_star_points}, got {len(contours)}"
        )
    logger.debug(f"final resize factor = {resize_factor:.3f}")

    if not contours:
        return _empty_detected_stars()

    candidates = _measure_contour_candidates(contours, img_rec, bw)
    detected = _filter_star_candidates(*candidates)
    logger.debug(f"Final star points = {len(detected.positions)}")
    return detected



def detect_star_points_median_detailed(
    img_gray: NDArray,
    mask=None,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    min_star_points: int = 400,
    min_area: float = 7.0,
    max_eccentricity: float = 0.85,
    enable_outer_rescue: bool = False,
) -> MedianDetectionDetails:
    """Run median detection and retain the intermediate candidate data.

    The detailed result is intended for diagnostics. Production callers should
    normally use :func:`detect_star_points_median` below.
    """
    image = np.asarray(img_gray)
    if image.ndim != 2:
        raise ValueError(
            "detect_star_points_median expects a 2D grayscale image")
    if median_ksize <= 0 or median_ksize % 2 == 0:
        raise ValueError("median_ksize must be a positive odd integer")
    if open_ksize < 0 or (open_ksize > 0 and open_ksize % 2 == 0):
        raise ValueError("open_ksize must be zero or a positive odd integer")
    if min_area < 0:
        raise ValueError("min_area must be non-negative")

    star_mask, response, threshold = _detect_starmask_by_threshold_details(
        image,
        ksize=median_ksize,
        med_algo="median",
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        mask=mask,
    )
    bw = star_mask * np.uint8(255)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) > 5]
    logger.debug("{} median-path star candidates detected", len(contours))

    if len(contours) < min_star_points:
        logger.warning(
            "Not enough median-path points: expected {}, got {}",
            min_star_points,
            len(contours),
        )
    if not contours:
        detected_stars = DetectedStars(
            positions=np.empty((0, 2), dtype=np.float64),
            volumes=np.empty((0, ), dtype=np.float64),
        )
        return MedianDetectionDetails(
            detected_stars=detected_stars,
            response=response,
            threshold=threshold,
            star_mask=star_mask,
            candidate_positions=np.empty((0, 2), dtype=np.float64),
            areas=np.empty((0, ), dtype=np.float64),
            intensities=np.empty((0, ), dtype=np.float64),
            eccentricities=np.empty((0, ), dtype=np.float64),
            strict_valid_stars=np.empty((0, ), dtype=bool),
            rescued_stars=np.empty((0, ), dtype=bool),
            valid_stars=np.empty((0, ), dtype=bool),
            area_percentile=float("nan"),
            intensity_percentile=float("nan"),
        )

    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    centroids = np.asarray([ellipse[0] for ellipse in ellipses],
                           dtype=np.float64)
    areas = np.asarray([
        cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours
    ],
                       dtype=np.float64)
    eccentricities = np.sqrt(
        np.asarray(
            [1 - (ellipse[1][0] / ellipse[1][1])**2 for ellipse in ellipses],
            dtype=np.float64))

    intensities = np.zeros(len(contours), dtype=np.float64)
    for index, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        response_roi = response[y:y + height, x:x + width]
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        local_contour = contour - np.array([[[x, y]]], dtype=contour.dtype)
        cv2.drawContours(contour_mask, [local_contour], -1, 255, -1)
        intensities[index] = cv2.mean(response_roi, contour_mask)[0]

    area_percentile = np.percentile(areas, STAR_FILTER_PERCENTILE)
    intensity_percentile = np.percentile(intensities, STAR_FILTER_PERCENTILE)
    passes_non_eccentricity_filters = ((areas >= min_area)
                                       & (areas > area_percentile)
                                       & (intensities > intensity_percentile))
    strict_valid_stars = (passes_non_eccentricity_filters
                          & np.isfinite(eccentricities)
                          & (eccentricities < max_eccentricity))
    rescued_stars = np.zeros(len(contours), dtype=bool)
    if enable_outer_rescue:
        rescued_stars = passes_non_eccentricity_filters & ~strict_valid_stars
    valid_stars = strict_valid_stars | rescued_stars

    positions = centroids[valid_stars]
    candidate_volumes = _candidate_volumes_with_rescue(
        areas,
        intensities,
        eccentricities,
        rescued_stars,
        max_eccentricity,
    )
    volumes = candidate_volumes[valid_stars]
    logger.debug(
        "Final median-path star points = {} strict={} rescued={}",
        len(positions),
        int(np.count_nonzero(strict_valid_stars)),
        int(np.count_nonzero(rescued_stars)),
    )
    detected_stars = DetectedStars(positions=positions,
                                   volumes=volumes,
                                   intensities=intensities[valid_stars])
    return MedianDetectionDetails(
        detected_stars=detected_stars,
        response=response,
        threshold=threshold,
        star_mask=star_mask,
        candidate_positions=centroids,
        areas=areas,
        intensities=intensities,
        eccentricities=eccentricities,
        strict_valid_stars=strict_valid_stars,
        rescued_stars=rescued_stars,
        valid_stars=valid_stars,
        area_percentile=float(area_percentile),
        intensity_percentile=float(intensity_percentile),
    )


def detect_star_points_median(
    img_gray: NDArray,
    mask=None,
    median_ksize: int = 13,
    threshold_ratio: float = 1.0,
    open_ksize: int = 3,
    min_star_points: int = 400,
    min_area: float = 7.0,
    max_eccentricity: float = 0.85,
    enable_outer_rescue: bool = False,
) -> DetectedStars:
    """Alternative Norma detector using local median-background subtraction."""
    details = detect_star_points_median_detailed(
        img_gray,
        mask=mask,
        median_ksize=median_ksize,
        threshold_ratio=threshold_ratio,
        open_ksize=open_ksize,
        min_star_points=min_star_points,
        min_area=min_area,
        max_eccentricity=max_eccentricity,
        enable_outer_rescue=enable_outer_rescue,
    )
    return details.detected_stars



def _detect_star_points_opencv(*args, **kwargs) -> DetectedStars:
    """Compatibility name for the production Python/OpenCV detector."""
    return _detect_star_points_contour(*args, **kwargs)


def _detect_star_points_native_hybrid(
    img_gray: NDArray | GraySource,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    """Run fused native pixel processing with exact OpenCV contour geometry.

    A :class:`GraySource` lets the CUDA backend convert the gray on the device;
    the native stage then reports a constant gray itself.
    """
    img_shape = img_gray.shape
    if not isinstance(img_gray, GraySource):
        img_gray = _normalize_gray(img_gray)
        if np.ptp(img_gray) == 0:
            return _empty_detected_stars()
    resize_factor = 1.0
    while max(img_shape) * resize_factor > resize_length:
        resize_factor /= 2.0

    while True:
        components = star_detect_fused_pixel_components(
            img_gray,
            mask,
            resize_factor,
            gaussian_ksize=gaussian_ksize,
            sigma=sigma,
        )
        if components is None:
            return _empty_detected_stars()
        component_positions, component_intensities, binary_mask = components
        candidates = _measure_native_hybrid_contour_candidates(
            component_positions, component_intensities, binary_mask)
        candidate_count = len(candidates[0])
        logger.debug(f"{candidate_count} native hybrid star candidates detected")
        if candidate_count < min_star_points and resize_factor < 1:
            logger.debug(
                "Not enough points, resize factor is now increasing by 2")
            resize_factor *= 2
            continue
        break

    if candidate_count < min_star_points:
        logger.warning(
            f"Not enough points: expected {min_star_points}, got {candidate_count}"
        )
    logger.debug(f"final resize factor = {resize_factor:.3f}")
    detected = _filter_star_candidates(*candidates)
    logger.debug(f"Final native hybrid star points = {len(detected.positions)}")
    return detected


def detect_star_points(
    img_gray: NDArray | GraySource,
    mask=None,
    resize_length=10000,
    gaussian_ksize: int = 9,
    sigma: float = 2,
    min_star_points: int = 400,
) -> DetectedStars:
    """Detect stars with native pixel work plus exact host contour geometry."""
    kwargs = {
        "mask": mask,
        "resize_length": resize_length,
        "gaussian_ksize": gaussian_ksize,
        "sigma": sigma,
        "min_star_points": min_star_points,
    }
    try:
        return _detect_star_points_native_hybrid(img_gray, **kwargs)
    except CustomOpUnavailableError as exc:
        logger.debug(
            "Native star detector unavailable; using contour fallback: {}", exc)
    except _NativeHybridGeometryMismatch as exc:
        logger.warning(
            "Native star detector geometry guard failed; using contour fallback: {}",
            exc,
        )
    except StarDetectCapacityError as exc:
        logger.warning(
            "Native star detector capacity guard reached; using contour fallback: {}",
            exc,
        )
    except CustomOpResourceExhaustedError as exc:
        logger.warning(
            "Native star detector resources exhausted; using contour fallback: {}",
            exc,
        )
    if isinstance(img_gray, GraySource):
        img_gray = img_gray.host_gray()
    return _detect_star_points_contour(img_gray, **kwargs)
