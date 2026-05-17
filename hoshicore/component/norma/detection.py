"""Star point detection from images."""
import dataclasses
from math import log

import cv2
import numpy as np
import pywt
from loguru import logger
from numpy.typing import NDArray


@dataclasses.dataclass
class DetectedStars:
    positions: NDArray[np.float64]
    volumes: NDArray[np.float64]


def _wavelet_dec_rec(img_blr, resize_factor=0.25):
    img_shape = img_blr.shape
    level = int(6 - log(1 / resize_factor, 2))

    img_blr_resize = cv2.resize(img_blr, None, fx=resize_factor,
                                fy=resize_factor)
    coeffs = pywt.wavedec2(img_blr_resize, "db8", level=level)
    coeffs[0].fill(0)
    coeffs[-1][0].fill(0)
    coeffs[-1][1].fill(0)
    coeffs[-1][2].fill(0)

    img_rec_resize = pywt.waverec2(coeffs, "db8")
    img_rec = cv2.resize(img_rec_resize, (img_shape[1], img_shape[0]))
    return img_rec

def _bandpass_dog(img_blr: np.ndarray, resize_factor: float = 0.25) -> np.ndarray:
    h, w = img_blr.shape
    small = cv2.resize(img_blr, None, fx=resize_factor, fy=resize_factor,
                       interpolation=cv2.INTER_AREA)
    diag_len = (h**2+w**2)**(1/2)
    # 假设原图最大星点 20px in 7000px
    fine_width = diag_len * 0.0001
    coarse_width = diag_len * 0.001
    # 粗尺度去背景，细尺度去噪，差值保留星点
    coarse = cv2.GaussianBlur(small, (0, 0), sigmaX=coarse_width)
    fine   = cv2.GaussianBlur(small, (0, 0), sigmaX=fine_width)
    dog = coarse - fine
    return cv2.resize(dog, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_star_points(
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
    tmp_mask = cv2.resize(img_gray, None, fx=resize_factor, fy=resize_factor)
    tmp_mask_10percent = np.percentile(tmp_mask, 10)
    tmp_mask = (tmp_mask < min(tmp_mask_10percent, 0.15)).astype(
        np.uint8) * 255
    
    dilate_size = int(max(img_shape) * 0.003 * resize_factor)
    tmp_mask = 255 - cv2.dilate(
        tmp_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (dilate_size, dilate_size)))
    tmp_mask = cv2.resize(tmp_mask, (img_shape[1], img_shape[0]))
    if mask is None:
        mask = tmp_mask > 127
    else:
        mask = np.logical_and(tmp_mask > 127, mask > 0)
    logger.debug("Mask calculation Complete")
    mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
    logger.debug(f"mask rate: {mask_rate:.2f}")
    if mask_rate < 50:
        mask = np.ones(tmp_mask.shape, dtype="bool")

    while True:
        img_rec = _wavelet_dec_rec(img_blr, resize_factor=resize_factor) * mask
        bw = ((img_rec > np.percentile(img_rec[mask], 99.5)) * mask).astype(
            np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if len(contour) > 5]
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

    elps = [cv2.fitEllipse(contour) for contour in contours]
    centroids = np.array([e[0] for e in elps])
    areas = np.array([
        cv2.contourArea(contour) + 0.5 * len(contour) for contour in contours
    ])
    eccentricities = np.sqrt(
        np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))

    mask_img = np.zeros(bw.shape, np.uint8)
    intensities = np.zeros(areas.shape)
    for i in range(len(contours)):
        cv2.drawContours(mask_img, contours, i, 255, -1)
        rect = cv2.boundingRect(contours[i])
        val = cv2.mean(
            img_rec[rect[1]:rect[1] + rect[3] + 1,
                    rect[0]:rect[0] + rect[2] + 1],
            mask_img[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
        mask_img[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
        intensities[i] = val[0]

    valid_stars = np.logical_and(areas > 20, eccentricities < .8)
    valid_stars = np.logical_and(
        np.logical_and(valid_stars, areas > np.percentile(areas, 20)),
        intensities > np.percentile(intensities, 20)
    )

    star_pts = centroids[valid_stars]
    areas = areas[valid_stars]
    intensities = intensities[valid_stars]
    logger.debug(f"Final star points = {len(star_pts)}")

    return DetectedStars(positions=star_pts, volumes=areas * intensities)
