"""CPU detection grayscale conversion, retaining OpenCV's color arithmetic."""

from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op._dispatch import apply_compiled_threads
from hoshicore._custom_op._dispatch import fallback_preference
from hoshicore._custom_op._dispatch import load_compiled_module as _load_compiled_module_result
from hoshicore._custom_op.backend_registry import native_backend_available


def detection_gray_f64_numpy(image: np.ndarray) -> NDArray[np.float64]:
    """Original Norma BGR-to-gray conversion and intensity normalization."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)
    if np.issubdtype(image.dtype, np.integer):
        gray /= np.iinfo(image.dtype).max
    else:
        maximum = gray.max()
        if maximum > 1.0:
            gray /= maximum
    return gray


def detection_gray_u16_numpy(image: np.ndarray) -> NDArray[np.uint16] | None:
    """Median gray without a float64 intermediate; None requests the original path."""
    if image.dtype != np.uint16:
        return None
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if not (gray.min() >= 0.0 and gray.max() <= 65535.0):
        return None
    return np.rint(gray, out=gray).astype(np.uint16)


def _supported_image(image: np.ndarray) -> bool:
    return (image.dtype in (np.uint8, np.uint16) and image.size > 0
            and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4))))


def _compiled_cast(image: np.ndarray) -> Callable[..., np.ndarray | None]:
    module, _ = _load_compiled_module_result()
    if module is None or not hasattr(module, "detection_gray_cast_cpu"):
        raise RuntimeError("compiled custom op backend is unavailable")
    apply_compiled_threads("detection_gray", image)
    return module.detection_gray_cast_cpu


def detection_gray_f64_compiled(image: np.ndarray) -> NDArray[np.float64]:
    if not _supported_image(image):
        return detection_gray_f64_numpy(image)
    cast = _compiled_cast(image)
    gray = np.ascontiguousarray(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(cast(gray, "prepare"), cv2.COLOR_BGR2GRAY)
    return cast(gray, "normalize", float(np.iinfo(image.dtype).max))


def detection_gray_u16_compiled(image: np.ndarray) -> NDArray[np.uint16] | None:
    if image.dtype != np.uint16 or image.ndim == 2 or not _supported_image(image):
        return detection_gray_u16_numpy(image)
    cast = _compiled_cast(image)
    gray = cv2.cvtColor(cast(np.ascontiguousarray(image), "prepare"), cv2.COLOR_BGR2GRAY)
    return cast(gray, "quantize")


def _native_available() -> bool:
    return native_backend_available(
        "detection_gray", fallback_preference(), load_module=_load_compiled_module_result)[0]


def detection_gray_f64(image: np.ndarray) -> NDArray[np.float64]:
    """Normalized float64 gray with the current OpenCV build's exact color conversion."""
    if _supported_image(image) and _native_available():
        return detection_gray_f64_compiled(image)
    return detection_gray_f64_numpy(image)


def detection_gray_u16(image: np.ndarray) -> NDArray[np.uint16] | None:
    """Uint16 median gray; unsupported dtypes/ranges return None for the caller's fallback."""
    if (image.dtype == np.uint16 and image.ndim == 3
            and _supported_image(image) and _native_available()):
        return detection_gray_u16_compiled(image)
    return detection_gray_u16_numpy(image)
