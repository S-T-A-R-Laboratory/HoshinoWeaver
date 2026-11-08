"""Small image-input bridge until workflow-wide pixel semantics are available."""
from numbers import Real

import numpy as np

from .data_container import FloatImage


def image_to_rgb8(image, *, color_order='BGR', value_range=None,
                  linear=False, stretch_gamma=2.2):
    """Map HWC pixels to RGB8 without modifying input or guessing its range.

    Unsigned integers default to their full range; floats default to [0,1].
    Explicit ranges are black/white levels: values outside them are clipped.
    Linear input receives x**(1/stretch_gamma) after range normalization.
    No color management, histogram stretch or per-image min/max is applied.
    """
    if isinstance(image, FloatImage):
        if value_range is None:
            source_dtype = np.dtype(image.dtype)
            if source_dtype.kind == 'u':
                value_range = (0, np.iinfo(source_dtype).max)
            elif source_dtype.kind != 'f':
                raise ValueError('FloatImage requires unsigned or floating-point range dtype')
        image = image.data
    pixels = np.asarray(image)
    if (pixels.ndim != 3 or pixels.shape[2] != 3 or not min(pixels.shape[:2])
            or pixels.dtype.kind not in ('u', 'f')):
        raise ValueError('Expected nonempty HxWx3 unsigned-integer or floating-point image')
    if not isinstance(color_order, str) or color_order.upper() not in ('RGB', 'BGR'):
        raise ValueError('color_order must be RGB or BGR')
    if not isinstance(linear, (bool, np.bool_)):
        raise ValueError('linear must be boolean')
    if (isinstance(stretch_gamma, (bool, np.bool_)) or not isinstance(stretch_gamma, Real)
            or not np.isfinite(stretch_gamma) or stretch_gamma <= 0):
        raise ValueError('stretch_gamma must be finite and positive')
    if not np.isfinite(pixels).all():
        raise ValueError('Image pixels must be finite')
    explicit_range = value_range is not None
    if value_range is None:
        low, high = (0., 1.) if pixels.dtype.kind == 'f' else (0., float(np.iinfo(pixels.dtype).max))
    else:
        levels = np.asarray(value_range, dtype=np.float64)
        if levels.shape != (2,) or not np.isfinite(levels).all() or levels[1] <= levels[0]:
            raise ValueError('value_range must contain finite increasing black/white levels')
        low, high = map(float, levels)
    span = high - low
    if not np.isfinite(span):
        raise ValueError('value_range span must be finite')
    if not explicit_range and pixels.dtype.kind == 'f' and (np.any(pixels < low) or np.any(pixels > high)):
        raise ValueError('Floating-point input defaults to [0,1]; supply value_range for other ranges')
    if pixels.dtype == np.uint8 and not explicit_range and not linear:
        result = pixels
    else:
        work_dtype = np.float64 if pixels.dtype.itemsize > 4 or max(abs(low), abs(high), span) > np.finfo(np.float32).max else np.float32
        normalized = pixels.astype(work_dtype)
        normalized -= float(low)
        normalized /= span
        np.clip(normalized, 0, 1, out=normalized)
        if linear:
            np.power(normalized, 1 / stretch_gamma, out=normalized)
        result = np.rint(normalized * 255).astype(np.uint8)
    return result if color_order.upper() == 'RGB' else result[:, :, ::-1]
