"""SkySeg v1 preprocessing and probability pyramid, independent of ORT.

Adapted from the model's standalone deployment predictor. Images are uint8
RGB/BGR in sRGB encoding; callers own orientation and color conversion.
"""
from dataclasses import dataclass
import json
from time import perf_counter

import cv2
import numpy as np

from .onnx_backend import InferenceBackend


@dataclass(frozen=True)
class SkyPrediction:
    """Native-size sky probability and uint8 labels (1=sky, 0=ground)."""

    probability: np.ndarray
    mask: np.ndarray
    info: dict


def validate_scales(scales):
    result = tuple(scales)
    if (not result or any(
            isinstance(x, (bool, np.bool_))
            or not isinstance(x, (int, np.integer)) or x < 32
            for x in result) or len(set(result)) != len(result)):
        raise ValueError('scales must be distinct integers >=32')
    return tuple(int(x) for x in result)


def prepare_image(rgb: np.ndarray, long_edge, mean, std, divisor, dtype):
    """Keep ratio, then resize dimensions up to divisor; do not letterbox."""
    height, width = rgb.shape[:2]
    ratio = long_edge / max(height, width)
    width, height = max(1,
                        int(width * ratio + .5)), max(1,
                                                      int(height * ratio + .5))
    image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    width = ((width + divisor - 1) // divisor) * divisor
    height = ((height + divisor - 1) // divisor) * divisor
    image = cv2.resize(image, (width, height),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # Match training's float32 destination and float64 normalization scalars.
    cv2.subtract(image,
                 np.asarray(mean, dtype=np.float64).reshape(1, -1), image)
    cv2.multiply(image, 1 / np.asarray(std, dtype=np.float64).reshape(1, -1),
                 image)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=dtype)


def sky_probability(logits, original_size):
    values = np.asarray(logits, dtype=np.float32)
    if (values.ndim != 4 or values.shape[:2] != (1, 2)
            or not min(values.shape[2:]) or not np.isfinite(values).all()):
        raise RuntimeError('Expected finite (1,2,H,W) sky logits')
    height, width = original_size
    aligned = cv2.resize(values[0].transpose(1, 2, 0), (width, height),
                         interpolation=cv2.INTER_LINEAR)
    difference = aligned[:, :, 1] - aligned[:, :, 0]
    return np.exp(-np.logaddexp(np.float32(0), -difference)).astype(np.float32)


class SkySegPredictor:
    """Model adapter using a synchronous backend owned by its caller."""

    def __init__(self, backend: InferenceBackend):
        description = backend.describe()
        inputs, outputs = description.inputs, description.outputs
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                'SkySeg requires one image input and one logits output')
        types = {'tensor(float16)': np.float16, 'tensor(float)': np.float32}
        if inputs[0].type not in types:
            raise ValueError(f'Unexpected SkySeg input type: {inputs[0].type}')
        raw = description.metadata.get('skyseg')
        if raw is None:
            raise ValueError(
                'Missing skyseg metadata; select the SkySeg v1 model')
        metadata = json.loads(raw)
        if metadata.get('classes') != [
                'non_sky', 'sky'
        ] or metadata.get('color_order') != 'RGB':
            raise ValueError('Unexpected SkySeg class order or color encoding')
        self.mean = np.asarray(metadata['mean'], dtype=np.float64)
        self.std = np.asarray(metadata['std'], dtype=np.float64)
        self.divisor = metadata['size_divisor']
        if (self.mean.shape != (3, ) or self.std.shape != (3, )
                or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std <= 0)
                or isinstance(self.divisor, bool)
                or not isinstance(self.divisor, int) or self.divisor < 1):
            raise ValueError('Invalid SkySeg normalization metadata')
        self.backend = backend
        self.providers = description.providers
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.dtype = types[inputs[0].type]

    def predict(self,
                image,
                scales=(512, 768, 1024),
                threshold=.5,
                color_order='BGR'):
        scales = validate_scales(scales)
        if isinstance(
                threshold,
                bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite in [0,1]')
        image = np.asarray(image)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or not min(image.shape[:2])):
            raise ValueError(
                'SkySeg requires a nonempty HxWx3 uint8 sRGB image')
        if color_order.upper() not in ('RGB', 'BGR'):
            raise ValueError('color_order must be RGB or BGR')
        rgb = image if color_order.upper() == 'RGB' else image[:, :, ::-1]
        probability = None
        timings = {}
        for scale in scales:
            started = perf_counter()
            tensor = prepare_image(rgb, scale, self.mean, self.std,
                                   self.divisor, self.dtype)
            logits, = self.backend.infer({self.input_name: tensor}, [self.output_name])
            current = sky_probability(logits, image.shape[:2])
            if probability is None:
                probability = current
            else:
                probability += current
            timings[str(scale)] = perf_counter() - started
        probability /= np.float32(len(scales))
        return SkyPrediction(
            probability, (probability >= threshold).astype(np.uint8), {
                'scales': list(scales),
                'threshold': float(threshold),
                'providers': list(self.providers),
                'input_dtype': np.dtype(self.dtype).name,
                'original_size': list(image.shape[:2]),
                'seconds': timings,
            })
