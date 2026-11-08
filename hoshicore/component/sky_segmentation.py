"""SkySeg v2 global pyramid and capped ROI refinement, independent of ORT.

Adapted from the model's standalone deployment predictor. Numeric input ranges,
channel order and optional linear-input stretch are explicit; no color management.
"""
from dataclasses import dataclass
import json
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from .onnx_backend import InferenceBackend
from .model_input import image_to_rgb8
from .model_metadata import validate_model_metadata
from .sky_roi import refine_rois


@dataclass(frozen=True)
class SkyPrediction:
    """Native-size sky probability and uint8 labels (1=sky, 0=ground)."""

    probability: np.ndarray
    mask: np.ndarray
    info: dict
    raw_probability: np.ndarray | None = None


@dataclass(frozen=True)
class SkyProgress:
    """Worker event; counts are phase-local global scales or selected ROI windows."""

    stage: str
    completed: int
    total: int
    scale: int | None = None
    seconds: float | None = None
    strategy: str = 'roi'


def report_progress(callback, event):
    if callback is not None:
        try:
            callback(event)
        except Exception as error:
            logger.warning('SkySeg progress callback failed: {}', error)


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


def sky_probability(logits, original_size, sky_channel=1):
    values = np.asarray(logits, dtype=np.float32)
    if (values.ndim != 4 or values.shape[:2] != (1, 2)
            or not min(values.shape[2:]) or not np.isfinite(values).all()):
        raise RuntimeError('Expected finite (1,2,H,W) sky logits')
    height, width = original_size
    aligned = cv2.resize(values[0].transpose(1, 2, 0), (width, height),
                         interpolation=cv2.INTER_LINEAR)
    difference = aligned[:, :, sky_channel] - aligned[:, :, 1 - sky_channel]
    return np.exp(-np.logaddexp(np.float32(0), -difference)).astype(np.float32)


class SkySegPredictor:
    """Model adapter using a synchronous backend owned by its caller."""

    def __init__(self, backend: InferenceBackend, metadata=None):
        description = backend.describe()
        inputs, outputs = description.inputs, description.outputs
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                'SkySeg requires one image input and one logits output')
        types = {'tensor(float16)': np.float16, 'tensor(float)': np.float32}
        if inputs[0].type not in types:
            raise ValueError(f'Unexpected SkySeg input type: {inputs[0].type}')
        self.sky_channel = 1
        if metadata is not None:
            validate_model_metadata(metadata)
            if metadata['task'] != 'segmentation' or metadata['adapter'] != 'sky_logits':
                raise ValueError('Sky service requires task=segmentation and adapter=sky_logits')
            if len(metadata['inputs']) != 1 or len(metadata['outputs']) != 1:
                raise ValueError('sky_logits requires one input and one output')
            image, output = metadata['inputs'][0], metadata['outputs'][0]
            for declared, actual in ((image, inputs[0]), (output, outputs[0])):
                if (declared['name'] != actual.name or declared['layout'] != 'NCHW'
                        or {'float16': 'tensor(float16)', 'float32': 'tensor(float)'}.get(declared['dtype']) != actual.type):
                    raise ValueError('Model tensor name, dtype or layout does not match sky_logits contract')
            expected_spatial = dict(size_divisor=image.get('spatial', {}).get('size_divisor'),
                                    global_resize='keep_ratio_then_ceil_resize', local_resize='none',
                                    padding='normalized_zero_right_bottom')
            if (image.get('channels') != 'RGB' or image.get('signal') != 'nonlinear'
                    or image.get('value_range') != [0, 255] or image.get('spatial') != expected_spatial
                    or output.get('kind') != 'logits' or output.get('spatial') != 'input'
                    or not isinstance(output.get('classes'), list) or len(output['classes']) != 2
                    or type(output.get('sky_channel')) is not int or output['sky_channel'] not in (0, 1)):
                raise ValueError('Unsupported sky_logits preprocessing or output semantics')
            self.sky_channel = output['sky_channel']
            normalization = image.get('normalization', {})
            self.mean = np.asarray(normalization.get('mean'), dtype=np.float64)
            self.std = np.asarray(normalization.get('std'), dtype=np.float64)
            self.divisor = expected_spatial['size_divisor']
            self.model_version = metadata['model'].get('version', 'unspecified')
        else:
            # A self-described model may supply the same contract in ONNX metadata.
            # Producer format_version/model_version are identifiers, not compatibility gates.
            raw = description.metadata.get('skyseg')
            if raw is None:
                raise ValueError('Missing model semantics; provide adjacent metadata.json')
            embedded = json.loads(raw)
            if embedded.get('classes') != ['non_sky', 'sky'] or embedded.get('color_order') != 'RGB':
                raise ValueError('Unexpected SkySeg class order or channel order')
            self.mean = np.asarray(embedded.get('mean'), dtype=np.float64)
            self.std = np.asarray(embedded.get('std'), dtype=np.float64)
            self.divisor = embedded.get('size_divisor')
            self.model_version = embedded.get('model_version', 'unspecified')
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
        if outputs[0].type not in types:
            raise ValueError('sky_logits requires floating-point output')
        for tensor, channels in ((inputs[0], 3), (outputs[0], 2)):
            shape = tensor.shape
            if shape is not None and (len(shape) != 4
                    or any(isinstance(size, int) and size != expected
                           for size, expected in zip(shape[:2], (1, channels)))
                    or any(isinstance(size, int) for size in shape[2:])):
                raise ValueError('sky_logits requires dynamic-spatial NCHW input/output with batch 1')

    def probability_native(self, rgb):
        """Normalize local RGB without resizing; pad, infer, crop, then softmax."""
        height, width = rgb.shape[:2]
        values = np.asarray(rgb, dtype=np.float32).copy()
        cv2.subtract(values, self.mean.reshape(1, -1), values)
        cv2.multiply(values, 1 / self.std.reshape(1, -1), values)
        ph = (height + self.divisor - 1) // self.divisor * self.divisor
        pw = (width + self.divisor - 1) // self.divisor * self.divisor
        values = np.pad(values, ((0, ph - height), (0, pw - width), (0, 0)), constant_values=0)
        tensor = np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=self.dtype)
        logits, = self.backend.infer({self.input_name: tensor}, [self.output_name])
        logits = np.asarray(logits)
        if logits.shape != (1, 2, ph, pw) or not np.isfinite(logits).all():
            raise RuntimeError('Expected finite native-size padded SkySeg v2 logits')
        return sky_probability(logits[:, :, :height, :width], (height, width), self.sky_channel)

    def predict(self,
                image,
                scales=(512, 768, 1024),
                threshold=.5,
                color_order='BGR',
                progress: Callable[[SkyProgress], None] | None = None,
                *, strategy='roi', working_long_edge=1536,
                value_range=None, linear=False, stretch_gamma=2.2):
        scales = validate_scales(scales)
        if strategy not in ('roi', 'pyramid'):
            raise ValueError('strategy must be roi or pyramid')
        if type(working_long_edge) is not int or working_long_edge < 32:
            raise ValueError('working_long_edge must be an integer >=32')
        if isinstance(
                threshold,
                bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite in [0,1]')
        rgb = image_to_rgb8(image, color_order=color_order, value_range=value_range,
                            linear=linear, stretch_gamma=stretch_gamma)
        probability = None
        timings = {}
        for index, scale in enumerate(scales):
            logger.info('SkySeg global scale started: {}/{} long_edge={}', index + 1, len(scales), scale)
            report_progress(progress, SkyProgress('scale_started', index, len(scales), scale, strategy=strategy))
            started = perf_counter()
            tensor = prepare_image(rgb, scale, self.mean, self.std,
                                   self.divisor, self.dtype)
            logits, = self.backend.infer({self.input_name: tensor}, [self.output_name])
            if np.shape(logits) != (1, 2, *tensor.shape[2:]):
                raise RuntimeError('sky_logits output must match padded input spatial size')
            current = sky_probability(logits, rgb.shape[:2], self.sky_channel)
            if probability is None:
                probability = current
            else:
                probability += current
            timings[str(scale)] = perf_counter() - started
            logger.info('SkySeg global scale completed: {}/{} long_edge={} seconds={:.3f}',
                        index + 1, len(scales), scale, timings[str(scale)])
            report_progress(progress, SkyProgress('scale_completed', index + 1, len(scales),
                                                   scale, timings[str(scale)], strategy))
        probability /= np.float32(len(scales))
        raw_probability, roi_info = None, None
        if strategy == 'roi':
            raw_probability = probability
            started = perf_counter()
            probability, roi_info = refine_rois(
                rgb, probability, self.probability_native, long_edge=working_long_edge,
                report=lambda stage, completed, total, seconds: report_progress(
                    progress, SkyProgress(stage, completed, total, seconds=seconds, strategy=strategy)))
            timings['roi'] = perf_counter() - started
        mask = (probability >= threshold).astype(np.uint8)
        report_progress(progress, SkyProgress('complete', 1, 1, strategy=strategy))
        return SkyPrediction(
            probability, mask, {
                'scales': list(scales),
                'threshold': float(threshold),
                'providers': list(self.providers),
                'input_dtype': np.dtype(self.dtype).name,
                'original_size': list(rgb.shape[:2]),
                'seconds': timings,
                'model_version': self.model_version,
                'strategy': strategy,
                'roi': roi_info,
            }, raw_probability)
