"""SkySeg v2 capped-resolution ROI fusion, adapted from upstream roi.py.

Only selected windows execute, one at a time. Local inputs are unscaled RGB;
the model adapter supplies normalization, padding and logits cropping.
"""
from time import perf_counter

import cv2
import numpy as np
from loguru import logger


def plan_tiles(shape, selection, core=384, halo=64, stride=256):
    height, width = shape

    def starts(length):
        if length <= core:
            return [0]
        return sorted(set(range(0, length - core + 1, stride)) | {length - core})

    return [dict(
        core=[x, y, min(width, x + core), min(height, y + core)],
        patch=[max(0, x - halo), max(0, y - halo), min(width, x + core + halo),
               min(height, y + core + halo)],
        selected=bool(selection[y:min(height, y + core), x:min(width, x + core)].any()))
        for y in starts(height) for x in starts(width)]


def refine_rois(rgb, parent, probability_fn, *, long_edge=1536, report=None):
    """Return native-size fused probability and the upstream-compatible ROI report.

    report(stage, completed, total, seconds) is optional; its caller owns error
    isolation and thread forwarding. Counts refer only to selected windows.
    """
    if parent.shape != rgb.shape[:2] or not np.isfinite(parent).all() or np.any((parent < 0) | (parent > 1)):
        raise ValueError('Expected aligned finite probabilities in [0,1]')
    begin = perf_counter()
    h0, w0 = parent.shape
    ratio = min(1., long_edge / max(h0, w0))
    height, width = max(1, round(h0 * ratio)), max(1, round(w0 * ratio))
    work = rgb if (height, width) == (h0, w0) else cv2.resize(
        rgb, (width, height), interpolation=cv2.INTER_AREA)
    prior = cv2.resize(parent, (width, height), interpolation=cv2.INTER_LINEAR)
    boundary = cv2.morphologyEx((prior >= .5).astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
    band = cv2.dilate(boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))) > 0
    selected = band | ((prior > .2) & (prior < .8))
    gate = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), 8)
    boxes = plan_tiles((height, width), selected)
    total = sum(box['selected'] for box in boxes)
    logger.info('SkySeg ROI planned: working_shape={} selected_windows={}/{} selection_fraction={:.4f}',
                (height, width), total, len(boxes), float(selected.mean()))
    if report:
        report('roi_planned', 0, total, None)
    acc, weights = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    calls = 0
    for box in boxes:
        if not box['selected']:
            continue
        x, y, right, bottom = box['core']
        px, py, pr, pb = box['patch']
        if report:
            report('tile_started', calls, total, None)
        started = perf_counter()
        probability = np.asarray(probability_fn(work[py:pb, px:pr]), dtype=np.float32)
        if (probability.shape != (pb - py, pr - px) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise ValueError('Invalid local probability')
        local = probability[y - py:bottom - py, x - px:right - px]
        weight = np.maximum(np.hanning(bottom - y)[:, None] * np.hanning(right - x)[None, :], .05).astype(np.float32)
        acc[y:bottom, x:right] += local * weight
        weights[y:bottom, x:right] += weight
        calls += 1
        elapsed = perf_counter() - started
        logger.info('SkySeg ROI window completed: {}/{} core={} patch={} seconds={:.3f}',
                    calls, total, box['core'], box['patch'], elapsed)
        if report:
            report('tile_completed', calls, total, elapsed)
    covered = weights > 0
    if not np.all(covered[selected]):
        raise RuntimeError('ROI windows did not cover selected pixels')
    if report:
        report('fusing', calls, total, None)
    local = np.divide(acc, weights, out=prior.copy(), where=covered)
    detail = cv2.resize(local, (w0, h0), interpolation=cv2.INTER_LINEAR)
    alpha = .7 * cv2.resize(gate * covered, (w0, h0), interpolation=cv2.INTER_LINEAR)
    conflict = ((parent >= .9) | (parent <= .1)) & ((parent >= .5) != (detail >= .5))
    alpha[conflict] = np.minimum(alpha[conflict], .2)
    result = np.clip(parent + alpha * (detail - parent), 0, 1).astype(np.float32)
    info = dict(original_shape=[h0, w0], working_shape=[height, width], long_edge_cap=long_edge,
                core=384, halo=64, stride=256, boundary_radius_work_pixels=32,
                uncertainty=[.2, .8], local_weight=.7, confident_conflict_max_alpha=.2,
                total_tiles=len(boxes), scanned_tiles=calls, selection_fraction=float(selected.mean()),
                covered_fraction=float(covered.mean()), native_conflict_fraction=float(conflict.mean()),
                elapsed_seconds=perf_counter() - begin, context_calls=0, audit_calls=0,
                recursion=False, boxes=boxes)
    return result, info
