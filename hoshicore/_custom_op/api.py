"""Public facade for custom-op APIs."""

from hoshicore._custom_op.ops.fgp import (
    fgp_add,
    fgp_accumulate,
    fgp_masked_mean_merge,
    huber_weighted_accumulate,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
)
from hoshicore._custom_op.ops.filter import median_filter_2d
from hoshicore._custom_op.ops.max import (
    build_info,
    custom_ops_available,
    max_combine,
    threshold_max_merge,
)
from hoshicore._custom_op.ops.median import median_reduce_chunk
from hoshicore._custom_op.ops.noise import equalize_noise_correct
from hoshicore._custom_op.ops.remap import camera_model_remap
from hoshicore._custom_op.ops.sigma_clip import (
    sigma_clip_iterative_chunk,
    sigma_clip_fused_chunk,
)

__all__ = [
    "build_info",
    "camera_model_remap",
    "custom_ops_available",
    "equalize_noise_correct",
    "fgp_add",
    "fgp_accumulate",
    "fgp_masked_mean_merge",
    "huber_weighted_accumulate",
    "max_combine",
    "median_filter_2d",
    "median_reduce_chunk",
    "sigma_clip_fused_chunk",
    "sigma_clip_fused_masked_merge",
    "sigma_clip_fused_merge",
    "sigma_clip_iterative_chunk",
    "threshold_max_merge",
]
