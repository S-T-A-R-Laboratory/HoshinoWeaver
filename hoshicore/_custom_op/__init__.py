"""Public entrypoint for optional custom-op APIs."""

from hoshicore._custom_op.api import (
    build_info,
    camera_model_remap,
    custom_ops_available,
    equalize_noise_correct,
    fgp_add,
    fgp_accumulate,
    fgp_masked_mean_merge,
    huber_weighted_accumulate,
    max_combine,
    median_reduce_chunk,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
    threshold_max_merge,
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
    "median_reduce_chunk",
    "sigma_clip_fused_masked_merge",
    "sigma_clip_fused_merge",
    "threshold_max_merge",
]
