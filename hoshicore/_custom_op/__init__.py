"""Public entrypoint for optional custom-op APIs."""

import os
import sys

# Python 3.8+ on Windows no longer searches the directory containing a .pyd for
# its DLL dependencies. Add this package's directory so that MinGW runtime DLLs
# copied here by build_ops.py (libgomp-1.dll, libgcc_s_seh-1.dll, etc.) are found.
if sys.platform == "win32":
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))

from hoshicore._custom_op.api import (
    build_info,
    calibration_divide,
    calibration_subtract,
    camera_model_remap,
    custom_ops_available,
    equalize_noise_correct,
    asterism_anchor_votes,
    asterism_mutual_nearest,
    asterism_tokens,
    extract_point_features,
    fgp_accumulate,
    fgp_masked_mean_merge,
    get_backend_preference,
    huber_weighted_chunk,
    huber_weighted_accumulate,
    max_combine,
    matching_cosine_bidirectional_nearest,
    median_filter_2d,
    median_reduce_chunk,
    median_star_mask,
    noise_equalization_params,
    noise_fill_local_mean,
    sigma_clip_fused_chunk,
    sigma_clip_fused_masked_merge,
    sigma_clip_fused_merge,
    sigma_clip_iterative_chunk,
    set_backend_preference,
    star_detect_fused_pixel_components,
    star_mask_dog,
    star_shrink_detect_mask,
    star_shrink_dog_process,
    star_shrink_process,
    threshold_max_merge,
    wavelet_dec_rec,
)

__all__ = [
    "build_info",
    "calibration_divide",
    "calibration_subtract",
    "camera_model_remap",
    "custom_ops_available",
    "equalize_noise_correct",
    "asterism_anchor_votes",
    "asterism_mutual_nearest",
    "asterism_tokens",
    "extract_point_features",
    "fgp_accumulate",
    "fgp_masked_mean_merge",
    "get_backend_preference",
    "huber_weighted_chunk",
    "huber_weighted_accumulate",
    "max_combine",
    "matching_cosine_bidirectional_nearest",
    "median_filter_2d",
    "median_reduce_chunk",
    "median_star_mask",
    "noise_equalization_params",
    "noise_fill_local_mean",
    "sigma_clip_fused_chunk",
    "sigma_clip_fused_masked_merge",
    "sigma_clip_fused_merge",
    "sigma_clip_iterative_chunk",
    "set_backend_preference",
    "star_detect_fused_pixel_components",
    "star_mask_dog",
    "star_shrink_detect_mask",
    "star_shrink_dog_process",
    "star_shrink_process",
    "threshold_max_merge",
    "wavelet_dec_rec",
]
