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
