import unittest


from hoshicore._custom_op import _dispatch
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore._custom_op.ops.fgp as fgp_ops
import hoshicore._custom_op.ops.filter as filter_ops
import hoshicore._custom_op.ops.max as max_ops
import hoshicore._custom_op.ops.median as median_ops
import hoshicore._custom_op.ops.noise as noise_ops
import hoshicore._custom_op.ops.star_shrink as star_shrink_ops
import hoshicore._custom_op.ops.sigma_clip as sigma_clip_chunk_ops


class CustomOpsTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        _dispatch.compiled_build_info.cache_clear()
        _dispatch.load_metal_module.cache_clear()
        filter_ops._load_compiled_module_result.cache_clear()
        filter_ops._select_median_filter_backend.cache_clear()
        fgp_ops._load_compiled_module_result.cache_clear()
        fgp_ops._select_fgp_backend.cache_clear()
        fgp_ops._select_huber_backend.cache_clear()
        max_ops._load_compiled_module_result.cache_clear()
        max_ops._select_max_backend.cache_clear()
        max_ops._select_threshold_max_backend.cache_clear()
        noise_ops._load_compiled_module_result.cache_clear()
        noise_ops._select_equalize_noise_backend.cache_clear()
        noise_ops._select_fill_local_mean_backend.cache_clear()
        noise_ops._select_equalization_params_backend.cache_clear()
        star_shrink_ops._load_compiled_module_result.cache_clear()
        star_shrink_ops._load_metal_module_result.cache_clear()
        star_shrink_ops._select_star_mask_dog_backend.cache_clear()
        star_shrink_ops._select_star_shrink_detect_mask_backend.cache_clear()
        star_shrink_ops._select_star_shrink_dog_process_backend.cache_clear()
        star_shrink_ops._select_star_shrink_process_backend.cache_clear()
        median_ops._load_compiled_module_result.cache_clear()
        sigma_clip_chunk_ops._load_compiled_module_result.cache_clear()
        alignment_ops._load_compiled_module_result.cache_clear()
