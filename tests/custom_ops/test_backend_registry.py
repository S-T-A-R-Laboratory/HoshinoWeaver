import json
from unittest import mock


from hoshicore._custom_op import (
    build_info,
    custom_ops_available,
)
import hoshicore._custom_op.backend_registry as backend_registry
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.cuda_memory import cuda_chunk_memory_model
from hoshicore._custom_op.cuda_memory import cuda_memory_estimate
from hoshicore._custom_op.cuda_memory import cuda_memory_model_kind
from hoshicore._custom_op.metal_memory import metal_memory_estimate
from hoshicore._custom_op.metal_memory import metal_memory_model_kind
import hoshicore._custom_op.ops.max as max_ops


from tests.custom_ops._base import CustomOpsTestCase


class TestBackendRegistry(CustomOpsTestCase):
    def test_custom_ops_available_returns_bool(self) -> None:
        self.assertIsInstance(custom_ops_available(), bool)

    def test_build_info_returns_minimal_metadata(self) -> None:
        info = build_info()
        self.assertIsInstance(info, dict)
        self.assertIn("available", info)

    def test_build_info_reports_fallback_backend(self) -> None:
        with mock.patch.object(
            max_ops, "_load_compiled_module_result", return_value=(None, "mock error")
        ):
            max_ops._select_max_backend.cache_clear()
            info = max_ops.build_info()

        self.assertFalse(info["available"])
        self.assertEqual(info["backend"], "numpy")

    def test_build_info_includes_thread_policy(self) -> None:
        info = build_info()
        self.assertIn("thread_policy", info)

    def test_backend_registry_exposes_registered_candidates(self) -> None:
        candidates = backend_registry.registered_backend_candidates(
            "median_reduce_chunk"
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].logical_op, "median_reduce_chunk")
        self.assertEqual(candidates[0].backend, "openmp_cpu")
        self.assertEqual(candidates[0].kernel_name, "median_reduce_chunk")

        filter_candidates = backend_registry.registered_backend_candidates(
            "median_filter_2d"
        )
        self.assertEqual(len(filter_candidates), 1)
        self.assertEqual(filter_candidates[0].backend, "openmp_cpu")
        self.assertEqual(filter_candidates[0].kernel_name, "median_filter_2d")

        feature_candidates = backend_registry.registered_backend_candidates(
            "extract_point_features"
        )
        self.assertEqual(
            {(candidate.backend, candidate.kernel_name) for candidate in feature_candidates},
            {
                ("cuda_host_io", "extract_point_features_cuda"),
                ("openmp_cpu", "extract_point_features"),
            },
        )

        asterism_candidates = backend_registry.registered_backend_candidates(
            "asterism_mutual_nearest"
        )
        self.assertEqual(
            {(candidate.backend, candidate.kernel_name) for candidate in asterism_candidates},
            {
                ("cuda_host_io", "asterism_mutual_nearest_cuda"),
                ("openmp_cpu", "asterism_mutual_nearest_cpu"),
            },
        )

        matching_candidates = backend_registry.registered_backend_candidates(
            "matching_cosine_bidirectional_nearest"
        )
        self.assertEqual(len(matching_candidates), 2)
        self.assertEqual(
            {candidate.backend for candidate in matching_candidates},
            {"cuda_host_io", "openmp_cpu"},
        )

        remap_candidates = backend_registry.registered_backend_candidates(
            "camera_model_remap"
        )
        self.assertTrue(
            any(
                candidate.backend == "cuda_host_io"
                and candidate.kernel_name == "camera_model_remap"
                for candidate in remap_candidates
            )
        )
        self.assertTrue(
            any(
                candidate.backend == "openmp_cpu"
                and candidate.kernel_name == "camera_model_remap_cpu"
                for candidate in remap_candidates
            )
        )

        huber_chunk_candidates = backend_registry.registered_backend_candidates(
            "huber_weighted_chunk"
        )
        self.assertEqual(len(huber_chunk_candidates), 1)
        self.assertEqual(huber_chunk_candidates[0].backend, "cuda_host_io")
        self.assertEqual(
            huber_chunk_candidates[0].kernel_name, "huber_weighted_chunk_cuda"
        )

        process_candidates = backend_registry.registered_backend_candidates(
            "star_shrink_process"
        )
        self.assertEqual(
            {candidate.backend for candidate in process_candidates},
            {"metal_host_io", "cuda_host_io", "openmp_cpu"},
        )
        metal_process = next(
            candidate
            for candidate in process_candidates
            if candidate.backend == "metal_host_io"
        )
        self.assertEqual(metal_process.module_key, "metal")
        self.assertEqual(metal_process.build_flag, "metal")

    def test_candidates_without_numpy_fallback_declare_it_explicitly(self) -> None:
        """Ops whose last resort lives outside the registry must not claim numpy."""
        detect_candidates = backend_registry.registered_backend_candidates(
            "star_detect_fused_pixel_components"
        )
        cpu_detect = [
            candidate
            for candidate in detect_candidates
            if candidate.backend == "openmp_cpu"
        ]
        self.assertEqual(len(cpu_detect), 1)
        self.assertIsNone(cpu_detect[0].fallback)

        dog_candidates = backend_registry.registered_backend_candidates(
            "star_shrink_dog_process"
        )
        cuda_dog = [
            candidate
            for candidate in dog_candidates
            if candidate.backend == "cuda_host_io"
        ]
        self.assertEqual(len(cuda_dog), 1)
        self.assertIsNone(cuda_dog[0].fallback)

    def test_builtin_cuda_candidates_declare_consumed_memory_models(self) -> None:
        candidates = backend_registry.registered_backend_candidates()

        backend_registry.validate_cuda_memory_policy_declarations(
            candidates,
            require_memory_models=True,
        )
        cuda_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.backend == "cuda_host_io"
        )
        self.assertTrue(cuda_candidates)
        self.assertTrue(
            all(candidate.memory_model is not None for candidate in cuda_candidates)
        )
        for candidate in cuda_candidates:
            self.assertEqual(
                candidate.memory_model,
                cuda_memory_model_kind(candidate.logical_op),
            )
        self.assertEqual(
            {candidate.logical_op for candidate in cuda_candidates},
            {
                "asterism_mutual_nearest",
                "camera_model_remap",
                "extract_point_features",
                "huber_weighted_chunk",
                "matching_cosine_bidirectional_nearest",
                "sigma_clip_fused_chunk",
                "star_detect_fused_pixel_components",
                "star_mask_dog",
                "star_shrink_dog_process",
                "star_shrink_process",
                "wavelet_dec_rec",
                "wavelet_dec_rec_cuda_core",
            },
        )

    def test_registered_cuda_chunk_models_are_consumable(self) -> None:
        chunk_candidates = (
            candidate
            for candidate in backend_registry.registered_backend_candidates()
            if candidate.memory_model == "cuda_chunk"
        )

        for candidate in chunk_candidates:
            model = cuda_chunk_memory_model(
                candidate.logical_op,
                n_frames=4,
                row_bytes=128,
                dtype_bytes=2,
            )
            self.assertGreater(model.device_bytes_per_row, 0)

    def test_builtin_metal_candidates_declare_consumed_memory_models(self) -> None:
        candidates = backend_registry.registered_backend_candidates()

        backend_registry.validate_metal_memory_policy_declarations(
            candidates,
            require_memory_models=True,
        )
        metal_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.backend == "metal_host_io"
        )
        # Pins which ops have a Metal kernel, so widening coverage is deliberate.
        self.assertEqual(
            sorted(candidate.logical_op for candidate in metal_candidates),
            ["median_star_mask", "star_mask_dog", "star_shrink_dog_process", "star_shrink_process"],
        )
        extra_estimate_args = {
            "star_mask_dog": {"small_kernel_size": 9, "large_kernel_size": 73},
            "star_shrink_dog_process": {
                "small_kernel_size": 9,
                "large_kernel_size": 73,
            },
        }
        for candidate in metal_candidates:
            with self.subTest(logical_op=candidate.logical_op):
                self.assertEqual(
                    candidate.memory_model,
                    metal_memory_model_kind(candidate.logical_op),
                )
                estimate = metal_memory_estimate(
                    candidate.logical_op,
                    height=32,
                    width=48,
                    channels=1 if candidate.logical_op == "median_star_mask" else 3,
                    dtype_bytes=2,
                    **extra_estimate_args.get(candidate.logical_op, {}),
                )
                self.assertGreater(estimate.peak_device_bytes, 0)

    def test_registered_cuda_non_chunk_models_are_consumable(self) -> None:
        sample_args = {
            "asterism_mutual_nearest": {"n1": 17, "n2": 19},
            "camera_model_remap": {
                "source_height": 8,
                "source_width": 10,
                "channels": 3,
                "dtype_bytes": 2,
                "out_height": 6,
                "out_width": 7,
            },
            "extract_point_features": {"n_points": 40, "k": 8},
            "matching_cosine_bidirectional_nearest": {
                "n1": 17,
                "n2": 19,
                "feature_dim": 11,
            },
            "star_detect_fused_pixel_components": {
                "height": 64,
                "width": 80,
                "small_height": 16,
                "small_width": 20,
                "level": 2,
                "gaussian_ksize": 9,
            },
            "star_mask_dog": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
                "small_kernel_size": 9,
                "large_kernel_size": 73,
            },
            "star_shrink_dog_process": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
                "small_kernel_size": 9,
                "large_kernel_size": 73,
            },
            "star_shrink_process": {
                "height": 32,
                "width": 48,
                "channels": 3,
                "dtype_bytes": 2,
            },
            "wavelet_dec_rec": {"height": 64, "width": 80, "level": 2},
            "wavelet_dec_rec_cuda_core": {
                "height": 64,
                "width": 80,
                "level": 2,
            },
        }
        candidates = (
            candidate
            for candidate in backend_registry.registered_backend_candidates()
            if candidate.backend == "cuda_host_io"
            and candidate.memory_model != "cuda_chunk"
        )

        for candidate in candidates:
            estimate = cuda_memory_estimate(
                candidate.logical_op,
                **sample_args[candidate.logical_op],
            )
            self.assertEqual(estimate.logical_op, candidate.logical_op)
            self.assertGreater(estimate.peak_device_bytes, 0)

    def test_cuda_candidate_without_memory_declaration_is_rejected(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "missing_memory_policy",
            "cuda_host_io",
            "missing_memory_policy_cuda",
            build_flag="cuda",
        )

        with self.assertRaisesRegex(RuntimeError, "must declare"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,))

    def test_explicit_third_party_memory_deferral_is_compatible_but_not_builtin(
        self,
    ) -> None:
        candidate = backend_registry.BackendCandidate(
            "third_party_cuda_op",
            "cuda_host_io",
            "third_party_cuda_op_kernel",
            build_flag="cuda",
            memory_model_reason="third-party compatibility declaration",
        )

        backend_registry.validate_cuda_memory_policy_declarations((candidate,))
        with self.assertRaisesRegex(RuntimeError, "must declare a consumed"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,),
                require_memory_models=True,
            )

    def test_cuda_candidate_with_unknown_memory_model_is_rejected(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "unknown_memory_model",
            "cuda_host_io",
            "unknown_memory_model_cuda",
            build_flag="cuda",
            memory_model="typo",
        )

        with self.assertRaisesRegex(RuntimeError, "unknown memory model"):
            backend_registry.validate_cuda_memory_policy_declarations(
                (candidate,))

    def test_cuda_candidate_model_must_resolve_to_an_estimator(self) -> None:
        candidate = backend_registry.BackendCandidate(
            "missing_static_estimator",
            "cuda_host_io",
            "missing_static_estimator_cuda",
            build_flag="cuda",
            memory_model="static_estimator",
        )

        with self.assertRaisesRegex(RuntimeError, "without a registered estimator"):
            backend_registry.validate_cuda_memory_policy_declarations((candidate,))

    def test_backend_registry_reports_missing_compiled_module(self) -> None:
        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (None, "mock import error"),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "mock import error")

    def test_backend_registry_selects_native_kernel(self) -> None:
        class Module:
            pass

        module = Module()
        module.median_reduce_chunk = lambda stack: stack

        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertIs(selection.module, module)
        self.assertEqual(selection.backend, "openmp_cpu")

    def test_backend_registry_selects_metal_from_separate_module(self) -> None:
        class MetalModule:
            def star_shrink_process_metal(self):
                return None

            def build_info(self):
                return {"metal": True}

        class CompiledModule:
            def star_shrink_process(self):
                return None

            def build_info(self):
                return {"cuda": False, "openmp": True}

        metal_module = MetalModule()
        selection = backend_registry.resolve_backend(
            "star_shrink_process",
            load_module=lambda: (CompiledModule(), None),
            module_loaders={"metal": lambda: (metal_module, None)},
            metal_probe=lambda: {
                "available": True,
                "status": "available",
            },
        )

        self.assertTrue(selection.native)
        self.assertIs(selection.module, metal_module)
        self.assertEqual(selection.backend, "metal_host_io")

    def test_metal_runtime_unavailable_falls_back_to_openmp(self) -> None:
        class MetalModule:
            def star_shrink_process_metal(self):
                return None

            def build_info(self):
                return {"metal": True}

        class CompiledModule:
            def star_shrink_process(self):
                return None

            def build_info(self):
                return {"cuda": False, "openmp": True}

        selection = backend_registry.resolve_backend(
            "star_shrink_process",
            load_module=lambda: (CompiledModule(), None),
            module_loaders={"metal": lambda: (MetalModule(), None)},
            metal_probe=lambda: {
                "available": False,
                "status": "explicitly_unavailable",
                "reason_code": "metal_unified_memory_required",
                "reason": "unified memory required",
            },
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "metal_unified_memory_required")
        self.assertEqual(
            selection.decision.reason_code,
            "metal_unified_memory_required",
        )

    def test_backend_registry_respects_build_flag(self) -> None:
        class Module:
            def wavelet_dec_rec_cuda_core(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec_cuda_core",
            load_module=lambda: (module, None),
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason, "compiled backend missing build flag: cuda")

    def test_backend_registry_selects_remap_cpu_after_cuda_build_flag_miss(
        self,
    ) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "camera_model_remap",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "camera_model_remap_cpu")

    @staticmethod
    def _fallback_helper_selections():
        metal = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "star_mask_dog", "metal_host_io", "star_mask_dog_metal",
                module_key="metal", build_flag="metal",
            ),
            object(),
        )
        cpu = backend_registry.BackendSelection(
            backend_registry.BackendCandidate(
                "star_mask_dog", "openmp_cpu", "star_mask_dog_cpu",
            ),
            object(),
        )
        return metal, cpu

    def test_fallback_helper_propagates_non_accelerator_errors(self) -> None:
        _, cpu = self._fallback_helper_selections()

        def map_backend(selection):
            return "cpu", mock.Mock(side_effect=RuntimeError("cpu kernel failed"))

        with self.assertRaisesRegex(RuntimeError, "cpu kernel failed"):
            backend_registry.run_with_accelerator_fallback(
                "star_mask_dog", cpu, map_backend, lambda fn: fn()
            )

    def test_fallback_helper_retries_the_mapped_fallback_once(self) -> None:
        metal, cpu = self._fallback_helper_selections()
        metal_entry = mock.Mock(
            side_effect=CustomOpResourceExhaustedError("admission denied"))
        cpu_entry = mock.Mock(return_value="cpu result")

        def map_backend(selection):
            if selection.candidate.backend == "metal_host_io":
                return "metal", metal_entry
            return "cpu", cpu_entry

        with mock.patch.object(
            backend_registry,
            "resolve_after_resource_exhausted",
            return_value=cpu,
        ):
            got = backend_registry.run_with_accelerator_fallback(
                "star_mask_dog", metal, map_backend, lambda fn: fn()
            )

        self.assertEqual(got, "cpu result")
        metal_entry.assert_called_once()
        cpu_entry.assert_called_once()

    def test_fallback_helper_rejects_a_same_backend_fallback(self) -> None:
        metal, _ = self._fallback_helper_selections()

        def map_backend(selection):
            return "metal", mock.Mock(
                side_effect=CustomOpResourceExhaustedError("admission denied"))

        with (
            mock.patch.object(
                backend_registry,
                "resolve_after_resource_exhausted",
                return_value=metal,
            ),
            self.assertRaisesRegex(RuntimeError, "remained selected"),
        ):
            backend_registry.run_with_accelerator_fallback(
                "star_mask_dog", metal, map_backend, lambda fn: fn()
            )

    def test_backend_registry_can_exclude_failed_runtime_backend(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        module = Module()

        selection = backend_registry.select_backend(
            "camera_model_remap",
            load_module=lambda: (module, None),
            exclude_backends={"cuda_host_io"},
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "camera_model_remap_cpu")

    def test_cpu_preference_selects_openmp_without_cuda_probe(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        probe = mock.Mock(side_effect=AssertionError("CUDA must not be probed"))
        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            "cpu",
            load_module=lambda: (Module(), None),
            cuda_probe=probe,
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "forced_cpu")
        self.assertEqual(selection.decision.reason_code, "forced_cpu")
        probe.assert_not_called()

    def test_cpu_preference_does_not_load_or_probe_metal(self) -> None:
        class Module:
            def star_shrink_process(self):
                return None

            def build_info(self):
                return {"openmp": True}

        metal_loader = mock.Mock(
            side_effect=AssertionError("Metal module must not load")
        )
        metal_probe = mock.Mock(
            side_effect=AssertionError("Metal runtime must not be probed")
        )
        selection = backend_registry.resolve_backend(
            "star_shrink_process",
            "cpu",
            load_module=lambda: (Module(), None),
            module_loaders={"metal": metal_loader},
            metal_probe=metal_probe,
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        metal_loader.assert_not_called()
        metal_probe.assert_not_called()

    def test_cpu_preference_uses_numpy_for_cuda_only_op(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("module must not load"))
        probe = mock.Mock(side_effect=AssertionError("CUDA must not be probed"))

        selection = backend_registry.resolve_backend(
            "huber_weighted_chunk",
            "cpu",
            load_module=loader,
            cuda_probe=probe,
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason_code, "forced_cpu")
        self.assertEqual(selection.decision.reason_code, "forced_cpu")
        loader.assert_not_called()
        probe.assert_not_called()

    def test_cpu_preference_never_selects_registered_cuda_candidate(self) -> None:
        class Module:
            def build_info(self):
                return {"cuda": True}

        module = Module()
        for candidate in backend_registry.registered_backend_candidates():
            setattr(Module, candidate.kernel_name, lambda self: None)

        logical_ops = {
            candidate.logical_op
            for candidate in backend_registry.registered_backend_candidates()
        }
        for logical_op in logical_ops:
            selection = backend_registry.select_backend(
                logical_op,
                "cpu",
                load_module=lambda: (module, None),
            )
            self.assertNotEqual(selection.backend, "cuda_host_io", logical_op)

    def test_backend_registry_reports_when_all_candidates_are_excluded(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        module = Module()

        selection = backend_registry.select_backend(
            "huber_weighted_chunk",
            load_module=lambda: (module, None),
            exclude_backends={"cuda_host_io"},
        )

        self.assertFalse(selection.native)
        self.assertEqual(selection.backend, "numpy")
        self.assertIn("after excluding: cuda_host_io", selection.reason)

    def test_backend_registry_continues_after_build_flag_miss(self) -> None:
        class Module:
            def wavelet_dec_rec_cuda_core(self):
                return None

            def wavelet_dec_rec_cpu(self):
                return None

            def build_info(self):
                return {"cuda": False}

        module = Module()

        selection = backend_registry.select_backend(
            "wavelet_dec_rec",
            load_module=lambda: (module, None),
        )

        self.assertTrue(selection.native)
        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.candidate.kernel_name, "wavelet_dec_rec_cpu")

    def test_backend_decision_is_serializable_and_sanitized(self) -> None:
        class Module:
            def median_reduce_chunk(self):
                return None

        selection = backend_registry.select_backend(
            "median_reduce_chunk",
            load_module=lambda: (Module(), None),
        )

        payload = selection.to_decision("median_reduce_chunk").to_dict()

        json.dumps(payload)
        self.assertEqual(payload["backend"], "openmp_cpu")
        self.assertEqual(payload["reason_code"], "selected_native")
        self.assertNotIn("module", payload)
        self.assertNotIn("reason_detail", payload)

    def test_runtime_resolver_keeps_available_cuda_candidate(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            load_module=lambda: (Module(), None),
            cuda_probe=lambda: {
                "available": True,
                "status": "available",
                "free_bytes": 1024,
                "total_bytes": 2048,
            },
        )

        self.assertEqual(selection.backend, "cuda_host_io")
        self.assertIsNotNone(selection.decision)
        self.assertEqual(selection.decision.reason_code, "selected_native")

    def test_runtime_resolver_falls_back_to_cpu_when_cuda_unavailable(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            load_module=lambda: (Module(), None),
            cuda_probe=lambda: {
                "available": False,
                "status": "explicitly_unavailable",
                "reason": "mock no device",
            },
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_runtime_unavailable")
        self.assertEqual(selection.decision.reason_code, "cuda_runtime_unavailable")

    def test_runtime_resolver_preserves_unsupported_device_reason(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            load_module=lambda: (Module(), None),
            cuda_probe=lambda: {
                "available": False,
                "status": "explicitly_unavailable",
                "reason_code": "cuda_compute_capability_unsupported",
                "reason": "CUDA compute capability 5.2 is unsupported",
            },
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(
            selection.reason_code,
            "cuda_compute_capability_unsupported",
        )
        self.assertEqual(
            selection.decision.reason_code,
            "cuda_compute_capability_unsupported",
        )

    def test_runtime_resolver_does_not_hide_probe_errors(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_backend(
                "huber_weighted_chunk",
                load_module=lambda: (Module(), None),
                cuda_probe=lambda: (_ for _ in ()).throw(
                    RuntimeError("cudaMemGetInfo: out of memory")
                ),
            )

    def test_runtime_resolver_rejects_generic_unavailable_status(self) -> None:
        class Module:
            def camera_model_remap(self):
                return None

            def camera_model_remap_cpu(self):
                return None

            def build_info(self):
                return {"cuda": True}

        with self.assertRaisesRegex(RuntimeError, "without an unavailable status"):
            backend_registry.resolve_backend(
                "camera_model_remap",
                load_module=lambda: (Module(), None),
                cuda_probe=lambda: {
                    "available": False,
                    "status": "unavailable",
                    "reason": "out of memory",
                },
            )

    def test_forced_numpy_skips_module_load_and_cuda_probe(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("module should not load"))
        probe = mock.Mock(side_effect=AssertionError("CUDA should not probe"))

        selection = backend_registry.resolve_backend(
            "camera_model_remap",
            "numpy",
            load_module=loader,
            cuda_probe=probe,
        )

        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.decision.reason_code, "forced_numpy")
        loader.assert_not_called()
        probe.assert_not_called()

    def test_runtime_retry_rejects_resource_errors_before_reresolve(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("resolver should not run"))

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_after_runtime_unavailable(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                RuntimeError("cudaMalloc: out of memory"),
                load_module=loader,
            )

        loader.assert_not_called()

    def test_resource_resolver_selects_cpu_with_distinct_reason(self) -> None:
        class Module:
            def sigma_clip_fused_chunk_cuda(self):
                return None

            def sigma_clip_fused_chunk(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_after_resource_exhausted(
            "sigma_clip_fused_chunk",
            "cuda_host_io",
            CustomOpResourceExhaustedError("estimated VRAM is insufficient"),
            load_module=lambda: (Module(), None),
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_resource_exhausted")
        self.assertEqual(
            selection.decision.reason_code, "cuda_resource_exhausted")

    def test_metal_resource_resolver_selects_openmp(self) -> None:
        class CompiledModule:
            def star_shrink_process(self):
                return None

            def build_info(self):
                return {"openmp": True}

        selection = backend_registry.resolve_after_resource_exhausted(
            "star_shrink_process",
            "metal_host_io",
            CustomOpResourceExhaustedError("working set is insufficient"),
            load_module=lambda: (CompiledModule(), None),
            module_loaders={"metal": lambda: (None, "failed Metal module")},
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "metal_resource_exhausted")

    def test_resource_resolver_selects_numpy_without_cpu_candidate(self) -> None:
        class Module:
            def huber_weighted_chunk_cuda(self):
                return None

            def build_info(self):
                return {"cuda": True}

        selection = backend_registry.resolve_after_resource_exhausted(
            "huber_weighted_chunk",
            "cuda_host_io",
            CustomOpResourceExhaustedError("estimated VRAM is insufficient"),
            load_module=lambda: (Module(), None),
        )

        self.assertEqual(selection.backend, "numpy")
        self.assertEqual(selection.reason_code, "cuda_resource_exhausted")
        self.assertEqual(
            selection.decision.reason_code, "cuda_resource_exhausted")

    def test_resource_resolver_rejects_unstructured_oom_errors(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("resolver should not run"))

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            backend_registry.resolve_after_resource_exhausted(
                "sigma_clip_fused_chunk",
                "cuda_host_io",
                RuntimeError("cudaMalloc: out of memory"),
                load_module=loader,
            )

        loader.assert_not_called()

    def test_cuda_failure_resolver_classifies_resource_exhaustion(self) -> None:
        class Module:
            def sigma_clip_fused_chunk_cuda(self):
                return None

            def sigma_clip_fused_chunk(self):
                return None

            def build_info(self):
                return {"cuda": True}

        log = mock.Mock()
        selection = backend_registry.resolve_after_cuda_failure(
            "sigma_clip_fused_chunk",
            CustomOpResourceExhaustedError("estimated VRAM is insufficient"),
            load_module=lambda: (Module(), None),
            log=log,
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_resource_exhausted")
        log.assert_called_once()
        self.assertIn("exhausted resources", log.call_args.args[0])

    def test_cuda_failure_resolver_classifies_runtime_unavailability(self) -> None:
        class Module:
            def sigma_clip_fused_chunk_cuda(self):
                return None

            def sigma_clip_fused_chunk(self):
                return None

            def build_info(self):
                return {"cuda": True}

        log = mock.Mock()
        selection = backend_registry.resolve_after_cuda_failure(
            "sigma_clip_fused_chunk",
            RuntimeError("no CUDA-capable device is detected"),
            load_module=lambda: (Module(), None),
            log=log,
        )

        self.assertEqual(selection.backend, "openmp_cpu")
        self.assertEqual(selection.reason_code, "cuda_runtime_unavailable")
        log.assert_called_once()
        self.assertIn("unavailable at runtime", log.call_args.args[0])

    def test_cuda_failure_resolver_reraises_unclassified_errors(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("resolver should not run"))
        log = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            backend_registry.resolve_after_cuda_failure(
                "sigma_clip_fused_chunk",
                RuntimeError("an illegal memory access was encountered"),
                load_module=loader,
                log=log,
            )

        loader.assert_not_called()
        log.assert_not_called()
