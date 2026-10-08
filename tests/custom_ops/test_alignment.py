from unittest import mock

import numpy as np

from hoshicore._custom_op import extract_point_features
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendCandidate
from hoshicore._custom_op.backend_registry import BackendSelection
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.ops.alignment as alignment_ops
import hoshicore.component.norma.matching as norma_matching


from tests.custom_ops._base import CustomOpsTestCase


def _make_alignment_match_inputs(seed: int = 0, n_points: int = 96):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_points, 3))
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    vec2 = vec + rng.normal(scale=1e-4, size=vec.shape)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    vol = rng.random(n_points) * 10 + 1
    vol2 = vol * (1.0 + rng.normal(scale=1e-3, size=n_points))
    pts = rng.random((n_points, 2)) * 1000
    pts2 = pts + rng.normal(scale=1.0, size=pts.shape)
    return vec, vec2, vol, vol2, pts, pts2


class TestAlignmentCustomOps(CustomOpsTestCase):
    def test_extract_point_features_compiled_matches_numpy(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=1)

        got = alignment_ops.extract_point_features_compiled(vec, vol, k=8)
        expected = alignment_ops.extract_point_features_numpy(vec, vol, k=8)

        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_extract_point_features_compiled_handles_duplicate_points(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=5, n_points=48)
        # Exact duplicates create similarity ties in the neighbor selection;
        # duplicated points must still get identical, finite descriptors.
        vec[24:] = vec[:24]
        vol[24:] = vol[:24]

        got = alignment_ops.extract_point_features_compiled(vec, vol, k=8)

        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_array_equal(got[24:], got[:24])
        rerun = alignment_ops.extract_point_features_compiled(vec, vol, k=8)
        np.testing.assert_array_equal(got, rerun)

    def test_extract_point_features_can_force_numpy_fallback(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=3)

        with mock.patch.dict(
            "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False
        ):
            with mock.patch.object(
                alignment_ops,
                "_load_compiled_module_result",
                return_value=(None, "mock error"),
            ):
                features1 = extract_point_features(vec, vol, k=8)

        expected_features1 = alignment_ops.extract_point_features_numpy(vec, vol, k=8)
        np.testing.assert_allclose(
            features1, expected_features1, rtol=1e-10, atol=1e-12
        )

    def test_extract_point_features_public_dispatch_uses_compiled_backend(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=4)

        with mock.patch.object(
            alignment_ops,
            "extract_point_features_compiled",
            wraps=alignment_ops.extract_point_features_compiled,
        ) as patched_extract:
            with mock.patch.dict(
                "os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "cpu"}, clear=False
            ):
                _ = extract_point_features(vec, vol, k=8)

        patched_extract.assert_called_once()

    def test_extract_point_features_grid_path_matches_numpy(self) -> None:
        # N >= 512 takes the direction-grid candidate search.
        rng = np.random.default_rng(11)
        z = rng.uniform(np.cos(np.radians(15.0)), 1.0, 1500)
        phi = rng.uniform(0.0, 2.0 * np.pi, 1500)
        radius = np.sqrt(1.0 - z * z)
        vec = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
        vol = rng.uniform(0.1, 5.0, len(vec))

        got = alignment_ops.extract_point_features_compiled(vec, vol, k=15)
        expected = alignment_ops.extract_point_features_numpy(vec, vol, k=15)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

        duplicated = alignment_ops.extract_point_features_compiled(
            np.vstack((vec, vec[:200])), np.concatenate((vol, vol[:200])), k=15)
        self.assertTrue(np.all(np.isfinite(duplicated)))
        np.testing.assert_array_equal(duplicated[1500:], duplicated[:200])

    def test_extract_point_features_cuda_matches_cpu_when_available(self) -> None:
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        if not module.build_info().get("cuda"):
            self.skipTest("compiled extension is CPU-only")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))
        rng = np.random.default_rng(12)
        cap = rng.normal(size=(1500, 3)) * [0.1, 0.1, 1.0] + [0.0, 0.0, 4.0]
        zero_norm = cap[:600].copy()
        zero_norm[7] = 0.0
        cases = {
            "grid": (np.vstack((cap, cap[:100])), 15),
            "full_scan": (cap[:40], 8),
            "zero_norm": (zero_norm, 15),
        }
        for name, (vec, k) in cases.items():
            with self.subTest(name=name):
                vol = rng.uniform(0.1, 5.0, len(vec))
                cpu = alignment_ops.extract_point_features_compiled(vec, vol, k=k)
                cuda = alignment_ops.extract_point_features_cuda(vec, vol, k=k)
                # Neighbour choice is exact; only device acos/atan2/exp
                # rounding separates the backends.
                np.testing.assert_allclose(cuda, cpu, rtol=0.0, atol=1e-12)

        vec = cap[:100]
        vol = rng.uniform(0.1, 5.0, len(vec))
        self.assertIsNone(module.extract_point_features_cuda(vec, vol, 17))
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False):
            dispatched = extract_point_features(vec, vol, k=17)
        np.testing.assert_array_equal(
            dispatched, alignment_ops.extract_point_features_compiled(vec, vol, k=17))

    def test_extract_point_features_grid_is_disabled_for_tiny_norms(self) -> None:
        # Squared norms near underflow make unit vectors too inexact for the
        # grid's margin; the result must equal the full scan.
        rng = np.random.default_rng(193)
        vec = rng.normal(size=(600, 3)) * [0.1, 0.1, 0.1] + [0.0, 0.0, 1.0]
        vec = vec / np.linalg.norm(vec, axis=1)[:, None] * 1e-160
        vol = rng.uniform(0.1, 5.0, 600)

        grid_candidate = alignment_ops.extract_point_features_compiled(vec, vol, k=15)
        # A zero vector disables the grid; its zero cosine never enters the
        # positive-cosine pools of the other points.
        full_scan = alignment_ops.extract_point_features_compiled(
            np.vstack((vec, np.zeros((1, 3)))), np.append(vol, 0.0), k=15)[:600]

        np.testing.assert_array_equal(grid_candidate, full_scan)

    def test_extract_point_features_cuda_declines_near_tied_ordering(self) -> None:
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            self.skipTest(error or "compiled custom ops unavailable")
        if not module.build_info().get("cuda"):
            self.skipTest("compiled extension is CPU-only")
        if not module.cuda_memory_info().get("available"):
            self.skipTest("CUDA runtime unavailable")
        rng = np.random.default_rng(17)
        vec = rng.normal(size=(4, 3))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        # Weights 1/rho make every neighbour's vol*rho key equal on the host,
        # so the device acos rounding could reorder them.
        norms = np.sqrt([np.dot(v, v) for v in vec])
        cosines = [np.dot(vec[0], v) / (norms[0] * n) for v, n in zip(vec, norms)]
        rho = np.arccos(np.clip(cosines, -1.0, 1.0))
        vol = np.concatenate(([1.0], 1.0 / rho[1:]))

        self.assertIsNone(module.extract_point_features_cuda(vec, vol, 2))
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "auto"}, clear=False):
            dispatched = extract_point_features(vec, vol, k=2)
        np.testing.assert_array_equal(
            dispatched, alignment_ops.extract_point_features_compiled(vec, vol, k=2))

    def test_extract_point_features_cuda_resource_error_falls_back_to_cpu(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=6)
        cuda = BackendSelection(
            BackendCandidate(
                "extract_point_features", "cuda_host_io", "extract_point_features_cuda"),
            object(),
        )
        cpu = BackendSelection(
            BackendCandidate("extract_point_features", "openmp_cpu", "extract_point_features"),
            object(),
        )
        with mock.patch.object(alignment_ops, "_resolve_backend", return_value=cuda):
            with mock.patch.object(
                alignment_ops,
                "extract_point_features_cuda",
                side_effect=CustomOpResourceExhaustedError("estimated VRAM"),
            ):
                with mock.patch.object(
                    backend_registry, "resolve_after_resource_exhausted", return_value=cpu
                ) as resolve:
                    with mock.patch.object(
                        alignment_ops,
                        "extract_point_features_compiled",
                        wraps=alignment_ops.extract_point_features_compiled,
                    ) as compiled:
                        extract_point_features(vec, vol, k=8)

        resolve.assert_called_once()
        compiled.assert_called_once()

    def test_norma_feature_extraction_routes_through_custom_op(self) -> None:
        vec, _, vol, _, _, _ = _make_alignment_match_inputs(seed=5)

        with mock.patch.object(
                norma_matching,
                "custom_extract_point_features",
                wraps=norma_matching.custom_extract_point_features) as extract:
            features1 = norma_matching.extract_point_features(vec, vol, k=8)

        extract.assert_called_once_with(vec, vol, k=8)

        expected_features1 = alignment_ops.extract_point_features_numpy(vec, vol, k=8)
        np.testing.assert_allclose(
            features1, expected_features1, rtol=1e-10, atol=1e-12
        )
