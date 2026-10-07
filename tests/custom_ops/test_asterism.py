import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np

from hoshicore._custom_op import asterism_anchor_votes
from hoshicore._custom_op import asterism_mutual_nearest
from hoshicore._custom_op import asterism_tokens
from hoshicore._custom_op import cuda_memory
from hoshicore._custom_op._dispatch import CustomOpResourceExhaustedError
from hoshicore._custom_op.backend_registry import BackendCandidate
from hoshicore._custom_op.backend_registry import BackendSelection
import hoshicore._custom_op.backend_registry as backend_registry
import hoshicore._custom_op.ops.alignment as alignment_ops

from tests.custom_ops._base import CustomOpsTestCase


LOGICAL_OP = "asterism_mutual_nearest"


def _selection(backend: str) -> BackendSelection:
    kernel_name = {
        "cuda_host_io": "asterism_mutual_nearest_cuda",
        "openmp_cpu": "asterism_mutual_nearest_cpu",
    }[backend]
    return BackendSelection(BackendCandidate(LOGICAL_OP, backend, kernel_name), object())


def _token_sets(seed: int, n1: int = 3000, n2: int = 2800) -> tuple[np.ndarray, np.ndarray]:
    """Clustered tokens where most set-1 tokens have a perturbed partner."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20.0, 20.0, size=(40, 3))
    values1 = centers[rng.integers(0, len(centers), n1)] + rng.normal(scale=3.0, size=(n1, 3))
    partners = values1[rng.permutation(n1)[:n2]]
    values2 = partners + rng.normal(scale=0.3, size=partners.shape)
    return values1, values2


def _assert_pairs_equal(got, expected) -> None:
    for got_array, expected_array in zip(got, expected):
        np.testing.assert_array_equal(got_array, expected_array)
        assert got_array.dtype == np.int64


class TestAsterismMutualNearest(CustomOpsTestCase):
    @staticmethod
    def _compiled_module():
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_compiled_cpu_matches_reference(self) -> None:
        self._compiled_module()
        for seed, threshold in ((1, 1.0), (2, 0.5), (3, 2.5)):
            with self.subTest(seed=seed, threshold=threshold):
                values1, values2 = _token_sets(seed)
                expected = alignment_ops.asterism_mutual_nearest_numpy(values1, values2, threshold)
                got = alignment_ops.asterism_mutual_nearest_cpu_compiled(values1, values2, threshold)
                self.assertGreater(len(expected[0]), 0)
                _assert_pairs_equal(got, expected)

    def test_cpu_result_is_thread_count_independent(self) -> None:
        module = self._compiled_module()
        values1, values2 = _token_sets(4, n1=6000, n2=6000)
        results = []
        for threads in (1, 3):
            module.set_openmp_threads(threads)
            results.append(module.asterism_mutual_nearest_cpu(values1, values2, 1.0))
        _assert_pairs_equal(results[1], results[0])

    def test_cuda_matches_cpu_when_available(self) -> None:
        module = self._compiled_module()
        if not module.build_info().get("cuda"):
            self.skipTest("compiled extension is CPU-only")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))
        rng = np.random.default_rng(11)
        lattice = rng.permutation(
            np.stack(np.meshgrid(*[np.arange(12.0)] * 3, indexing="ij"), axis=-1).reshape(-1, 3))
        cases = {
            "clustered": _token_sets(5),
            # Nearest distances are unique here, so no tie fallback applies.
            "lattice_offset": (lattice, lattice + [0.3, 0.1, 0.05]),
            "single": (np.zeros((1, 3)), np.full((1, 3), 0.1)),
        }
        for name, (values1, values2) in cases.items():
            with self.subTest(name=name):
                cpu = module.asterism_mutual_nearest_cpu(values1, values2, 1.0)
                cuda = alignment_ops.asterism_mutual_nearest_cuda(values1, values2, 1.0)
                _assert_pairs_equal(cuda, cpu)

    def test_native_declines_nearest_distance_ties(self) -> None:
        module = self._compiled_module()
        kernels = [module.asterism_mutual_nearest_cpu]
        if module.build_info().get("cuda") and module.cuda_memory_info().get("available"):
            kernels.append(module.asterism_mutual_nearest_cuda)
        lattice = np.stack(np.meshgrid(*[np.arange(3.0)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        cases = {
            # Each offset token has eight lattice tokens at exactly equal
            # distance; SciPy's pick among them is not the smaller index.
            "equidistant": (lattice, lattice + 0.5),
            "duplicates": (np.vstack([lattice, lattice]), lattice + 0.25),
        }
        for name, (values1, values2) in cases.items():
            expected = alignment_ops.asterism_mutual_nearest_numpy(values1, values2, 1.0)
            for kernel in kernels:
                with self.subTest(name=name, kernel=kernel.__name__):
                    self.assertIsNone(kernel(values1, values2, 1.0))
            with self.subTest(name=name, path="public"):
                _assert_pairs_equal(asterism_mutual_nearest(values1, values2, 1.0), expected)

    def test_native_returns_none_when_extent_cannot_be_gridded(self) -> None:
        module = self._compiled_module()
        kernels = [module.asterism_mutual_nearest_cpu]
        if module.build_info().get("cuda") and module.cuda_memory_info().get("available"):
            kernels.append(module.asterism_mutual_nearest_cuda)
        values = _token_sets(6, n1=50, n2=50)[0]
        nonfinite = values.copy()
        nonfinite[3, 1] = np.nan
        wide = values.copy()
        wide[0] = [-1e4, -1e4, -1e4]
        # The cell count would overflow int64 if sized after conversion.
        huge = values.copy()
        huge[0] = [1e30, 0.0, 0.0]
        cases = (("nonfinite", nonfinite), ("too_many_cells", wide), ("huge_extent", huge))
        for kernel in kernels:
            for name, values1 in cases:
                with self.subTest(kernel=kernel.__name__, name=name):
                    self.assertIsNone(kernel(values1, values, 1.0))

    def test_public_falls_back_to_reference_when_not_griddable(self) -> None:
        self._compiled_module()
        values1, values2 = _token_sets(7, n1=200, n2=200)
        values1[0] = [-1e4, -1e4, -1e4]
        expected = alignment_ops.asterism_mutual_nearest_numpy(values1, values2, 1.0)
        got = asterism_mutual_nearest(values1, values2, 1.0)
        _assert_pairs_equal(got, expected)

    def test_empty_inputs_return_empty_pairs(self) -> None:
        module = self._compiled_module()
        values = np.ones((4, 3))
        empty = np.empty((0, 3))
        kernels = [module.asterism_mutual_nearest_cpu]
        if module.build_info().get("cuda") and module.cuda_memory_info().get("available"):
            kernels.append(module.asterism_mutual_nearest_cuda)
        for kernel in kernels:
            for values1, values2 in ((empty, values), (values, empty)):
                pairs1, pairs2 = kernel(values1, values2, 1.0)
                self.assertEqual((len(pairs1), len(pairs2)), (0, 0))
                self.assertEqual(pairs1.dtype, np.int64)

    def test_rejects_invalid_arguments(self) -> None:
        module = self._compiled_module()
        values = np.ones((4, 3))
        with self.assertRaises(ValueError):
            module.asterism_mutual_nearest_cpu(np.ones((4, 2)), values, 1.0)
        with self.assertRaises(ValueError):
            module.asterism_mutual_nearest_cpu(values, values, 0.0)

    def test_forced_numpy_uses_reference(self) -> None:
        values1, values2 = _token_sets(8, n1=100, n2=100)
        expected = alignment_ops.asterism_mutual_nearest_numpy(values1, values2, 1.0)
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(
                alignment_ops,
                "asterism_mutual_nearest_numpy",
                wraps=alignment_ops.asterism_mutual_nearest_numpy,
            ) as reference:
                got = asterism_mutual_nearest(values1, values2, 1.0)
        reference.assert_called_once()
        _assert_pairs_equal(got, expected)

    def test_typed_cuda_resource_error_falls_back_to_cpu(self) -> None:
        self._compiled_module()
        values1, values2 = _token_sets(9, n1=300, n2=300)
        expected = alignment_ops.asterism_mutual_nearest_numpy(values1, values2, 1.0)
        with mock.patch.object(
            alignment_ops, "_resolve_backend", return_value=_selection("cuda_host_io")
        ):
            with mock.patch.object(
                alignment_ops,
                "asterism_mutual_nearest_cuda",
                side_effect=CustomOpResourceExhaustedError("estimated VRAM"),
            ):
                with mock.patch.object(
                    backend_registry,
                    "resolve_after_resource_exhausted",
                    return_value=_selection("openmp_cpu"),
                ) as resolve:
                    got = asterism_mutual_nearest(values1, values2, 1.0)
        resolve.assert_called_once()
        _assert_pairs_equal(got, expected)

    def test_cuda_estimate_matches_workspace_high_water(self) -> None:
        module = self._compiled_module()
        if not module.build_info().get("cuda"):
            self.skipTest("compiled extension is CPU-only")
        memory_info = module.cuda_memory_info()
        if not memory_info.get("available"):
            self.skipTest(memory_info.get("reason", "CUDA runtime unavailable"))
        values1, values2 = _token_sets(10)
        both = np.vstack([values1, values2])
        cell = 1.0 * (1.0 + 1e-9)
        cells = int(np.prod(np.floor((both.max(axis=0) - both.min(axis=0)) / cell).astype(np.int64) + 1))
        estimate = cuda_memory.estimate_asterism_mutual_nearest(n1=len(values1), n2=len(values2))

        self.assertTrue(module.clear_cuda_host_io_cache())
        self.assertIsNotNone(module.asterism_mutual_nearest_cuda(values1, values2, 1.0))
        cache_info = module.cuda_host_io_cache_info()

        self.assertEqual(
            cache_info["last_device_peak_bytes"],
            cuda_memory.asterism_device_bytes(len(values1), len(values2), cells),
        )
        self.assertLessEqual(cache_info["last_device_peak_bytes"], estimate.peak_device_bytes)
        self.assertEqual(cache_info["last_pinned_peak_bytes"], 0)


def _star_vectors(seed: int, n: int, *, whole_sphere: bool = False) -> np.ndarray:
    """Unnormalized star directions over one field of view, or the whole sphere."""
    rng = np.random.default_rng(seed)
    if whole_sphere:
        return rng.normal(size=(n, 3))
    xy = rng.normal(scale=0.2, size=(n, 2))
    return np.column_stack((xy, np.ones(n))) * rng.uniform(0.5, 3.0, size=(n, 1))


def _assert_tokens_equal(got, expected) -> None:
    for got_array, expected_array in zip(got, expected):
        np.testing.assert_array_equal(got_array, expected_array)
        assert got_array.dtype == expected_array.dtype


def _assert_votes_equal(got, expected) -> None:
    np.testing.assert_array_equal(got[0], expected[0])
    assert got[0].dtype == np.int32 and got[0].shape == expected[0].shape
    assert got[1] == expected[1]
    np.testing.assert_array_equal(got[2], expected[2])
    assert got[2].dtype == np.int32


class TestAsterismTokens(CustomOpsTestCase):
    @staticmethod
    def _compiled_module():
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_compiled_cpu_matches_reference(self) -> None:
        self._compiled_module()
        cases = {
            "field": (_star_vectors(1, 3000), 8),
            "small_field": (_star_vectors(2, 40), 8),
            "fewer_stars_than_neighbours": (_star_vectors(3, 6), 8),
            "whole_sphere": (_star_vectors(4, 700, whole_sphere=True), 3),
            "wide_neighbourhood": (_star_vectors(5, 900), 12),
            "two_neighbours": (_star_vectors(6, 300), 2),
        }
        for name, (vectors, neighbor_count) in cases.items():
            with self.subTest(name=name):
                expected = alignment_ops.asterism_tokens_numpy(vectors, neighbor_count)
                got = alignment_ops.asterism_tokens_cpu_compiled(vectors, neighbor_count)
                self.assertGreater(len(expected[0]), 0)
                _assert_tokens_equal(got, expected)

    def test_cpu_result_is_thread_count_independent(self) -> None:
        module = self._compiled_module()
        vectors = _star_vectors(7, 4000)
        results = []
        for threads in (1, 3):
            module.set_openmp_threads(threads)
            results.append(module.asterism_tokens_cpu(vectors, 8))
        _assert_tokens_equal(results[1], results[0])

    def test_underflowed_grid_cell_returns_without_hanging(self) -> None:
        self._compiled_module()
        # Isolate the native call: a zero cell used to make grid sizing loop forever.
        code = textwrap.dedent("""
            import numpy as np
            from hoshicore._custom_op import _C
            from hoshicore._custom_op.ops.alignment import asterism_tokens

            vectors = np.array([[1., 0., 0.], [1., 2.3e-162, 0.], [1., 0., 2.3e-162]])
            assert _C.asterism_tokens_cpu(vectors, 2) is None
            values, anchors = asterism_tokens(vectors, 2)
            assert values.shape == (0, 3) and anchors.shape == (0,)
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_degenerate_long_edges_are_dropped(self) -> None:
        self._compiled_module()
        vectors = _star_vectors(8, 200)
        # Two stars within float64 chord resolution of star 0, at distinct
        # distances: their mutual token has a zero long edge.
        vectors[1] = vectors[0] + [1e-10, 0.0, 0.0]
        vectors[2] = vectors[0] + [0.0, 2e-10, 0.0]
        expected = alignment_ops.asterism_tokens_numpy(vectors, 8)
        got = alignment_ops.asterism_tokens_cpu_compiled(vectors, 8)
        self.assertLess(len(expected[0]), len(vectors) * 28)
        _assert_tokens_equal(got, expected)

    def test_native_declines_neighbour_distance_ties(self) -> None:
        module = self._compiled_module()
        duplicate = _star_vectors(9, 600)
        duplicate[5] = duplicate[9]
        lattice = np.stack(np.meshgrid(np.arange(20.0), np.arange(20.0), indexing="ij"),
                           axis=-1).reshape(-1, 2) * 1e-3
        cases = {
            "duplicate": duplicate,
            # Lattice neighbours sit at exactly equal distances.
            "lattice": np.column_stack((lattice, np.ones(len(lattice)))),
        }
        for name, vectors in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(module.asterism_tokens_cpu(vectors, 8))
                _assert_tokens_equal(asterism_tokens(vectors, 8),
                                     alignment_ops.asterism_tokens_numpy(vectors, 8))

    def test_rejects_invalid_arguments(self) -> None:
        module = self._compiled_module()
        with self.assertRaises(ValueError):
            module.asterism_tokens_cpu(np.ones((4, 2)), 8)
        with self.assertRaises(ValueError):
            module.asterism_tokens_cpu(np.ones((2, 3)), 8)
        vectors = _star_vectors(10, 20)
        nonfinite = vectors.copy()
        nonfinite[3, 0] = np.inf
        zero = vectors.copy()
        zero[4] = 0.0
        for name, args in (("nonfinite", (nonfinite, 8)), ("zero", (zero, 8)),
                           ("one_neighbour", (vectors, 1))):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    asterism_tokens(*args)

    def test_forced_numpy_uses_reference(self) -> None:
        vectors = _star_vectors(11, 100)
        expected = alignment_ops.asterism_tokens_numpy(vectors, 8)
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(
                alignment_ops,
                "asterism_tokens_numpy",
                wraps=alignment_ops.asterism_tokens_numpy,
            ) as reference:
                got = asterism_tokens(vectors, 8)
        reference.assert_called_once()
        _assert_tokens_equal(got, expected)


class TestAsterismAnchorVotes(CustomOpsTestCase):
    @staticmethod
    def _compiled_module():
        module, error = alignment_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_compiled_cpu_matches_reference_on_matched_fields(self) -> None:
        self._compiled_module()
        vectors = _star_vectors(20, 1500)
        rng = np.random.default_rng(21)
        kept = np.sort(rng.choice(len(vectors), 1200, replace=False))
        second = vectors[kept[rng.permutation(len(kept))]]
        tokens1 = alignment_ops.asterism_tokens_numpy(vectors, 8)
        tokens2 = alignment_ops.asterism_tokens_numpy(second, 8)
        scale = np.array([0.025, 0.025, 0.04])
        pairs1, pairs2 = alignment_ops.asterism_mutual_nearest_numpy(
            tokens1[0] / scale, tokens2[0] / scale, 1.0)
        args = (pairs1, pairs2, tokens1[1], tokens2[1], len(vectors), len(second), 5, 1)
        expected = alignment_ops.asterism_anchor_votes_numpy(*args)
        self.assertGreater(len(expected[0]), 1000)
        _assert_votes_equal(alignment_ops.asterism_anchor_votes_cpu_compiled(*args), expected)

    def test_compiled_cpu_matches_reference_with_vote_ties(self) -> None:
        self._compiled_module()
        rng = np.random.default_rng(22)
        for trial in range(200):
            num1, num2 = (int(value) for value in rng.integers(1, 30, 2))
            tokens1, tokens2 = (int(value) for value in rng.integers(1, 150, 2))
            count = int(rng.integers(0, 300))
            args = (
                rng.integers(0, tokens1, count),
                rng.integers(0, tokens2, count),
                rng.integers(0, num1, tokens1).astype(np.int32),
                rng.integers(0, num2, tokens2).astype(np.int32),
                num1,
                num2,
                int(rng.integers(1, 4)),
                int(rng.integers(0, 3)),
            )
            with self.subTest(trial=trial):
                _assert_votes_equal(alignment_ops.asterism_anchor_votes_cpu_compiled(*args),
                                    alignment_ops.asterism_anchor_votes_numpy(*args))

    def test_no_token_pairs_give_no_star_pairs(self) -> None:
        self._compiled_module()
        empty = np.empty(0, dtype=np.int64)
        anchors = np.zeros(4, dtype=np.int32)
        for backend in (alignment_ops.asterism_anchor_votes_numpy,
                        alignment_ops.asterism_anchor_votes_cpu_compiled):
            with self.subTest(backend=backend.__name__):
                pair_idx, voted_pairs, votes = backend(empty, empty, anchors, anchors, 3, 3, 5, 1)
                self.assertEqual(pair_idx.shape, (0, 2))
                self.assertEqual((voted_pairs, len(votes)), (0, 0))

    def test_rejects_out_of_range_indices(self) -> None:
        module = self._compiled_module()
        anchors = np.array([0, 1, 2], dtype=np.int32)
        pairs = np.array([0, 1], dtype=np.int64)
        cases = {
            "token": (np.array([0, 3], dtype=np.int64), pairs, anchors, anchors, 3, 3),
            "anchor": (np.array([0, 2], dtype=np.int64), pairs, anchors, anchors, 2, 3),
            "shape": (pairs, pairs[:1], anchors, anchors, 3, 3),
        }
        for name, args in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    module.asterism_anchor_votes_cpu(*args, 5, 1)

    def test_forced_numpy_uses_reference(self) -> None:
        pairs = np.arange(6, dtype=np.int64)
        anchors = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        args = (pairs, pairs, anchors, anchors, 3, 3, 2, 1)
        with mock.patch.dict("os.environ", {"HNW_CUSTOM_OPS_FALLBACK": "numpy"}, clear=False):
            with mock.patch.object(
                alignment_ops,
                "asterism_anchor_votes_numpy",
                wraps=alignment_ops.asterism_anchor_votes_numpy,
            ) as reference:
                got = asterism_anchor_votes(*args)
        reference.assert_called_once()
        np.testing.assert_array_equal(got[0], [[0, 0], [1, 1], [2, 2]])
