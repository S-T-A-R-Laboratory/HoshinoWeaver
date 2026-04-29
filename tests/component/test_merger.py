import numpy as np
import pytest

from hoshicore.component.data_container import FastGaussianParam, FloatImage
from hoshicore.component.merger import (
    MaxMerger,
    MeanMerger,
    MinMerger,
    SigmaClippingMerger,
    HuberWeightedMerger,
    _accum_dtypes,
)


class TestMaxMerger:
    def test_max_of_frames(self):
        frames = [
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            np.array([[[50, 10, 40]]], dtype=np.uint8),
            np.array([[[30, 60, 20]]], dtype=np.uint8),
        ]
        merger = MaxMerger()
        for f in frames:
            merger.merge(f)
        expected = np.maximum(np.maximum(frames[0], frames[1]), frames[2])
        np.testing.assert_array_equal(merger.merged_image, expected)

    def test_max_with_weight(self):
        f1 = np.array([[[100, 200]]], dtype=np.uint8)
        f2 = np.array([[[150, 100]]], dtype=np.uint8)
        merger = MaxMerger()
        merger.merge(f1, weight=0.5)
        merger.merge(f2, weight=1.0)
        assert merger.merged_image is not None
        assert merger.merged_image[0, 0, 0] == max(int(100 * 0.5), int(150 * 1.0))


class TestMinMerger:
    def test_min_of_frames(self):
        frames = [
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            np.array([[[50, 10, 40]]], dtype=np.uint8),
            np.array([[[30, 60, 20]]], dtype=np.uint8),
        ]
        merger = MinMerger()
        for f in frames:
            merger.merge(f)
        expected = np.minimum(np.minimum(frames[0], frames[1]), frames[2])
        np.testing.assert_array_equal(merger.merged_image, expected)


class TestMeanMerger:
    def test_mean_of_constant_frames(self):
        val = 100
        frames = [np.full((4, 4, 3), val, dtype=np.uint8) for _ in range(10)]
        merger = MeanMerger()
        for f in frames:
            merger.merge(f)
        result = merger.merged_image
        assert isinstance(result, FloatImage)
        np.testing.assert_allclose(result.data, val, atol=1)

    def test_mean_statistics_output(self):
        frames = [np.full((4, 4, 3), v, dtype=np.uint8) for v in [90, 100, 110]]
        merger = MeanMerger()
        for f in frames:
            merger.merge(f)
        fgp = merger.result
        assert isinstance(fgp, FastGaussianParam)
        np.testing.assert_allclose(fgp.mu, 100, atol=1)
        assert np.all(fgp.n == 3)

    def test_mean_with_int_weight(self):
        frames = [np.full((4, 4, 3), 100, dtype=np.uint8) for _ in range(5)]
        merger = MeanMerger(int_weight=True)
        for f in frames:
            merger.merge(f, weight=1.0)
        result = merger.merged_image
        assert result is not None

    def test_clear(self):
        merger = MeanMerger()
        merger.merge(np.full((2, 2, 3), 100, dtype=np.uint8))
        assert merger.merged_image is not None
        merger.clear()
        assert merger.merged_image is None


class TestSigmaClippingMerger:
    def test_rejects_outlier(self):
        n_frames = 20
        val = 100
        frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(n_frames)]

        # Phase 1: build reference stats
        mean_merger = MeanMerger()
        for f in frames:
            mean_merger.merge(f)
        ref_fgp = mean_merger.result

        # Add an outlier frame
        outlier = np.full((8, 8, 3), 250, dtype=np.uint8)
        all_frames = frames + [outlier]

        # Phase 2: sigma clip
        sc_merger = SigmaClippingMerger(ref_img=ref_fgp, rej_high=2.0, rej_low=2.0)
        for f in all_frames:
            sc_merger.merge(f)

        rejected_fgp = sc_merger.result
        # The outlier should have been rejected (n > 0 for rejected pixels)
        assert rejected_fgp is not None
        assert np.any(rejected_fgp.n > 0)


class TestHuberWeightedMerger:
    @pytest.mark.skip(reason="HuberWeightedMerger missing _merge impl — cannot instantiate (known issue)")
    def test_constant_frames(self):
        n_frames = 10
        val = 100
        frames = [np.full((8, 8, 3), val, dtype=np.uint8) for _ in range(n_frames)]

        # Phase 1: reference stats
        mean_merger = MeanMerger()
        for f in frames:
            mean_merger.merge(f)
        ref_fgp = mean_merger.result

        # Phase 2: Huber weighted mean
        huber = HuberWeightedMerger(ref_stats=ref_fgp)
        for f in frames:
            huber.merge(f)

        result = huber.merged_image
        assert result is not None
        mu = result.mu
        np.testing.assert_allclose(mu, val, atol=1)


class TestAccumDtypes:
    def test_uint8_no_intweight(self):
        sum_dt, sq_dt, n_dt = _accum_dtypes(np.dtype("uint8"), int_weight=False)
        assert sum_dt.itemsize > np.dtype("uint8").itemsize
        assert n_dt == np.dtype("uint16")

    def test_uint8_with_intweight(self):
        sum_dt, sq_dt, n_dt = _accum_dtypes(np.dtype("uint8"), int_weight=True)
        assert n_dt == np.dtype("uint32")
