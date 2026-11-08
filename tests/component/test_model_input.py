"""Pixel ranges, channel ordering and linear stretch at the model boundary."""
import numpy as np
import pytest

from hoshicore.component.data_container import FloatImage
from hoshicore.component.model_input import image_to_rgb8


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_full_range_and_input_unchanged(dtype):
    high = np.iinfo(dtype).max
    image = np.repeat(np.array([0, high // 4, high // 2, high], dtype=dtype)[None, :, None], 3, axis=2)
    before = image.copy()
    result = image_to_rgb8(image, color_order='RGB')
    np.testing.assert_allclose(result[0, :, 0], [0, 64, 128, 255], atol=1)
    np.testing.assert_array_equal(image, before)
    assert result.dtype == np.uint8


def test_float_default_range_channel_order_and_explicit_black_white_levels():
    image = np.array([[[0., .5, 1.]]], np.float32)
    np.testing.assert_array_equal(image_to_rgb8(image), [[[255, 128, 0]]])
    np.testing.assert_array_equal(image_to_rgb8(image * 65535, value_range=(0, 65535)), [[[255, 128, 0]]])
    np.testing.assert_array_equal(image_to_rgb8(np.array([[[-1., .5, 2.]]]), value_range=(0, 1)), [[[255, 128, 0]]])
    with pytest.raises(ValueError, match='value_range'):
        image_to_rgb8(image * 255)


def test_float_image_uses_range_dtype_unless_overridden():
    image = FloatImage(np.full((2, 3, 3), 32767.5, np.float32), np.dtype('uint16'))
    assert (image_to_rgb8(image) == 128).all()
    assert (image_to_rgb8(image, value_range=(0, 32767.5)) == 255).all()
    assert (image.data == 32767.5).all()


def test_linear_gamma_stretch_occurs_after_range_mapping():
    image = np.array([[[0., 25., 100.]]], np.float32)
    result = image_to_rgb8(image, value_range=(0, 100), color_order='RGB', linear=True, stretch_gamma=2)
    np.testing.assert_array_equal(result, [[[0, 128, 255]]])
    np.testing.assert_array_equal(image, [[[0, 25, 100]]])
    np.testing.assert_array_equal(image_to_rgb8(image, value_range=(0, 100), color_order='RGB'), [[[0, 64, 255]]])


def test_large_float_range_is_normalized_before_reducing_precision():
    result = image_to_rgb8(np.array([[[0., 5e99, 1e100]]]), value_range=(0, 1e100), color_order='RGB')
    np.testing.assert_array_equal(result, [[[0, 128, 255]]])


@pytest.mark.parametrize('image,options', [
    (np.zeros((1, 2, 3), np.int16), {}),
    (np.full((1, 2, 3), np.nan), {}),
    (np.full((1, 2, 3), np.inf), {}),
    (np.zeros((1, 2, 3)), {'value_range': (1, 1)}),
    (np.zeros((1, 2, 3)), {'value_range': (0, np.inf)}),
    (np.zeros((1, 2, 3)), {'value_range': (0,)}),
    (np.zeros((1, 2, 3)), {'stretch_gamma': 0}),
    (np.zeros((1, 2, 3)), {'stretch_gamma': 'auto'}),
    (np.zeros((1, 2, 3)), {'linear': 'auto'}),
])
def test_invalid_image_semantics_rejected(image, options):
    with pytest.raises(ValueError):
        image_to_rgb8(image, **options)
