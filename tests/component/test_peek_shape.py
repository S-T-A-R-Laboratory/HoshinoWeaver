import numpy as np
import pytest
import tifffile

from hoshicore.component.image_io import load_tiff_preview, peek_shape


class TestPeekShapeTiff:
    def test_uint8_rgb(self, tmp_path):
        path = str(tmp_path / "test.tif")
        data = np.zeros((100, 200, 3), dtype=np.uint8)
        tifffile.imwrite(path, data)
        shape, dtype_bytes = peek_shape(path)
        assert shape == (100, 200, 3)
        assert dtype_bytes == 1

    def test_uint16_rgb(self, tmp_path):
        path = str(tmp_path / "test.tif")
        data = np.zeros((50, 80, 3), dtype=np.uint16)
        tifffile.imwrite(path, data)
        shape, dtype_bytes = peek_shape(path)
        assert shape == (50, 80, 3)
        assert dtype_bytes == 2

    def test_grayscale(self, tmp_path):
        path = str(tmp_path / "gray.tif")
        data = np.zeros((64, 64), dtype=np.float32)
        tifffile.imwrite(path, data)
        shape, dtype_bytes = peek_shape(path)
        assert shape == (64, 64)
        assert dtype_bytes == 4


class TestLoadTiffPreview:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_unsigned_rgb_uses_dtype_full_range(self, tmp_path, dtype):
        path = tmp_path / f"{np.dtype(dtype).name}_rgb.tif"
        max_value = np.iinfo(dtype).max
        data = np.array(
            [0, max_value // 2, max_value], dtype=dtype).reshape(1, 1, 3)
        tifffile.imwrite(path, data, photometric="rgb")

        preview = load_tiff_preview(str(path))

        assert preview.shape == data.shape
        assert preview.dtype == np.uint8
        assert preview.flags.c_contiguous
        np.testing.assert_array_equal(
            preview, np.array([0, 127, 255], dtype=np.uint8).reshape(1, 1, 3))

    def test_float32_grayscale_handles_nonfinite_values(self, tmp_path):
        path = tmp_path / "float32_gray.tif"
        data = np.linspace(-0.5, 1.5, 16, dtype=np.float32).reshape(4, 4)
        data[0, 0] = np.nan
        data[0, 1] = np.inf
        tifffile.imwrite(path, data)

        preview = load_tiff_preview(str(path))

        assert preview.shape == (4, 4, 3)
        assert preview.dtype == np.uint8
        np.testing.assert_array_equal(preview[:, :, 0], preview[:, :, 1])
        np.testing.assert_array_equal(preview[:, :, 1], preview[:, :, 2])


class TestPeekShapeCommon:
    def test_png_rgb(self, tmp_path):
        import PIL.Image
        path = str(tmp_path / "test.png")
        img = PIL.Image.new("RGB", (320, 240))
        img.save(path)
        shape, dtype_bytes = peek_shape(path)
        assert shape == (240, 320, 3)
        assert dtype_bytes == 1

    def test_jpg(self, tmp_path):
        import PIL.Image
        path = str(tmp_path / "test.jpg")
        img = PIL.Image.new("RGB", (640, 480))
        img.save(path)
        shape, dtype_bytes = peek_shape(path)
        assert shape == (480, 640, 3)
        assert dtype_bytes == 1


class TestPeekShapeErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            peek_shape("/nonexistent/path.tif")

    def test_unsupported_format(self, tmp_path):
        path = str(tmp_path / "test.xyz")
        path_obj = tmp_path / "test.xyz"
        path_obj.write_text("dummy")
        with pytest.raises(ValueError):
            peek_shape(path)
