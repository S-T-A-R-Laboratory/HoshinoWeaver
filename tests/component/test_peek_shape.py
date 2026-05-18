import numpy as np
import pytest
import tifffile

from hoshicore.component.image_io import peek_shape


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
