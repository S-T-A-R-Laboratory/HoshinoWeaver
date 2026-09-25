import asyncio

import numpy as np
import pytest
import tifffile

import hoshicore.component.image_io as image_io
import hoshicore.ops.image_saver as image_saver
from hoshicore.ops.image_saver import (
    BatchImageSaveOp, ImageSaveOp, _format_output_path)


async def _ready(value):
    return value


@pytest.mark.parametrize("dtype", [np.uint32, np.float32])
def test_save_32bit_tiff_uses_tifffile(monkeypatch, tmp_path, dtype):
    def reject_opencv(*args, **kwargs):
        raise AssertionError("32-bit TIFF must not use OpenCV")

    monkeypatch.setattr(image_io.cv2, "imencode", reject_opencv)
    path = tmp_path / f"image_{np.dtype(dtype).name}.tiff"
    values = (np.linspace(-0.5, 1.5, 12, dtype=dtype)
              if dtype is np.float32 else np.arange(12, dtype=dtype))
    bgr = values.reshape(2, 2, 3)
    image_io.save_img(str(path), bgr)

    stored_rgb = tifffile.imread(path)
    assert stored_rgb.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(stored_rgb, bgr[:, :, ::-1])
    np.testing.assert_array_equal(image_io.load_img(str(path)), bgr)


@pytest.mark.parametrize("dtype_name", ["uint32", "float32"])
def test_image_save_op_supports_32bit_tiff(tmp_path, dtype_name):
    path = tmp_path / f"op_{dtype_name}.tif"
    op = ImageSaveOp("save")
    source = np.array([0, 32768, 65535], dtype=np.uint16).reshape(1, 1, 3)
    asyncio.run(op._async_execute({
        "image": source,
        "output_filename": str(path),
        "output_dtype": dtype_name,
        "exif": None,
        "jpg_quality": 85,
        "png_compressing": 7,
    }))
    saved = tifffile.imread(path)
    assert saved.dtype == np.dtype(dtype_name)
    if dtype_name == "uint32":
        expected = source.astype(np.uint32) * np.uint32(65537)
    else:
        expected = source.astype(np.float32)
    np.testing.assert_array_equal(saved, expected[:, :, ::-1])


def test_format_output_path_supports_sequence_and_frame_indices():
    assert _format_output_path(
        "frame_{index:04d}_src_{frame_index:04d}.png",
        3,
        20,
        frame_index=11,
    ) == "frame_0003_src_0011.png"


def test_format_output_path_requires_wired_frame_index():
    with pytest.raises(ValueError, match="frame_indices is not wired"):
        _format_output_path("frame_{frame_index:04d}.png", 3, 20)


def test_batch_saver_passes_frame_index_and_exif(monkeypatch, tmp_path):
    calls = []

    def fake_save(path, frame, **kwargs):
        calls.append((path, frame, kwargs))

    monkeypatch.setattr(image_saver, "save_img", fake_save)
    op = BatchImageSaveOp("save")
    op.inputs["frame_indices"].active = True
    op.inputs["exifs"].active = True
    op._frame_counter = 0
    frame = np.zeros((2, 3), dtype=np.uint8)
    exif = object()

    result = asyncio.run(op._async_execute_single(
        {
            "data": _ready(frame),
            "frame_indices": _ready(7),
            "exifs": _ready(exif),
        },
        {
            "output_dir": str(tmp_path),
            "output_template": "frame_{index:03d}_{frame_index:03d}.png",
            "output_dtype": None,
            "png_compressing": 4,
            "jpg_quality": 91,
        },
    ))

    assert result["result"].endswith("frame_000_007.png")
    assert calls[0][1] is frame
    assert calls[0][2] == {
        "png_compressing": 4,
        "jpg_quality": 91,
        "exif": exif,
    }


def test_batch_saver_propagates_save_failure(monkeypatch, tmp_path):
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(image_saver, "save_img", fail_save)
    op = BatchImageSaveOp("save")
    op.inputs["frame_indices"].active = False
    op.inputs["exifs"].active = False
    op._frame_counter = 0

    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(op._async_execute_single(
            {"data": _ready(np.zeros((2, 3), dtype=np.uint8))},
            {
                "output_dir": str(tmp_path),
                "output_template": "frame_{index:03d}.png",
                "output_dtype": None,
            },
        ))
