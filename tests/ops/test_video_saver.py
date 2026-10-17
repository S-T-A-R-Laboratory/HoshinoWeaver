"""Video export behavior, including real encode/decode when FFmpeg is available."""
import asyncio
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from hoshicore.component.data_container import FloatImage
from hoshicore.component.exif import ExifData
from hoshicore.component.queue import BaseQueue, RichContextQueue
from hoshicore.component.video_io import (
    FFmpegVideoWriter, VideoColorConverter, VideoExportOptions, output_size,
    probe_ffmpeg, resolve_ffmpeg,
)
from hoshicore.engine.build import _load_yaml, validate_and_build_order
from hoshicore.engine.flatten import flatten_sub_dags
from hoshicore.engine.meta import meta_resolve
from hoshicore.engine.wiring import instantiate_and_wire, run_from_yaml
from hoshicore.ops.video_saver import VideoSaveOp


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ffmpeg():
    try:
        return resolve_ffmpeg(None)
    except ValueError:
        pytest.skip("Real FFmpeg executable is not available")


def test_video_options_preserve_missing_none_and_zero_semantics():
    options = VideoExportOptions.from_mapping({
        "output_filename": "output.mp4", "ffmpeg_path": None,
        "width": 0, "height": 0, "crf": 0,
    })
    assert options.fps == 25.0
    assert options.ffmpeg_path is None
    assert (options.width, options.height, options.crf) == (0, 0, 0)
    assert output_size((35, 71, 3), options) == (72, 36)
    resized = replace(options, width=64, chroma="422")
    assert output_size((35, 71, 3), resized) == (64, 32)
    assert options.width == 0


def test_video_options_normalize_and_do_not_retain_the_callers_mapping(tmp_path):
    mapping = {"output_filename": tmp_path / "output.mp4", "fps": "29.97",
               "crf": "19.5", "chroma": 422, "width": 64.0}
    options = VideoExportOptions.from_mapping(mapping)
    assert options.output_filename == str(tmp_path / "output.mp4")
    assert options.fps == 29.97 and options.crf == 19.5
    assert options.chroma == "422" and type(options.width) is int
    mapping["fps"] = 1
    assert options.fps == 29.97
    with pytest.raises(FrozenInstanceError):
        options.fps = 1


@pytest.mark.parametrize("override,field", [
    ({"output_filename": None}, "output_filename"),
    ({"output_filename": "output.mov"}, "output_filename"),
    ({"codec": "av1"}, "codec"),
    ({"codec": "raw_rgb"}, "does not support mp4"),
    ({"fps": None}, "fps"),
    ({"fps": 0}, "fps"),
    ({"fps": False}, "fps"),
    ({"fps": float("nan")}, "fps"),
    ({"width": None}, "width"),
    ({"width": False}, "width"),
    ({"height": 1.5}, "height"),
    ({"crf": None}, "crf"),
    ({"bitrate_kbps": 0}, "bitrate_kbps"),
    ({"bitrate_kbps": 1.5}, "bitrate_kbps"),
    ({"chroma": None}, "chroma"),
    ({"fpps": 25}, "Unknown video option"),
])
def test_invalid_video_options_fail_at_the_configuration_boundary(override, field):
    with pytest.raises(ValueError, match=field):
        VideoExportOptions.from_mapping({"output_filename": "output.mp4", **override})


def test_video_options_require_an_output_path_and_share_op_defaults():
    with pytest.raises(ValueError, match="output_filename is required"):
        VideoExportOptions.from_mapping({})
    defaults = VideoExportOptions.defaults()
    assert "output_filename" not in defaults
    assert "default" not in VideoSaveOp.CONFIGS["output_filename"]
    assert defaults == {name: entry["default"] for name, entry in VideoSaveOp.CONFIGS.items()
                        if "default" in entry}


def test_icc_conversion_preserves_source_and_applies_transfer():
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = ExifData({"example": "unchanged"}, icc)
    image = np.full((3, 5, 3), 32768, np.uint16)
    before = image.copy()
    output = VideoColorConverter().convert(image, exif)
    assert output.dtype == np.uint8 and output.flags.c_contiguous
    # Encoded sRGB middle gray is approximately 115 in BT.709, not 128.
    assert 113 <= int(output[0, 0, 0]) <= 117
    np.testing.assert_array_equal(image, before)
    assert exif.colorprofile == icc and exif.exif == {"example": "unchanged"}


def test_float_image_retains_its_declared_integer_range():
    wrapped = FloatImage(np.full((2, 2, 3), 32767.5, np.float32), np.dtype("uint16"))
    output = VideoColorConverter("linear_srgb").convert(wrapped)
    assert np.all((output >= 179) & (output <= 181))


def test_explicit_decoded_color_overrides_source_icc():
    # Only an explicit declaration may override source ICC metadata.
    exif = ExifData({}, b"not the decoded pixel profile")
    output = VideoColorConverter("rawpy_prophoto").convert(
        np.full((2, 2, 3), 32768, np.uint16), exif)
    assert np.max(output) - np.min(output) <= 1
    assert np.all((output >= 127) & (output <= 129))


def test_auto_without_icc_uses_srgb_assumption():
    frame = np.full((2, 2, 3), 32768, np.uint16)
    output = VideoColorConverter().convert(frame, ExifData({}, b""))
    expected = VideoColorConverter("srgb").convert(frame)
    np.testing.assert_array_equal(output, expected)


def test_libraw_prophoto_red_converts_back_to_rec709_red():
    # Rounded BGR8 ProPhoto/709 encoding of linear sRGB red, using LibRaw's
    # published output primaries. Exercises matrix direction and channel order.
    frame = np.array([[[19, 74, 184]]], np.uint8)
    output = VideoColorConverter("rawpy_prophoto").convert(frame)[0, 0]
    assert output[0] <= 5 and output[1] <= 5 and output[2] >= 252


@pytest.mark.parametrize("chroma,expected", [("420", "yuv420p"), ("422", "yuv422p"), ("444", "yuv444p")])
@pytest.mark.parametrize("color_range,range_code", [("limited", 1), ("full", 2)])
@pytest.mark.parametrize("rate_control", ["crf", "bitrate"])
@pytest.mark.parametrize("container_format", ["mp4", "avi"])
def test_real_encoding_dimensions_frames_and_color_tags(ffmpeg, tmp_path, chroma, expected,
                                                       color_range, range_code, rate_control,
                                                       container_format):
    probe_ffmpeg(ffmpeg, chroma, container_format)
    output = tmp_path / f"视频.{container_format}"
    options = VideoExportOptions.from_mapping({
        "output_filename": str(output), "ffmpeg_path": ffmpeg,
        "chroma": chroma, "color_range": color_range, "width": 64,
        "fps": 12.5, "rate_control": rate_control, "crf": 0, "bitrate_kbps": 1000,
    })
    writer = FFmpegVideoWriter(options, (35, 71, 3))
    try:
        writer.start()
        for gray in (0, 128, 255):
            writer.write(np.full((35, 71, 3), gray, np.uint8))
        writer.finish()
    finally:
        writer.abort()
    with av.open(str(output)) as container:
        assert container_format in container.format.name.split(",")
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == writer.size
        assert stream.pix_fmt.replace("yuvj", "yuv") == expected
        assert float(stream.average_rate) == 12.5
        context = stream.codec_context
        assert (context.color_primaries, context.color_trc, context.colorspace) == (1, 1, 1)
        assert context.color_range == range_code
        frames = list(container.decode(video=0))
        assert len(frames) == 3
        luma = [int(np.median(np.frombuffer(frame.planes[0], dtype=np.uint8)
                             .reshape(stream.height, frame.planes[0].line_size)[:, :stream.width]))
                for frame in frames]
        black, white = (16, 235) if color_range == "limited" else (0, 255)
        assert abs(luma[0] - black) <= 3 and abs(luma[-1] - white) <= 3
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("target_width", [0, 64])
def test_uncompressed_avi_keeps_rgb_samples_and_omits_color_tags(ffmpeg, tmp_path, target_width):
    output = tmp_path / "无压缩.avi"
    options = VideoExportOptions(output_filename=str(output), ffmpeg_path=ffmpeg,
                                 codec="raw_rgb", width=target_width, fps=12.5)
    assert options.pixel_format == "bgr24" and options.color_range == "full"
    probe_ffmpeg(ffmpeg, options.chroma, options.container, options.codec)
    frames = np.random.default_rng(7).integers(0, 256, (3, 35, 71, 3), dtype=np.uint8)
    writer = FFmpegVideoWriter(options, frames[0].shape)
    try:
        writer.start()
        for frame in frames:
            writer.write(frame)
        writer.finish()
    finally:
        writer.abort()
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "rawvideo"
        assert stream.pix_fmt == "bgr24"
        assert (stream.width, stream.height) == ((64, 32) if target_width else (71, 35))
        assert float(stream.average_rate) == 12.5
        context = stream.codec_context
        assert (context.color_primaries, context.color_trc, context.colorspace) == (2, 2, 2)
        assert context.color_range == 0  # deliberately untagged RGB
        decoded = list(container.decode(video=0))
        assert len(decoded) == len(frames)
        if not target_width:
            for actual, expected in zip(decoded, frames):
                np.testing.assert_array_equal(actual.to_ndarray(format="bgr24"), expected)
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("workflow", ["timelapse_startrail", "norma_bundle_window"])
@pytest.mark.parametrize("sequence,video", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("codec", ["h264", "raw_rgb"])
def test_workflow_export_selection_compiles(workflow, sequence, video, codec):
    from hoshicore.engine.preflight import _resolve_node_configs
    from hoshicore.engine.wiring import _resolve_configs

    configs = {"enable_sequence_export": sequence, "enable_video_export": video,
               "video_codec": codec, "output_dir": "frames",
               "video_output_filename": "out.avi" if codec == "raw_rgb" else "out.mp4"}
    spec = _load_yaml(str(ROOT / "hoshicore/dag" / f"{workflow}.meta.yaml"))
    spec = meta_resolve(spec, {}, configs)
    dag = validate_and_build_order(flatten_sub_dags(spec))
    assert ("batch_saver" in dag.exec_order) is sequence
    assert ("video_saver" in dag.exec_order) is video
    assert ("result" in dag.output_links) is sequence
    assert ("video" in dag.output_links) is video
    # Also exercise static stream-source checks, including NORMA's filtered
    # image and EXIF fan-out. No processing or FFmpeg process is started here.
    _, feeders, _, _ = instantiate_and_wire(
        dag, {"fnames": ["unused.tif"]}, configs)
    for feeder in feeders:
        feeder.close()
    if video:
        effective = _resolve_configs(dag.global_configs, configs)
        options = VideoExportOptions.from_mapping(
            _resolve_node_configs(dag.nodes["video_saver"], effective, VideoSaveOp))
        assert options.codec == codec
        assert options.pixel_format == ("bgr24" if codec == "raw_rgb" else "yuv420p")


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("container_format,codec", [
    ("mp4", "h264"), ("avi", "h264"), ("avi", "raw_rgb"),
])
async def test_timelapse_video_with_optional_image_export(ffmpeg, tmp_path, monkeypatch, sequence,
                                                         container_format, codec):
    import hoshicore.engine.wiring as wiring
    monkeypatch.setattr(wiring, "_load_default_settings", lambda: {})
    files = []
    for index in range(4):
        path = tmp_path / f"input{index}.tif"
        tifffile.imwrite(path, np.full((24, 32, 3), 20 + index * 20, np.uint8))
        files.append(str(path))
    result = await asyncio.wait_for(run_from_yaml(
        str(ROOT / "hoshicore/dag/timelapse_startrail.meta.yaml"),
        {"fnames": files}, {
            "enable_sequence_export": sequence, "enable_video_export": True,
            "video_output_filename": str(tmp_path / f"output.{container_format}"),
            "video_codec": codec,
            "ffmpeg_path": ffmpeg, "video_input_color": "srgb",
            "output_dir": str(tmp_path / "frames") if sequence else None,
            "mode.sliding_max.window_size": 2,
        }, progress=False), timeout=30)
    assert result["video"]["frame_count"] == 4
    assert result["video"]["codec"] == codec
    assert result["video"]["color_metadata_embedded"] is (codec == "h264")
    assert ("result" in result) is sequence
    assert (tmp_path / "frames").exists() is sequence
    with av.open(result["video"]["path"]) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 4
        if codec == "raw_rgb":
            assert result["video"]["color_range"] == "full"
            assert result["video"]["pixel_format"] == "bgr24"
            expected = VideoColorConverter("srgb").convert(np.full((24, 32, 3), 20, np.uint8))
            np.testing.assert_array_equal(frames[0].to_ndarray(format="bgr24"), expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("container_format,codec", [
    ("mp4", "h264"), ("avi", "h264"), ("avi", "raw_rgb"),
])
async def test_unknown_length_and_cancellation_preserve_existing_output(ffmpeg, tmp_path,
                                                                       container_format, codec):
    output = tmp_path / f"existing.{container_format}"
    output.write_bytes(b"previous completed output")
    op = VideoSaveOp("video")
    op.inputs["exifs"].active = False
    op.length = None
    options = {key: entry.get("default") for key, entry in op.CONFIGS.items()}
    options.update(output_filename=str(output), ffmpeg_path=ffmpeg, input_color="srgb", codec=codec)
    await op.inputs["data"].put(np.zeros((32, 32, 3), np.uint8))
    task = asyncio.create_task(op._async_execute(options))
    for _ in range(200):
        if list(tmp_path.glob(f".existing-*.{container_format}")):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Video writer did not start")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
    assert output.read_bytes() == b"previous completed output"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.asyncio
async def test_empty_video_fails_without_creating_files(tmp_path):
    op = VideoSaveOp("video")
    op.inputs["exifs"].active = False
    op.length = 0
    options = {key: entry.get("default") for key, entry in op.CONFIGS.items()}
    options["output_filename"] = str(tmp_path / "empty.mp4")
    with pytest.raises(ValueError, match="empty"):
        await op._async_execute(options)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_sentinel_driven_video_receipt(ffmpeg, tmp_path):
    op = VideoSaveOp("filtered_video")
    op.inputs["exifs"].active = False
    op.length = None
    receipt = RichContextQueue(maxsize=1)
    op.outputs["result"].append(receipt)
    options = {"output_filename": str(tmp_path / "filtered.mp4"),
               "ffmpeg_path": ffmpeg, "input_color": "srgb"}

    async def feed():
        for value in (32, 128, 255):
            await op.inputs["data"].put(np.full((32, 32, 3), value, np.uint8))
        await op.inputs["data"].put(BaseQueue._SENTINEL)

    await asyncio.wait_for(asyncio.gather(op._async_execute(options), feed()), 15)
    result = await receipt.get()
    assert result["frame_count"] == 3
    assert result["fps"] == 25.0 and result["pixel_format"] == "yuv420p"


def test_failed_child_preserves_previous_video_and_cleans_partial(tmp_path):
    output = tmp_path / "previous.mp4"
    output.write_bytes(b"keep this video")
    # Python rejects FFmpeg flags and exits: exercise a real failed child,
    # rather than replacing the production process/pipe implementation.
    writer = FFmpegVideoWriter(VideoExportOptions(
        output_filename=str(output), ffmpeg_path=sys.executable), (512, 512, 3))
    try:
        writer.start()
        with pytest.raises(RuntimeError, match="FFmpeg"):
            writer.write(np.zeros((512, 512, 3), np.uint8))
            writer.finish()
    finally:
        writer.abort()
    assert writer._process.poll() is not None
    assert output.read_bytes() == b"keep this video"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.asyncio
async def test_cancelling_blocked_pipe_reaps_child_and_writer_thread(tmp_path, monkeypatch):
    import hoshicore.component.video_io as video_io

    popen = subprocess.Popen
    processes = []

    def nonreading_child(command, **kwargs):
        child = popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(video_io, "resolve_ffmpeg", lambda path: sys.executable)
    monkeypatch.setattr(video_io.subprocess, "Popen", nonreading_child)
    op = VideoSaveOp("blocked_video")
    op.inputs["exifs"].active = False
    op.length = 1
    options = {key: entry.get("default") for key, entry in op.CONFIGS.items()}
    options.update(output_filename=str(tmp_path / "cancelled.mp4"), input_color="srgb")
    await op.inputs["data"].put(np.zeros((1024, 1024, 3), np.uint8))
    task = asyncio.create_task(op._async_execute(options))
    for _ in range(200):
        if processes:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Child did not start")
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    assert processes[0].poll() is not None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("configs,error", [
    ({"enable_sequence_export": False, "enable_video_export": False}, "At least one"),
    ({"enable_sequence_export": False, "enable_video_export": True,
      "video_output_filename": "out.mp4", "ffmpeg_path": "missing-video-encoder.exe"}, "FFmpeg"),
    ({"enable_sequence_export": False, "enable_video_export": True,
      "video_output_filename": "out.mp4", "video_codec": "raw_rgb"}, "does not support mp4"),
])
async def test_invalid_export_setup_stops_before_processing(monkeypatch, configs, error):
    import hoshicore.engine.wiring as wiring
    monkeypatch.setattr(wiring, "_load_default_settings", lambda: {})

    async def reject_run(*args, **kwargs):
        raise AssertionError("Invalid export must not start image processing")

    monkeypatch.setattr(wiring, "run_dag", reject_run)
    with pytest.raises(ValueError, match=error):
        await wiring.run_from_yaml(
            str(ROOT / "hoshicore/dag/timelapse_startrail.meta.yaml"),
            {"fnames": ["not-read.tif"]}, configs, progress=False)
