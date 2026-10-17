"""Ordered video sink; images travel directly to FFmpeg without frame files."""
import asyncio
from typing import Any

from ..component.queue import StreamExhausted
from ..component.video_io import FFmpegVideoWriter, VideoColorConverter, VideoExportOptions
from ..engine.registry import register_op
from .base import BaseOp

_VIDEO_DEFAULTS = VideoExportOptions.defaults()


async def _writer_call(writer, method, *args):
    # A cancelled to_thread await does not stop its thread. Reap the child and
    # join the pending operation before allowing a DAG task to finish.
    worker = asyncio.create_task(asyncio.to_thread(getattr(writer, method), *args))
    try:
        return await asyncio.shield(worker)
    except BaseException:
        await asyncio.to_thread(writer.abort)
        await asyncio.gather(worker, return_exceptions=True)
        raise


@register_op()
class VideoSaveOp(BaseOp):
    INPUTS = {
        "data": {"type": "sequence"},
        "exifs": {"type": "sequence", "required": False},
    }
    CONFIGS = {
        "output_filename": {"type": "str"},
        "ffmpeg_path": {"type": "str", "global": True},
        "codec": {"type": "str"},
        "fps": {"type": "float"},
        "width": {"type": "int"},
        "height": {"type": "int"},
        "chroma": {"type": "str"},
        "rate_control": {"type": "str"},
        "crf": {"type": "float"},
        "bitrate_kbps": {"type": "int"},
        "preset": {"type": "str"},
        "color_range": {"type": "str"},
        "input_color": {"type": "str"},
    }
    CONFIGS = {
        name: {**spec, "default": _VIDEO_DEFAULTS[name]}
        if name in _VIDEO_DEFAULTS else spec
        for name, spec in CONFIGS.items()
    }
    OUTPUTS = {"result": {"type": "object", "description": "Completed video receipt"}}

    @classmethod
    def estimate_resources(cls, configs, frame_bytes, n_frames, dtype_bytes=None):
        # Includes color conversion buffers; FFmpeg encoder memory is external.
        return 8 * frame_bytes, 0

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        options = VideoExportOptions.from_mapping(configs)
        converter = VideoColorConverter(options.input_color)
        writer = None
        count = 0
        try:
            for _ in self._input_range():
                pending = self._async_convert_inputs()
                try:
                    values = await asyncio.gather(*pending.values())
                except StreamExhausted:
                    if self.length is not None:
                        raise ValueError("Video input ended before its announced length")
                    break
                data = dict(zip(pending, values))
                frame = await asyncio.to_thread(converter.convert, data["data"], data.get("exifs"))
                if writer is None:
                    writer = FFmpegVideoWriter(options, frame.shape)
                    await _writer_call(writer, "start")
                await _writer_call(writer, "write", frame)
                count += 1
            if writer is None:
                raise ValueError("Cannot export a video from an empty image sequence")
            await _writer_call(writer, "finish")
            await self._broadcast_outputs({"result": {
                "path": str(writer.path), "frame_count": count,
                "codec": options.codec, "container": options.container,
                "fps": options.fps, "width": writer.size[0],
                "height": writer.size[1], "pixel_format": options.pixel_format,
                "color_space": "bt709", "color_range": options.color_range,
                "color_metadata_embedded": options.codec == "h264",
            }})
        finally:
            if writer is not None:
                await asyncio.to_thread(writer.abort)
