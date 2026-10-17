"""Bounded FFmpeg video output and color conversion for the image pipeline.

ICC conversion uses Pillow/LittleCMS at 8 bits. Explicit input-color overrides
describe decoded pixels independently of source-file metadata. The explicit
rawpy_prophoto mode uses the matrix from LibRaw src/tables/colorconst.cpp.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageCms, features
from loguru import logger

from .data_container import FloatImage

PIXEL_FORMATS = {"420": "yuv420p", "422": "yuv422p", "444": "yuv444p"}
VIDEO_CONTAINERS = {".mp4": "mp4", ".avi": "avi"}
VIDEO_CODECS = {
    "h264": {"encoder": "libx264", "containers": ("mp4", "avi")},
    "raw_rgb": {"encoder": "rawvideo", "containers": ("avi",)},
}
INPUT_COLORS = ("auto", "srgb", "linear_srgb", "bt709", "rawpy_prophoto")
# LibRaw's linear sRGB -> ProPhoto matrix, including its white-point handling.
# https://github.com/LibRaw/LibRaw/blob/master/src/tables/colorconst.cpp
_PROPHOTO_TO_SRGB = np.linalg.inv(
    np.array([
        [0.529317, 0.330092, 0.140588],
        [0.098368, 0.873465, 0.028169],
        [0.016879, 0.117663, 0.865457],
    ],
             dtype=np.float64)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class VideoExportOptions:
    """Normalized video settings; missing fields use these component defaults.

    Explicit None is meaningful only for ffmpeg_path. Zero width/height means
    automatic sizing. Runtime state (resolved paths, dimensions, subprocesses)
    belongs to the writer, outside this immutable configuration.
    """

    output_filename: str
    ffmpeg_path: str | None = None
    codec: Literal["h264", "raw_rgb"] = "h264"
    fps: float = 25.0
    width: int = 0
    height: int = 0
    chroma: Literal["420", "422", "444"] = "420"
    rate_control: Literal["crf", "bitrate"] = "crf"
    crf: float = 18.0
    bitrate_kbps: int = 12000
    preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast",
                    "medium", "slow", "slower", "veryslow"] = "medium"
    color_range: Literal["limited", "full"] = "limited"
    input_color: Literal["auto", "srgb", "linear_srgb", "bt709",
                         "rawpy_prophoto"] = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.output_filename, (str, os.PathLike)) or Path(
                self.output_filename).suffix.lower() not in VIDEO_CONTAINERS:
            raise ValueError("Video output_filename must be an MP4 or AVI file path")
        object.__setattr__(self, "output_filename", os.fspath(self.output_filename))
        if self.codec not in VIDEO_CODECS:
            raise ValueError(f"Unsupported video codec: {self.codec}")
        if self.container not in VIDEO_CODECS[self.codec]["containers"]:
            raise ValueError(f"Video codec {self.codec} does not support {self.container} output")
        if self.ffmpeg_path is not None:
            if not isinstance(self.ffmpeg_path, (str, os.PathLike)):
                raise ValueError("Video ffmpeg_path must be a path or None")
            object.__setattr__(self, "ffmpeg_path", os.fspath(self.ffmpeg_path))
        for name in ("fps", "crf"):
            value = getattr(self, name)
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Video {name} must be a finite number") from exc
            if isinstance(value, bool) or not math.isfinite(number):
                raise ValueError(f"Video {name} must be a finite number")
            object.__setattr__(self, name, number)
        for name in ("width", "height", "bitrate_kbps"):
            value = getattr(self, name)
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Video {name} must be an integer") from exc
            if isinstance(value, bool) or number != value:
                raise ValueError(f"Video {name} must be an integer")
            object.__setattr__(self, name, number)
        if not 0 < self.fps <= 240:
            raise ValueError("Video fps must be between 0 and 240")
        if not 0 <= self.crf <= 51:
            raise ValueError("Video crf must be between 0 and 51")
        for name in ("width", "height"):
            if not 0 <= getattr(self, name) <= 32768:
                raise ValueError(f"Video {name} must be between 0 and 32768")
        if self.bitrate_kbps <= 0:
            raise ValueError("Video bitrate_kbps must be positive")
        object.__setattr__(self, "chroma", str(self.chroma))
        if self.chroma not in PIXEL_FORMATS:
            raise ValueError("Video chroma must be 420, 422 or 444")
        if self.rate_control not in ("crf", "bitrate"):
            raise ValueError("Video rate_control must be crf or bitrate")
        if self.preset not in ("ultrafast", "superfast", "veryfast", "faster",
                               "fast", "medium", "slow", "slower", "veryslow"):
            raise ValueError("Unsupported H.264 encoding preset")
        if self.color_range not in ("limited", "full"):
            raise ValueError("Video color_range must be limited or full")
        if self.input_color not in INPUT_COLORS:
            raise ValueError(f"Video input_color must be one of {INPUT_COLORS}")
        if self.codec == "raw_rgb":
            # RGB samples always use the full 8-bit range. Chroma and rate
            # control settings are retained but do not apply to this codec.
            object.__setattr__(self, "color_range", "full")

    @classmethod
    def from_mapping(cls, configs: Mapping[str, Any]) -> VideoExportOptions:
        """Adapt an Op's resolved mapping, preserving absent/None/zero semantics."""
        if "output_filename" not in configs:
            raise ValueError("Video output_filename is required")
        unknown = set(configs) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown video option(s): {sorted(unknown)}")
        return cls(**configs)

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Component defaults used by the Op's parameter schema as well."""
        return {field.name: field.default for field in fields(cls)
                if field.default is not MISSING}

    @property
    def container(self) -> str:
        return VIDEO_CONTAINERS[Path(self.output_filename).suffix.lower()]

    @property
    def pixel_format(self) -> str:
        return "bgr24" if self.codec == "raw_rgb" else PIXEL_FORMATS[self.chroma]

    @property
    def encoder(self) -> str:
        return VIDEO_CODECS[self.codec]["encoder"]


def resolve_ffmpeg(path: str | None) -> str:
    if path:
        executable = shutil.which(path)
    else:
        name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        roots = [
            Path(sys.executable).parent,
            Path(__file__).resolve().parents[2]
        ]
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            roots.insert(0, Path(sys._MEIPASS))
        executable = next(
            (str(root / "external" / name)
             for root in roots if (root / "external" / name).is_file()), None)
        executable = executable or shutil.which("ffmpeg")
    if executable is None:
        raise ValueError("FFmpeg executable not found; configure ffmpeg_path")
    return executable


def probe_ffmpeg(path: str | None, chroma: str = "420", container: str = "mp4",
                 codec: str = "h264") -> str:
    """Check the executable, encoder and muxer before processing frames."""
    if container not in VIDEO_CONTAINERS.values():
        raise ValueError(f"Unsupported video container: {container}")
    if codec not in VIDEO_CODECS or container not in VIDEO_CODECS[codec]["containers"]:
        raise ValueError(f"Unsupported video codec/container combination: {codec}/{container}")
    encoder = VIDEO_CODECS[codec]["encoder"]
    executable = resolve_ffmpeg(path)
    result = subprocess.run(
        [executable, "-hide_banner", "-h", f"encoder={encoder}"],
        capture_output=True,
        timeout=15,
        creationflags=_creationflags())
    message = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    pixel_format = "bgr24" if codec == "raw_rgb" else PIXEL_FORMATS[str(chroma)]
    formats = next((line.split(":", 1)[1].split()
                    for line in message.splitlines()
                    if "Supported pixel formats:" in line), [])
    # rawvideo accepts raw pixel layouts without listing Supported formats.
    if (result.returncode or f"Encoder {encoder} [" not in message or
            (codec != "raw_rgb" and pixel_format not in formats)):
        raise ValueError(
            f"FFmpeg {encoder} does not support {pixel_format}: {message[-1500:]}"
        )
    result = subprocess.run(
        [executable, "-hide_banner", "-h", f"muxer={container}"],
        capture_output=True,
        timeout=15,
        creationflags=_creationflags())
    message = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode or f"Muxer {container} " not in message:
        raise ValueError(f"FFmpeg does not support {container} output: {message[-1500:]}")
    return executable


def _creationflags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def output_size(shape: tuple, options: VideoExportOptions) -> tuple[int, int]:
    height, width = shape[:2]
    target_w, target_h = options.width, options.height
    factors = [target_w / width] if target_w else []
    if target_h:
        factors.append(target_h / height)
    scale = min(factors) if factors else 1.0
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    chroma = options.chroma if options.codec == "h264" else None
    if chroma in ("420", "422"):
        width += width % 2
    if chroma == "420":
        height += height % 2
    return width, height


def _linear_to_709(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    return np.where(values < 0.018, 4.5 * values, 1.099 * values**0.45 - 0.099)


def _709_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values < 0.081, values / 4.5,
                    ((values + 0.099) / 1.099)**(1 / 0.45))


def _srgb_to_709(values: np.ndarray) -> np.ndarray:
    linear = np.where(values <= 0.04045, values / 12.92,
                      ((values + 0.055) / 1.055)**2.4)
    return _linear_to_709(linear)


_SRGB_TO_709_LUT = np.rint(
    _srgb_to_709(np.arange(256, dtype=np.float32) / 255) * 255).astype(
        np.uint8)


@lru_cache(maxsize=8)
def _icc_transform(profile: bytes):
    if not features.check("littlecms2"):
        raise RuntimeError(
            "Pillow LittleCMS support is required for ICC video conversion")
    return ImageCms.buildTransformFromOpenProfiles(
        ImageCms.ImageCmsProfile(BytesIO(profile)),
        ImageCms.createProfile("sRGB"),
        "RGB",
        "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)


class VideoColorConverter:
    """Produce owned, contiguous BT.709-encoded BGR8 without mutating a branch."""

    def __init__(self, input_color: str = "auto"):
        self.input_color = input_color
        self._warned_missing_profile = False

    def convert(self, frame, exif=None) -> np.ndarray:
        array = frame.data if isinstance(frame,
                                         FloatImage) else np.asarray(frame)
        dtype = np.dtype(frame.dtype)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected grayscale or BGR image, got {array.shape}")
        array = array[:, :, :3]
        if dtype.kind == "u":
            maximum = float(np.iinfo(dtype).max)
        elif dtype.kind == "f":
            maximum = 1.0  # bare float arrays use normalized [0, 1]
        else:
            raise ValueError(f"Unsupported video image dtype: {dtype}")
        normalized = np.nan_to_num(array.astype(np.float32) / maximum,
                                   nan=0.0,
                                   posinf=1.0,
                                   neginf=0.0)
        normalized = np.clip(normalized, 0, 1)
        color = self.input_color
        profile = getattr(exif, "colorprofile", None)
        if color == "auto":
            if profile:
                rgb = Image.fromarray(
                    np.rint(normalized[:, :, ::-1] * 255).astype(np.uint8))
                rgb = ImageCms.applyTransform(rgb,
                                              _icc_transform(bytes(profile)))
                return np.ascontiguousarray(
                    _SRGB_TO_709_LUT[np.asarray(rgb)][:, :, ::-1])
            else:
                color = "srgb"
                if not self._warned_missing_profile:
                    logger.warning(
                        "Video input has no ICC profile; assuming sRGB. "
                        "Set video_input_color to override.")
                    self._warned_missing_profile = True
        if color == "rawpy_prophoto":
            # Explicitly selected ProPhoto primaries and BT.709-like transfer.
            rgb = _709_to_linear(normalized[:, :, ::-1])
            rgb = rgb @ _PROPHOTO_TO_SRGB.T
            converted = _linear_to_709(rgb)[:, :, ::-1]
        elif color == "linear_srgb":
            converted = _linear_to_709(normalized)
        elif color == "srgb":
            converted = _srgb_to_709(normalized)
        elif color == "bt709":
            converted = normalized
        else:
            raise ValueError(f"Unknown video input color: {color}")
        return np.ascontiguousarray(
            np.rint(np.clip(converted, 0, 1) * 255).astype(np.uint8))


class FFmpegVideoWriter:
    """One ordered pipe, bounded diagnostics, and atomic video publication.

    Blocking methods run in a worker thread. abort() terminates the child first
    so cancellation also releases a writer blocked on a full stdin pipe.
    """

    def __init__(self, options: VideoExportOptions, shape: tuple):
        self.options = options
        self.path = Path(options.output_filename).absolute()
        self.input_shape = shape
        self.size = output_size(shape, options)
        self._errors = deque(maxlen=16)
        self._process = None
        self._reader = None
        self._temporary = None
        self._lock = threading.Lock()
        self._aborted = False

    def start(self) -> None:
        with self._lock:
            if self._aborted:
                raise RuntimeError("Video export was cancelled")
            self._start()

    def _start(self) -> None:
        options = self.options
        executable = resolve_ffmpeg(options.ffmpeg_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.path.stem}-",
                                    suffix=f".{options.container}",
                                    dir=self.path.parent)
        os.close(fd)
        self._temporary = Path(name)
        height, width = self.input_shape[:2]
        out_w, out_h = self.size
        chroma = options.chroma
        pixel_format = options.pixel_format
        color_range = options.color_range
        if options.codec == "raw_rgb":
            # Color conversion has already produced full-range Rec.709 RGB8.
            # Keep RGB channels intact and deliberately omit color metadata.
            filters = f"scale={out_w}:{out_h}:flags=lanczos,format=bgr24,setsar=1"
        else:
            filters = (f"scale={out_w}:{out_h}:flags=lanczos:in_range=full:"
                       f"out_range={color_range}:out_color_matrix=bt709,"
                       f"format={pixel_format},setsar=1")
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y", "-f",
            "rawvideo", "-pixel_format", "bgr24", "-video_size",
            f"{width}x{height}", "-framerate",
            str(options.fps), "-i", "pipe:0", "-an", "-vf",
            filters, "-c:v", options.encoder, "-pix_fmt", pixel_format,
        ]
        if options.codec == "h264":
            profile = {"420": "high", "422": "high422", "444": "high444"}[chroma]
            if options.rate_control == "crf" and options.crf == 0:
                profile = "high444"  # x264 lossless requires the predictive profile
            command += [
                "-preset", options.preset, "-profile:v", profile,
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range",
                "tv" if color_range == "limited" else "pc",
            ]
            if options.rate_control == "crf":
                command += ["-crf", str(options.crf)]
            else:
                command += ["-b:v", f"{options.bitrate_kbps}k"]
        if options.container == "mp4":
            command += ["-movflags", "+faststart"]
        command += ["-f", options.container, str(self._temporary)]
        self._process = subprocess.Popen(command,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE,
                                         bufsize=0,
                                         creationflags=_creationflags())
        self._reader = threading.Thread(target=self._read_errors, daemon=True)
        self._reader.start()

    def _read_errors(self) -> None:
        while chunk := self._process.stderr.read(4096):
            self._errors.append(chunk)

    def _failure(self) -> RuntimeError:
        details = b"".join(self._errors).decode("utf-8", errors="replace")
        return RuntimeError(
            f"FFmpeg video encoding failed: {details or self._process.returncode}"
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.input_shape or frame.dtype != np.uint8 or not frame.flags.c_contiguous:
            raise ValueError(
                "Video frames must have constant shape and contiguous BGR8 pixels"
            )
        view = memoryview(frame).cast("B")
        try:
            while view:
                written = self._process.stdin.write(view)
                if not written:
                    raise BrokenPipeError("FFmpeg stdin closed")
                view = view[written:]
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._failure() from exc

    def finish(self) -> None:
        self._process.stdin.close()
        code = self._process.wait(timeout=120)
        self._reader.join(timeout=2)
        self._process.stderr.close()
        if code:
            raise self._failure()
        if not self._temporary.stat().st_size:
            raise RuntimeError("FFmpeg produced an empty video")
        with self._lock:
            if self._aborted:
                raise RuntimeError("Video export was cancelled")
            os.replace(self._temporary, self.path)
            self._temporary = None

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._abort()

    def _abort(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            self._process.stdin.close()
            if self._reader:
                self._reader.join(timeout=2)
            self._process.stderr.close()
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
            self._temporary = None
