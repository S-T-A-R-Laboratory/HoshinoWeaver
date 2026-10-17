"""Image and video format presets for OutputPanel.

Encodes physical constraints of each image format:
- Allowed dtypes (e.g. JPG only supports uint8)
- Format-specific extra parameters (jpg_quality, png_compressing)
- Default file extension

The intersection of these physical constraints with workflow constraints
declared in ui.yaml's ``outputs`` block determines the final selectable options.
"""
from __future__ import annotations

from typing import Any


IMAGE_FORMAT_PRESETS: dict[str, dict[str, Any]] = {
    "JPG": {
        "ext": [".jpg", ".jpeg"],
        "default_ext": ".jpg",
        "allowed_dtypes": ["uint8"],
        "params": {
            "jpg_quality": {
                "label": "图像质量",
                "widget": "slider",
                "min": 1,
                "max": 100,
                "default": 85,
            },
        },
    },
    "PNG": {
        "ext": [".png"],
        "default_ext": ".png",
        "allowed_dtypes": ["uint8", "uint16"],
        "params": {
            "png_compressing": {
                "label": "压缩级别",
                "widget": "slider",
                "min": 1,
                "max": 9,
                "default": 7,
            },
        },
    },
    "TIFF": {
        "ext": [".tif", ".tiff"],
        "default_ext": ".tif",
        "allowed_dtypes": ["uint8", "uint16", "uint32","float32"],
        "params": {},
    },
}


_H264_PARAMS = frozenset({
    "video_codec",
    "video_fps", "video_width", "video_height", "video_chroma",
    "video_rate_control", "video_crf", "video_bitrate_kbps", "video_preset",
    "video_color_range", "video_input_color",
})
VIDEO_CODEC_PRESETS: dict[str, dict[str, Any]] = {
    "h264": {
        "label": "H.264",
        "description": "   H.264 · 8 位 · Rec.709 SDR",
        "params": _H264_PARAMS,
    },
    "raw_rgb": {
        "label": "无压缩 RGB",
        "description": "   无压缩 RGB8 · Full · Rec.709",
        "params": frozenset({"video_codec", "video_fps", "video_width",
                             "video_height", "video_input_color"}),
    },
}
VIDEO_FORMAT_PRESETS: dict[str, dict[str, Any]] = {
    "MP4": {
        "ext": [".mp4"],
        "default_ext": ".mp4",
        "description": "MP4",
        "codecs": ["h264"],
        "default_codec": "h264",
        "params": _H264_PARAMS,
    },
    "AVI": {
        "ext": [".avi"],
        "default_ext": ".avi",
        "description": "AVI",
        "codecs": ["h264", "raw_rgb"],
        "default_codec": "h264",
        "params": _H264_PARAMS,
    },
}


def all_format_keys() -> list[str]:
    return list(IMAGE_FORMAT_PRESETS.keys())


def detect_format_by_path(path: str, presets: dict | None = None) -> str | None:
    """Return preset key matching the file extension, or None."""
    if not path:
        return None
    lower = path.lower()
    for fmt, preset in (IMAGE_FORMAT_PRESETS if presets is None else presets).items():
        for ext in preset["ext"]:
            if lower.endswith(ext):
                return fmt
    return None
