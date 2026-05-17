"""Image format presets for OutputPanel.

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
        "allowed_dtypes": ["uint8", "uint16", "uint32"],
        "params": {},
    },
}


def all_format_keys() -> list[str]:
    return list(IMAGE_FORMAT_PRESETS.keys())


def detect_format_by_path(path: str) -> str | None:
    """Return preset key matching the file extension, or None."""
    if not path:
        return None
    lower = path.lower()
    for fmt, preset in IMAGE_FORMAT_PRESETS.items():
        for ext in preset["ext"]:
            if lower.endswith(ext):
                return fmt
    return None
