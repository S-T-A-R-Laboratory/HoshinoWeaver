"""Shared, serialized Photoshop RAW rendering."""
import os
from pathlib import Path
from typing import Any

from ..component.exif import ExifData
from ..component.photoshop import PhotoshopRawRenderer
from .registry import SerialService, ServiceKey


class PhotoshopService(SerialService):

    def __init__(self, settings: dict[str, int | bool | str | None]):
        super().__init__()
        self._settings = settings
        self._renderer = None

    def _render(self, path):
        if self._renderer is None:
            self._renderer = PhotoshopRawRenderer(**self._settings)
        return self._renderer.render(path)

    async def render(self, path) -> tuple[Any, ExifData]:
        return await self._run(self._render, path)

    def _close(self):
        if self._renderer is not None:
            self._renderer.close()


PHOTOSHOP = ServiceKey('photoshop', PhotoshopService)


def photoshop_settings(configs):
    path = configs.get('photoshop_path')
    return {
        'path':
        os.path.normcase(str(Path(path).expanduser().resolve()))
        if path else None,
        'timeout':
        configs.get('photoshop_timeout', 180),
        'hide':
        configs.get('photoshop_hide', True)
    }
