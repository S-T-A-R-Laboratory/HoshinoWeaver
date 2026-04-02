import asyncio
import os
from typing import Any, TypeAlias, Union

import av
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .imgfio import load_img
from .queue import RichContextQueue
from .utils import COMMON_SUFFIX, NOT_RECOM_SUFFIX, is_support_format

Frame: TypeAlias = Union[NDArray[np.uint8], NDArray[np.uint16], None]


class BaseLoader(object):

    def __init__(self, src: RichContextQueue, length: int, config: dict[str,
                                                                        Any]):
        self.src = src
        self.length = length
        self.config = config

    def load(self, item: Any) -> Any:
        raise NotImplementedError("Subclass must implement this method")

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= self.length:
            raise StopAsyncIteration
        img = await asyncio.to_thread(self.load, await self.src.get())
        self._idx += 1
        return img


class ImgFileListLoader(BaseLoader):

    def load(self, item: str):
        return load_img(item)


class ArrayLoader(BaseLoader):

    def load(self, item: int):
        return self.config['configs']['data'][item]


class VideoFileLoader(BaseLoader):

    def __init__(
        self,
        src: str,
        config: dict[str, Any],
    ):
        self.container = av.open(src, options={'threads': str(os.cpu_count())})
        self.video = self.container.streams.video[0]
        self.video.thread_type = "FRAME"
        self.video_frame_cache: list[av.VideoFrame] = []
        self.start_frame: int = config.get("start_frame", 0)
        self.end_frame: int = config.get("end_frame", self.video.frames)
        self.fps = self.video.average_rate
        self.set_to(self.start_frame)
        self.length = self.end_frame - self.start_frame

    def load(self, item: int):
        # 跳转访问至指定帧
        self.set_to(self.start_frame + item)
        return self.load_frame()

    def load_frame(self):
        try:
            while True:
                if self.video_frame_cache:
                    return self.video_frame_cache.pop(0).to_ndarray(
                        format='bgr24')
                frames: list[av.VideoFrame] = self.container.demux(
                    video=0).__next__().decode()  # type: ignore
                if not frames:
                    continue
                if len(frames) > 1:
                    self.video_frame_cache.extend(frames[1:])
                return frames[0].to_ndarray(format='bgr24')
        except Exception as e:
            logger.error(f"{e.__repr__()} encountered when reading"
                         f"video frame with {self.__class__.__name__}.")
            return None

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        # 使用load_frame()代替load()
        if self._idx >= self.end_frame - self.start_frame:
            raise StopIteration
        img = self.load_frame()
        self._idx += 1
        return img

    def set_to(self, frame_num: int):
        """设置当前指针位置。
        """
        if self.video.time_base is None:
            raise av.error.ValueError(
                code=-1,
                message="Invalid time_base value: None",
            )
        # backward seeking makes sure cur frame is before the target.
        # seems seek using us instead of ms.
        self.container.seek(int(round(frame_num * 1e6 / self.fps)),
                            any_frame=False,
                            backward=True)
        # 2-stage seeking, decoding until find the frame_num.
        for packet in self.container.demux(video=0):
            for decoded_frame in packet.decode():
                cur_frame = self.pts2frame(decoded_frame.pts)
                if cur_frame >= frame_num:
                    return True
        return True

    def pts2frame(self, pts: int):
        if self.video.time_base is None:
            return -1
        return int(pts * float(self.video.time_base) * self.fps)
