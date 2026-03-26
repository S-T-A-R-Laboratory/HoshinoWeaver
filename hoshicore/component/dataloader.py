import os
from typing import Any, Optional, Union, TypeAlias, Sequence
from numpy.typing import NDArray
import av
import cv2
import numpy as np
import rawpy
import asyncio
from loguru import logger

from .utils import COMMON_SUFFIX, NOT_RECOM_SUFFIX, is_support_format

Frame: TypeAlias = Union[NDArray[np.uint8], NDArray[np.uint16], None]


class BaseLoader(object):

    def __init__(self, src: Any, config: dict[str, Any]):
        self.length = len(src)

    def load(self, index: int) -> Any:
        raise NotImplementedError("Subclass must implement this method")

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self.length:
            raise StopIteration
        img = self.load(self._idx)
        self._idx += 1
        return img


class ImgFileListLoader(BaseLoader):

    def __init__(self, src: list[str], config: dict[str, Any]):
        super().__init__(src, config)
        self.img_list = src

    def load(self, index: int):
        return load_img(self.img_list[index])


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

    def load(self, index: int):
        # 跳转访问至指定帧
        self.set_to(self.start_frame + index)
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


class ArrayLoader(BaseLoader):

    def __init__(self, src: np.ndarray, config: dict[str, Any]):
        super().__init__(src, config)
        self.img_array = src

    def load(self, index: int):
        return self.img_array[index]


class AsyncDataLoader(object):
    """
    通用异步数据加载器，用于异步预取数据，提高数据加载效率。
    
    Args:
        loader (BaseLoader): 实际数据加载源。
        max_poolsize (int): 缓冲区最大存储样本数。
    
    用法：
        dataloader = AsyncDataLoader(loader, max_poolsize=4)
        dataloader.start()
        for data in dataloader:
            ...
        dataloader.stop()
    """
    _SENTINEL = object()

    def __init__(self, loader: BaseLoader, max_poolsize: int = 1):
        self.loader = loader
        self.max_poolsize = max_poolsize
        self.data_queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=max_poolsize)
        self._length = loader.length
        self._worker_task = None

    async def _worker(self):
        for i in range(self._length):
            try:
                item = self.loader.__next__()
            except Exception as e:
                logger.error(
                    f"Error loading item {i} in {self.__class__.__name__}: {e.__repr__()}"
                )
                continue
            await self.data_queue.put(item)
        await self.data_queue.put(self._SENTINEL)

    async def execute(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
        for _ in range(self._length):
            item = await self.data_queue.get()
            if item is self._SENTINEL:
                break
            yield item

    def __len__(self):
        return self._length


class DataLoaderGenerator(object):

    def __init__(self,
                 loader_type: str,
                 src: Union[list[str], np.ndarray, str],
                 config: dict[str, int],
                 max_poolsize: int = 1):
        self.loader_type = loader_type
        self.src = src
        self.config = config
        self.max_poolsize = max_poolsize
        self.loader_class = self.build_loader_class()

    def build_loader_class(self):
        if self.loader_type == "img_file_list":
            return ImgFileListLoader
        elif self.loader_type == "video_file":
            return VideoFileLoader
        elif self.loader_type == "img_array":
            return ArrayLoader
        else:
            raise ValueError(f"Unsupported loader type: {self.loader_type}")

    def build_subloader_list(self,
                             num_subloader: int) -> list[AsyncDataLoader]:
        if self.loader_type in ["img_file_list", "img_array"]:
            subloader_list: list[AsyncDataLoader] = []
            assert isinstance(
                self.src,
                (list,
                 np.ndarray)), "src must be a list of strings or a numpy array"
            total_imgs = len(self.src)
            num_subloader = min(num_subloader, total_imgs)
            imgs_per_subloader = total_imgs / num_subloader
            for idx in range(num_subloader):
                start_idx = int(round(idx * imgs_per_subloader))
                end_idx = int(round((idx + 1) * imgs_per_subloader))
                subloader_list.append(
                    AsyncDataLoader(self.loader_class(
                        self.src[start_idx:end_idx], self.config),
                                    max_poolsize=self.max_poolsize))
            return subloader_list
        elif self.loader_type == "video_file":
            subloader_list = []
            assert isinstance(self.src, str), "src must be a string"
            start_frame = self.config.get("start_frame", 0)
            end_frame = self.config.get("end_frame", None)
            if end_frame is None:
                raise ValueError("end_frame must be provided")
            total_frames = end_frame - start_frame
            num_subloader = min(num_subloader, total_frames)
            frames_per_subloader = total_frames / num_subloader
            for idx in range(num_subloader):
                start_offset = int(round(idx * frames_per_subloader))
                end_offset = int(round((idx + 1) * frames_per_subloader))
                subloader_list.append(
                    AsyncDataLoader(self.loader_class(
                        self.src, {
                            "start_frame": start_frame + start_offset,
                            "end_frame": start_frame + end_offset
                        }),
                                    max_poolsize=self.max_poolsize))
            return subloader_list
        else:
            raise ValueError(f"Unsupported loader type: {self.loader_type}")


def load_img(file_path: str) -> Optional[np.ndarray]:
    """ Using OpenCV API to load a single image from the given path.
    
    If necessary, the image will be converted to the given dtype.

    Args:
        file_path (str): /path/to/the/image.suffix

    Returns:
        np.ndarray: normally a `numpy.ndarray` object will be returned. 
        But the image fails to be loaded, an error will be logged, and `None` will be returned under such condition.
    """
    try:
        # suffix check and warning raising
        suffix = file_path.split(".")[-1].lower()
        assert is_support_format(
            file_path), f"Unsupported img suffix:{suffix}."
        if suffix in NOT_RECOM_SUFFIX:
            logger.warning("Got an Image with not recommended suffix. \
                We do not guarantee the stability of EXIF extraction and the output image quality."
                           )
        if (suffix in COMMON_SUFFIX) or (suffix in NOT_RECOM_SUFFIX):
            # TODO: not sure if uint32/float is available.
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint16),
                               cv2.IMREAD_UNCHANGED)
            if img is None:
                # some images can not be decoded using option dtype=np.uint16.
                # this is a temp fix.
                logger.info(
                    "Uint16 decoding failed. Fallback to uint8 loading...")
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                                   cv2.IMREAD_UNCHANGED)
        else:
            # load images with rawpy
            with rawpy.imread(file_path) as raw:
                img = raw.postprocess(
                    output_bps=16,
                    output_color=rawpy.rawpy.ColorSpace(4))  # type: ignore
            # switch RGB to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Failed to read {file_path} Because {e}!")
        return None
