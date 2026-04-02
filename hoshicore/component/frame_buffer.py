"""
磁盘帧缓冲：将帧写入临时 .npz 文件，支持按索引随机读取。
用于 SigmaClipping 等需要多 pass 遍历帧数据的场景。
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger


class DiskFrameBuffer:
    """磁盘帧缓冲：将帧（及可选权重）写入临时 .npz 文件，按索引随机读取。

    用法：
        buffer = DiskFrameBuffer()
        buffer.append(frame1, weight1)
        buffer.append(frame2)

        frame, weight = buffer[0]   # 从磁盘读取
        buffer.cleanup()             # 删除所有临时文件
    """

    def __init__(self, temp_path: Optional[Union[str, Path]] = None):
        self.temp_path = Path(temp_path) if temp_path else Path(
            tempfile.gettempdir())
        os.makedirs(self.temp_path, exist_ok=True)
        self.prefix = base64.urlsafe_b64encode(os.urandom(6)).decode("ascii")
        self._count = 0
        self._paths: list[Path] = []

    def append(self, frame: np.ndarray,
               weight: Optional[Union[float, np.ndarray]] = None) -> None:
        """保存一帧（及可选权重）到磁盘。

        Args:
            frame: 图像数据 (np.ndarray)。
            weight: 可选权重，标量或与 frame 同形状的 ndarray。
        """
        path = self.temp_path / f"{self.prefix}_{self._count:05d}.npz"
        if weight is not None:
            if isinstance(weight, (int, float)):
                weight = np.array(weight)
            np.savez(path, frame=frame, weight=weight)
        else:
            np.savez(path, frame=frame)
        self._paths.append(path)
        self._count += 1

    def __getitem__(self, idx: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """按索引从磁盘读取帧和权重。

        Args:
            idx: 帧索引。

        Returns:
            (frame, weight) 元组。weight 为 None 表示该帧无权重。
        """
        if idx < 0 or idx >= self._count:
            raise IndexError(
                f"DiskFrameBuffer index {idx} out of range [0, {self._count})"
            )
        data = np.load(self._paths[idx])
        frame = data['frame']
        weight = data['weight'] if 'weight' in data else None
        # 标量权重还原
        if weight is not None and weight.ndim == 0:
            weight = float(weight)
        return frame, weight

    def __len__(self) -> int:
        return self._count

    def cleanup(self) -> None:
        """删除所有临时缓冲文件。"""
        for p in self._paths:
            try:
                p.unlink()
            except OSError:
                pass
        self._paths.clear()
        self._count = 0
        logger.debug(
            f"DiskFrameBuffer cleaned up (prefix={self.prefix}).")
