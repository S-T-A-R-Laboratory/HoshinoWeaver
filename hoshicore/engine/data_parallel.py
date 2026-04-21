"""兼容性 re-export — 真实实现已拆分到 segment_detect / segment_worker / segment_adapter。"""
from .segment_detect import *      # noqa: F401,F403
from .segment_worker import *      # noqa: F401,F403
from .segment_adapter import *     # noqa: F401,F403
