import multiprocessing as mp
import threading
from tqdm import tqdm
from typing import Optional

FAIL_FLAG = 0
SUCC_FLAG = 1
END_FLAG = -1
DEFAULT_TIMEOUT = 60

class QueueProgressbar(object):
    """ Master叠加进程使用的进程基类。Queue启动独立的线程管理进度条，并通过进程共享的队列更新当前进度。
    重写其他类型进度条时，需要重写以下方法：
    1. __init__
    2. start
    3. update
    4. stop

    可以通过调用`QueueProgressbar.progress`获取当前进度。
    Args:
        object (_type_): _description_
    """

    def __init__(self, tot_num: int = 0, desc: str = "") -> None:
        self.queue = mp.Manager().Queue()
        self.thread = threading.Thread(target=self.loop, args=())
        self.progress = 0
        self.reset(tot_num, desc)
        self.stopped = True

    def reset(self, tot_num: Optional[int] = None, desc: str = "") -> None:
        if tot_num:
            self.tot_num = tot_num
        if desc:
            self.desc = desc

    def start(self, desc=None):
        self.progress = 0
        self.stopped = False
        if desc is not None:
            self.desc = desc
        self.thread.start()

    def stop(self):
        if not self.stopped:
            self.queue.put(END_FLAG)
            self.stopped = True

    def put(self, flag):
        self.queue.put(flag)

    def loop(self):
        try:
            status = None
            while (self.progress < self.tot_num and not self.stopped):
                try:
                    # 60s兜底
                    status = self.queue.get(timeout=DEFAULT_TIMEOUT)
                except KeyboardInterrupt as e:
                    pass
                if status == END_FLAG:
                    self.stopped = True
                self.progress += 1
                self.update()

        except Exception as e:
            self.stopped = True

    def update(self):
        raise NotImplementedError


class TqdmProgressbar(QueueProgressbar):
    """使用tqdm作为前端展示的进度条类。适用于命令行运行时。

    Args:
        object (_type_): _description_
    """

    def start(self, desc=None):
        self.tqdm_manager = tqdm(total=self.tot_num,
                                 unit="imgs",
                                 dynamic_ncols=True,
                                 desc=self.desc)
        super().start(desc=desc)

    def update(self):
        self.tqdm_manager.update(1)
    
    def stop(self):
        self.tqdm_manager.close()
        super().stop()