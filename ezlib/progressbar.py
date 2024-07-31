import multiprocessing as mp
import threading
from tqdm import tqdm
from typing import Optional

FAIL_FLAG = 0
SUCC_FLAG = 1
END_FLAG = -1

class QueueProgressbar(object):
    def __init__(self, tot_num: int, desc: str = "") -> None:
        self.tot_num = tot_num
        self.desc = desc
        self.stopped = True
        self.queue = mp.Manager().Queue()
        self.thread = threading.Thread(target=self.loop, args=())
        self.progress = 0

    def reset(self, tot_num: Optional[int]=None, desc: str = "")->None:
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
                # TODO: timeout硬编码为60？
                try:
                    status = self.queue.get(timeout=60)
                except KeyboardInterrupt as e:
                    pass
                if status == END_FLAG:
                    # TODO: fix: 如果某一进程出现意外造成终止，会阻塞全局进度条。
                    self.stopped = True
                self.progress +=1
                self.update()
                
        except Exception as e:
            self.stopped = True
    
    def update(self):
        raise NotImplementedError

class TqdmProgressbar(object):
    """A simple threading progressbar

    Args:
        object (_type_): _description_
    """

    def __init__(self, tot_num: int, desc: str = "") -> None:
        self.tot_num = tot_num
        self.desc = desc
        self.stopped = True
        self.queue = mp.Manager().Queue()
        self.thread = threading.Thread(target=self.loop, args=())

    def start(self, desc=None):
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
            for _ in tqdm(range(self.tot_num),
                          total=self.tot_num,
                          unit="imgs",
                          dynamic_ncols=True,
                          desc=self.desc):
                if self.stopped: break
                # TODO: timeout硬编码为60？
                try:
                    status = self.queue.get(timeout=60)
                except KeyboardInterrupt as e:
                    pass
                if status == END_FLAG:
                    # TODO: fix: 如果某一进程出现意外造成终止，会阻塞全局进度条。
                    self.stopped = True
        except Exception as e:
            self.stopped = True