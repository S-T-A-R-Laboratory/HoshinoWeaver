import sys
from typing import Optional, Any

from loguru import logger

from ..component.merger import MaxMerger, BaseMerger
from ..component.progressbar import (END_FLAG, FAIL_FLAG, SUCC_FLAG,
                                     QueueProgressbar, TqdmProgressbar)
from ..component.imgfio import ImgSeriesLoader
from ..component.queue import RichContextQueue, FileCacheQueue
from .base import BaseOp
from asyncio import Queue

ON_ERR_CONTINUE = "continue"
ON_ERR_STOP = "break"


class TrailStackerOp(BaseOp):
    """
    叠加星轨
    """
    EXECUTOR = "cpu"
    INPUTS: dict[str, dict[str, Any]] = {
        "data": {
            "type": "sequence",
            "required": True
        },
        "weight": {
            "type": "sequence",
            "required": True
        },
    }
    CONFIGS: dict[str, dict[str, Any]] = {
        "int_weight": {
            "type": "bool",
            "default": False
        }
    }
    OUTPUTS = {
        "result": {
            "type": "image"
        },
    }
    MERGER = MaxMerger
    MAX_SIZE: int = 1
    _SENTINEL = object()

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        int_weight: bool = configs['int_weight']
        img_queue: RichContextQueue = self.inputs['data']
        weight_queue: RichContextQueue = self.inputs['weight']
        merger = self.MERGER(proc_id=proc_id, weight_list=weight_list)
        # TODO: 可以await 直到tot_num被改写，一来是获取准确数量，二来是等待img_loader准备好。
        # 但这会不兼容文件中转。
        tot_num = img_queue.tot_num

        # TODO: 其他暂不支持的特性：
        #proc_id：由多进程调度器单独提供在名称中
        #debug模式：作为固定参数
        #progressbar： 还没想好
        #on_error_action ： 固定配置，可能和debug一起由全局同一指向
        # reset logger level
        #logger.remove()
        #if debug:
        #    logger.add(sys.stdout, level="DEBUG")
        #    logger.info(f"Debug mode activated.")
        #else:
        #    logger.add(sys.stdout, level="INFO")

        # init img_loader and merger

        # weight用法的变更，Merger不再在初始化时接收weight_list，而是在merge时接收当前帧的weight

        stacked_num = 0
        failed_num = 0
        err_msg_collector = []
        progressbar = None

        # main progress
        try:
            for i in range(tot_num):
                # filename 暂时不支持，应该由img_queue传入
                cur_filename = f"the {i+1}-th frame"
                #if fname_list is not None:
                #    cur_filename = fname_list[i]

                raw_img = await img_queue.get()
                # Empty result handling
                if raw_img is None:
                    # add err msg
                    warning_msg = f"{self.name} failed to load {cur_filename}."
                    err_msg_collector.append(warning_msg)
                    logger.warning(warning_msg)
                    # TODO: 添加支持,对于可能预期外的叠加中间（读入失败，尺寸不匹配等）抛出额外错误
                    logger.warning(f"Skip {cur_filename}.")
                    failed_num += 1
                    if progressbar:
                        progressbar.put(FAIL_FLAG)
                    # When on_error_action = ON_ERR_STOP, stop iteration immediately
                    #if on_error_action == ON_ERR_STOP:
                    #    logger.warning(f"{self.name} will stop immediately.")
                    #    break
                    continue
                cur_img = merger.post_process(raw_img, index=i)
                # TODO: this looks ugly. Optimize this in the future.
                try:
                    merger.merge(cur_img)
                except AssertionError as e:
                    err_msg_collector.append(
                        f"Shape of {cur_filename} does not match.")
                    raise e
                if progressbar:
                    progressbar.put(SUCC_FLAG)
                stacked_num += 1
        except (KeyboardInterrupt, Exception) as e:
            logger.error(
                f"Fatal error:{e.__repr__()}. {self.name} will be terminated. "
                + "The final result cam be unexpected.")
            if progressbar:
                progressbar.put(END_FLAG)

        if stacked_num == 0:
            logger.warning(f"No valid frames are loaded!")
            return None
        logger.info(f"{self.name} successfully stacked {stacked_num} " +
                    f"images from {tot_num} images. ({failed_num} fail(s)).")
        return dict(result=merger.merged_image, err_msg=err_msg_collector)


class MeanStackerOp(GroupedBaseOp):
    pass
