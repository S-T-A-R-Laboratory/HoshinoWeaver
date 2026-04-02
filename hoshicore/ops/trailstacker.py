import sys
from asyncio import Queue, gather
from typing import Any, Optional

from loguru import logger

from ..component.merger import BaseMerger, MaxMerger, MinMerger, MeanMerger, SigmaClippingMerger
from ..component.progressbar import (END_FLAG, FAIL_FLAG, SUCC_FLAG,
                                     QueueProgressbar, TqdmProgressbar)
from ..component.queue import RichContextQueue, CancellationError
from .base import BaseOp

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
            "required": False
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

    async def _async_execute(self, configs: dict[str, Any]) -> None:
        int_weight: bool = configs['int_weight']
        merger = self.MERGER(int_weight=int_weight)
        tot_num = self.length
        assert tot_num is not None, "TrailStackerOp requires sequence length information."

        # TODO: 其他暂不支持的特性：
        #proc_id：由多进程调度器单独提供在名称中
        #debug模式：作为固定参数
        #progressbar： 还没想好
        #on_error_action ： 固定配置，可能和debug一起由全局同一指向

        has_weight = self.inputs['weight'].active

        stacked_num = 0
        failed_num = 0
        err_msg_collector = []
        progressbar = None

        try:
            for i in range(tot_num):
                # filename 暂时不支持，应该由img_queue传入
                cur_filename = f"the {i+1}-th frame"

                try:
                    upper_stream_data = self._async_convert_inputs()
                    cur_img = await upper_stream_data['data']
                    weight = (await upper_stream_data['weight']) if has_weight else None
                except StopIteration:
                    logger.warning(f"{self.name}: upstream ended at {i}/{tot_num}")
                    break

                # Empty result handling
                if cur_img is None:
                    warning_msg = f"{self.name} failed to load {cur_filename}."
                    err_msg_collector.append(warning_msg)
                    logger.warning(warning_msg)
                    logger.warning(f"Skip {cur_filename}.")
                    failed_num += 1
                    if progressbar:
                        progressbar.put(FAIL_FLAG)
                    # When on_error_action = ON_ERR_STOP, stop iteration immediately
                    #if on_error_action == ON_ERR_STOP:
                    #    logger.warning(f"{self.name} will stop immediately.")
                    #    break
                    continue

                try:
                    merger.merge(cur_img, weight)
                except AssertionError as e:
                    err_msg_collector.append(
                        f"Shape of {cur_filename} does not match.")
                    raise e
                if progressbar:
                    progressbar.put(SUCC_FLAG)
                stacked_num += 1

            if stacked_num == 0:
                logger.warning(f"No valid frames are loaded!")
                return

            logger.info(f"{self.name} successfully stacked {stacked_num} " +
                        f"images from {tot_num} images. ({failed_num} fail(s)).")

            # 输出结果
            put_tasks = []
            for queue in self.outputs['result']:
                put_tasks.append(queue.put(merger.merged_image))
            await gather(*put_tasks)

        except Exception as e:
            logger.error(f"{self.name} failed: {e}")
            if progressbar:
                progressbar.put(END_FLAG)
            raise

class MinStackerOp(TrailStackerOp):
    MERGER = MinMerger
    
class MeanStackerOp(TrailStackerOp):
    MERGER = MeanMerger
