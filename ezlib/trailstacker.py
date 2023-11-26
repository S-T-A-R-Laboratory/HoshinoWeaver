import logging
import multiprocessing as mp
import time

from math import sqrt, floor
from typing import Optional, Union

import numpy as np

from .imgfio import ImgSeriesLoader, load_exif
from .utils import get_resize,time_cost_warpper,GaussianParam
from .logging import get_default_logger

DTYPE_UPSCALE_MAP={
    np.dtype('uint8'):np.dtype('uint16'),
    np.dtype('uint16'):np.dtype('uint32'),
    np.dtype('uint32'):np.dtype('uint64')
}

logger = get_default_logger()

def generate_weight(length: int, fin: float, fout: float, int_weight=False) -> list[float,int]:
    """为渐入渐出星轨生成每张图像分配的权重。

    Args:
        length (int): 序列长度。
        fin (float): 渐入比例(0-1)。
        fout (float): 渐出比例(0-1)。
        int_weight (bool, optional): 是否转换为uint8（范围将从0-1映射到0-255）以加速运算. Defaults to False.

    Returns:
        list[float,int]: 权重序列
    """
    assert fin + fout <= 1
    in_len = int(length * fin)
    out_len = int(length * fout)
    ret_weight = np.ones((length, ), dtype=np.float16)
    if in_len > 0:
        l = np.arange(1, 100, 100 / in_len) / 100
        ret_weight[:in_len] = l
    if out_len > 0:
        r = np.arange(1, 100, 100 / out_len)[::-1] / 100
        ret_weight[-out_len:] = r
    if int_weight:
        # 启用uint8权重时，权重转换为uint8；
        # 非渐入渐出模式时，不乘以255（以进一步减少格式转换，加速计算）
        if in_len+out_len>0:
            return np.array(ret_weight*255, dtype=np.uint8)
        return np.array(ret_weight, dtype=np.uint8)
    return ret_weight

@time_cost_warpper
def StarTrailMaster(fname_list: list[str],
                    fin_ratio: float,
                    fout_ratio: float,
                    resize_length: Optional[int] = None,
                    int_weight: bool = True,
                    output_bits: int = -1,
                    ground_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """星轨最大值叠加的入口函数。

    Args:
        fname_list (list[str]): 图像名列表
        fin_ratio (float): 渐入效果比值
        fout_ratio (float): 渐出效果比值
        resize_length (Optional[int], optional): _description_. Defaults to None.
        output_bits (int, optional): _description_. Defaults to -1.
        ground_mask (Optional[np.ndarray], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: 叠加完成的图像
    """
    tot_length = len(fname_list)
    # 调整处理器使用数目，尽可能使每个处理器叠加sqrt(N)张图像
    # 推导：n 图像分 m 组叠加，时间开销近似为 [n/m]+m ；min([n/m]+m)-> m取得sqrt(N)
    mp_num = min(floor(sqrt(tot_length)), mp.cpu_count())
    sub_length = tot_length / mp_num
    pool = mp.Pool(processes=mp_num)
    results = mp.Queue()
    
    
    # 基于参考图像，获取EXIF信息
    base_exif=load_exif(fname_list[0])
    dtype_opt=base_exif.dtype
    
    # 对于需要快速预览的场景，resize_opt被设置为具体的值；否则为None。
    resize_opt = get_resize(resize_length,base_exif.size) if resize_length else None
    
    # 整形输入的位深度选项
    if (fin_ratio==0 and fout_ratio==0): 
        # 1权重不进行类型转换，计算时按*1d运算
        int_weight=True
    elif int_weight and dtype_opt in DTYPE_UPSCALE_MAP:
        #INT权重 -> base图像扩大范围
        print(f"Upscale Datatype From {dtype_opt} To {DTYPE_UPSCALE_MAP[dtype_opt]}.")
        dtype_opt = DTYPE_UPSCALE_MAP[dtype_opt]
    weight_list = generate_weight(tot_length, fin_ratio, fout_ratio,int_weight=int_weight)

    # DEBUG INFO
    logger.debug(f"mp_num = {mp_num}; int_weight={int_weight} ;dtype={dtype_opt}")
    logger.debug(weight_list)
    try:    
        # 多线程叠加
        for i in range(mp_num):
            l, r = int(i * sub_length), int((i + 1) * sub_length)
            pool.apply_async(StarTrailStacker,
                             args=(fname_list[l:r], weight_list[l:r]),kwds=dict(
                                 dtype=dtype_opt,resize=resize_opt),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: print(error))
        # 合并多线程叠加结果
        base_img = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = np.max([base_img, cur_img], axis=0)
        # TODO: 转换到输出类型
        if (not int_weight) and (base_exif.dtype in DTYPE_UPSCALE_MAP):
            # 输入为整形，但未使用整数权重时，会被转换为float16进行计算
            # 因此需要强制转换为整型
            # TODO: [但使用的整形精度应当根据输入进一步权衡]
            base_img = np.array(base_img, dtype=base_exif.dtype)
        if int_weight and (base_exif.dtype!=dtype_opt):
            # 使用int_weight的需要做范围回退
            base_img = np.array(base_img//255,dtype=base_exif.dtype)
        
    except KeyboardInterrupt as e:
        pool.join()
    # TODO: EXIF最终写入合成图像中
    return base_img


def StarTrailStacker(fname_list: list[str],
                     weight_list: Union[list[float],list[int]],
                     dtype: Union[type,np.dtype],
                     resize=None,) -> np.ndarray:
    """最大值叠加子进程。

    Args:
        fname_list (list[str]): _description_
        weight_list (list[float]): _description_

    Raises:
        e: _description_

    Returns:
        np.ndarray: _description_
    """
    img_loader = ImgSeriesLoader(fname_list,dtype,resize,max_poolsize=8)
    base_img = None
    try:
        img_loader.start()
        base_img = img_loader.pop() * weight_list[0]
        for weight in weight_list[1:]:
            cur_img = img_loader.pop() * weight
            base_img = np.max([base_img, cur_img], axis=0)
    except Exception as e:
        # TODO: 明确错误类型。目前不确定，因此捕获全部错误
        raise e
    finally:
        img_loader.stop()
    return base_img

def MedianTrackMaster(fname_list: list[str],resize_length: Optional[int] = None, dtype_opt=None, resize_opt=None) -> np.ndarray:
    """平均值叠加的入口函数。

    Args:
        fname_list (list[str]): _description_
        resize_length (Optional[int], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    tot_length = len(fname_list)
    # 调整处理器使用数目，尽可能使每个处理器叠加sqrt(N)张图像
    # 推导：n 图像分 m 组叠加，时间开销近似为 [n/m]+m ；min([n/m]+m)-> m取得sqrt(N)
    mp_num = min(floor(sqrt(tot_length)), mp.cpu_count())
    sub_length = tot_length / mp_num
    pool = mp.Pool(processes= mp_num)
    results = mp.Queue()
    try:    
        # 多线程叠加
        for i in range(mp_num):
            l, r = int(i * sub_length), int((i + 1) * sub_length)
            pool.apply_async(StarTrailStacker,
                             args=(fname_list[l:r]),kwds=dict(
                                 dtype=dtype_opt,resize=resize_opt),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: print(error))
        # 合并多线程叠加结果
        base_img = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = np.max([base_img, cur_img], axis=0)
        
    except KeyboardInterrupt as e:
        pool.join()
    # TODO: EXIF最终写入合成图像中
    return base_img

def MedianTrackStacker(fname_list: list[str],
                     dtype: Union[type,np.dtype],
                     resize=None,) -> GaussianParam:
    """平均值叠加子进程。

    Args:
        fname_list (list[str]): _description_
        weight_list (list[float]): _description_

    Raises:
        e: _description_

    Returns:
        GaussianParam: _description_
    """
    img_loader = ImgSeriesLoader(fname_list,dtype,resize,max_poolsize=8)
    base_img = None
    try:
        img_loader.start()
        base_img = GaussianParam(mu=img_loader.pop())
        for _ in range(len(fname_list)-1):
            cur_img = GaussianParam(mu=img_loader.pop())
            base_img = base_img+cur_img
    except Exception as e:
        # TODO: 明确错误类型。目前不确定，因此捕获全部错误
        raise e
    finally:
        img_loader.stop()
    return base_img
