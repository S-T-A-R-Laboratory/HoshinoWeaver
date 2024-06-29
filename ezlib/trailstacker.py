import multiprocessing as mp

from typing import Optional, Union, Any

from easydict import EasyDict

import numpy as np

from .imgfio import ImgSeriesLoader, get_color_profile, load_info, load_img
from .utils import get_resize, time_cost_warpper, GaussianParam, get_mp_num, DTYPE_UPSCALE_MAP, FastGaussianParam
from loguru import logger


def generate_weight(length: int,
                    fin: float,
                    fout: float,
                    int_weight=False) -> np.ndarray:
    """为渐入渐出星轨生成每张图像分配的权重。

    Args:
        length (int): 序列长度。
        fin (float): 渐入比例(0-1)。
        fout (float): 渐出比例(0-1)。
        int_weight (bool, optional): 是否将权重转换为uint8（范围将从0-1映射到0-255）以加速运算。Defaults to False.

    Returns:
        list[float,int]: 权重序列
    """
    assert fin + fout <= 1
    in_len = int(length * fin)
    out_len = int(length * fout)
    ret_weight = np.ones((length, ), dtype=np.float16)
    if in_len > 0:
        l = np.arange(1, 100, 99 / in_len) / 100
        ret_weight[:in_len] = l
    if out_len > 0:
        r = np.arange(1, 100, 99 / out_len)[::-1] / 100
        ret_weight[-out_len:] = r
    if int_weight:
        # 启用uint8权重时，权重转换为uint8；
        # 非渐入渐出模式时，不乘以255（以进一步减少格式转换，加速计算）
        if in_len + out_len > 0:
            return np.array(ret_weight * 255, dtype=np.uint8)
        return np.array(ret_weight, dtype=np.uint8)
    return ret_weight


class BaseMergerSubprocess(object):
    """通用的叠加子进程。
    使用时需要定义图像的加载函数和融合函数。

    Args:
        object (_type_): _description_
    """

    def __init__(self, img_loader_type: type) -> None:
        self.id = None
        self.img_loader_type = img_loader_type
        self.stacked_num = 0
        self.merged_img = None

    def merge_image(self, new_img) -> None:
        raise NotImplementedError

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def run(self, **kwargs):
        self.id = kwargs.get("id")
        img_loader: ImgSeriesLoader = self.img_loader_type(**kwargs)
        tot_num = img_loader.tot_num
        self.merged_img = None
        self.stacked_num = 0
        failed_num = 0
        try:
            img_loader.start()
            for i in range(tot_num):
                raw_img = img_loader.pop()
                if raw_img is None:
                    logger.warning("Skip the failed frame.")
                    failed_num += 1
                    continue
                cur_img = self.post_process(raw_img, index=i)
                if self.merged_img is None:
                    self.merged_img = cur_img
                else:
                    self.merge_image(cur_img)
                self.stacked_num += 1
        except Exception as e:
            # TODO: 明确错误类型。目前不确定，因此捕获全部错误。这会导致进程无法终止，因此最好合理处理。
            logger.error(
                f"Fatal error:{e}. {self.__class__.__name__}#{self.id} will terminated."
                + "The final result cam be unexpected.")

        finally:
            img_loader.stop()
        if self.stacked_num == 0:
            logger.warning(f"No valid frames are loaded!")
            return None
        logger.info(
            f"{self.__class__.__name__}#{self.id} successfully stacked {self.stacked_num} "
            + f"images from {tot_num} images. ({failed_num} fail(s)).")
        return self.merged_img


class MaxMergerSubprocessor(BaseMergerSubprocess):
    """基于通用的叠加子进程模板的最大值叠加类。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, img_loader_type) -> None:
        super().__init__(img_loader_type)
        self.weight_list = []

    def merge_image(self, new_img) -> None:
        assert self.merged_img is not None, "Unexcepted None encoutered"
        self.merged_img = np.max([self.merged_img, new_img], axis=0)

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> np.ndarray:
        if index is not None:
            assert 0 <= index < len(
                self.weight_list
            ), f"Invalid index {index} encountered. Expect 0<=index<={len(self.weight_list)}."
            return img * self.weight_list[index]
        else:
            return img

    def run(self, **kwargs):
        self.weight_list = kwargs["weight_list"]
        return super().run(**kwargs)


class MeanMergerSubprocessor(BaseMergerSubprocess):
    """基于通用的叠加子进程模板的平均值叠加类。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def merge_image(self, new_img) -> None:
        assert self.merged_img is not None, "Unexcepted None encoutered"
        self.merged_img = self.merged_img + new_img

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> FastGaussianParam:
        return FastGaussianParam(img)


@time_cost_warpper
def StarTrailMaster(fname_list: list[str],
                    fin_ratio: float,
                    fout_ratio: float,
                    resize_length: Optional[int] = None,
                    int_weight: bool = True,
                    output_bits: int = -1,
                    ground_mask: Optional[np.ndarray] = None) -> EasyDict:
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

    mp_num, sub_length = get_mp_num(tot_length)
    pool = mp.Pool(processes=mp_num)
    results = mp.Queue()

    # 基于参考图像，获取EXIF信息
    base_info: Any = load_info(fname_list[0])
    # TODO: 可能出错:load_img可能会抛出错误。尚未处理
    sample_img = load_img(fname_list[0])
    dtype_opt = sample_img.dtype

    # DEBUG INFO
    logger.info(
        f"mp_num = {mp_num}; int_weight = {int_weight}; dtype = {dtype_opt}")
    logger.info(f"ICC_profile = {get_color_profile(base_info.colorprofile)}")
    logger.info(f"EXIF TAG num = {len(base_info.exif)}")
    logger.debug(f"Raw basic infomation={base_info}")

    # 对于需要快速预览的场景，resize_opt被设置为具体的值；否则为None。
    resize_opt = get_resize(resize_length,
                            sample_img.shape[:2]) if resize_length else None

    # 整形输入的位深度选项
    if (fin_ratio == 0 and fout_ratio == 0):
        # 1权重不进行类型转换，计算时按*1d运算
        int_weight = True
    elif int_weight and dtype_opt in DTYPE_UPSCALE_MAP:
        #INT权重 -> base图像扩大范围
        logger.info(
            f"Upscale datatype from {dtype_opt} To {DTYPE_UPSCALE_MAP[dtype_opt]} to apply int-weight."
        )
        dtype_opt = DTYPE_UPSCALE_MAP[dtype_opt]
    weight_list = generate_weight(tot_length,
                                  fin_ratio,
                                  fout_ratio,
                                  int_weight=int_weight)
    logger.debug(f"Full weight list:{weight_list}")

    try:
        subprocessor = MaxMergerSubprocessor(ImgSeriesLoader)
        # 多线程叠加
        for i in range(mp_num):
            l, r = int(i * sub_length), int((i + 1) * sub_length)
            #subprocessor.run(fname_list=fname_list[l:r],
            #                 dtype=dtype_opt,
            #                 resize=resize_opt,
            #                 weight_list=weight_list[l:r])
            pool.apply_async(subprocessor.run,
                             kwds=dict(id=i,
                                       fname_list=fname_list[l:r],
                                       dtype=dtype_opt,
                                       resize=resize_opt,
                                       weight_list=weight_list[l:r]),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: logger.error(error))
        # 合并多线程叠加结果
        base_img = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = np.max([base_img, cur_img], axis=0)
        # TODO: 转换到指定的输出类型。包括从8位图像叠加时创建16位图像的需求
        if (not int_weight) and (sample_img.dtype in DTYPE_UPSCALE_MAP):
            # 输入为整形，但未使用整数权重时，会被转换为float16进行计算
            # 因此需要强制转换为整型
            # TODO: [但使用的整形精度应当根据输入进一步权衡]
            base_img = np.array(base_img, dtype=sample_img.dtype)
        if int_weight and (sample_img.dtype != dtype_opt):
            # 使用int_weight的需要做范围回退
            logger.info(
                f"Downscale datatype from {base_img.dtype} To {sample_img.dtype} to apply int-weight."
            )
            base_img = np.array(base_img // 255, dtype=sample_img.dtype)

    except KeyboardInterrupt as e:
        pool.join()
        raise e
    # TODO: 异常处理
    return EasyDict(img=base_img,
                    exif=base_info.exif,
                    colorprofile=base_info.colorprofile)


@time_cost_warpper
def MeanTrackMaster(fname_list: list[str],
                    resize_length: Optional[int] = None,
                    dtype_opt=None,
                    resize_opt=None) -> EasyDict:
    """平均值叠加的入口函数。

    Args:
        fname_list (list[str]): _description_
        resize_length (Optional[int], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    tot_length = len(fname_list)
    mp_num, sub_length = get_mp_num(tot_length)
    pool = mp.Pool(processes=mp_num)
    # 基于参考图像，获取EXIF信息，並默認提升範圍
    # TODO：一些定制化的變換，比如用uint8得到uint16
    base_info: Any = load_info(fname_list[0])
    sample_img = load_img(fname_list[0])
    raw_dtype_opt = sample_img.dtype

    dtype_opt = DTYPE_UPSCALE_MAP[raw_dtype_opt]
    results = mp.Queue()
    logger.info(
        f"Processor Num = {mp_num}; Dtype = {dtype_opt}; Resize = {resize_opt}"
    )
    try:
        subprocessor = MeanMergerSubprocessor(ImgSeriesLoader)
        # 多线程叠加
        # TODO: 出问题的话，pool似乎不会直接中断，也不抛出堆栈错误
        for i in range(mp_num):
            l, r = int(i * sub_length), int((i + 1) * sub_length)
            pool.apply_async(subprocessor.run,
                             kwds=dict(id=i,
                                       fname_list=fname_list[l:r],
                                       dtype=dtype_opt,
                                       resize=resize_opt),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: print(error))
        # 合并多线程叠加结果
        base_img = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = base_img + cur_img

    except KeyboardInterrupt as e:
        pool.join()
        raise e
    return EasyDict(img=np.array(base_img.mu, dtype=raw_dtype_opt),
                    exif=base_info.exif,
                    colorprofile=base_info.colorprofile)