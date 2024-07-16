import multiprocessing as mp
import sys
from typing import Optional, Union, Any

from easydict import EasyDict

import numpy as np

from .imgfio import ImgSeriesLoader, get_color_profile, load_info, load_img
from .utils import BITS2DTYPE, DTYPE_NUM2TYPE, error_raiser, get_max_expmean, get_resize, time_cost_warpper, GaussianParam, get_mp_num, DTYPE_UPSCALE_MAP, FastGaussianParam, DTYPE_REVERSE_MAP, DTYPE_MAX_VALUE
from loguru import logger

import cv2


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


def load_sample_img(fname_list: list[str]) -> Optional[np.ndarray]:
    """load sample image from the given filename list if possible.

    Args:
        fname_list (list[str]): filename list

    Returns:
        Optional[np.ndarray]: the sample image
    """
    sample_index = 0
    sample_img = None
    while sample_index < len(fname_list):
        sample_img = load_img(fname_list[sample_index])
        if sample_img is not None:
            break
        sample_index += 1
    return sample_img


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


# TODO: 也许未来进一步抽象出叠加方式。快速开发阶段先忽略。
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
        logger.remove()
        if kwargs["debug"]:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")
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


class SigmaClipSubprocessor(BaseMergerSubprocess):
    """基于通用的叠加子进程模板的，带有N*Sigma拒绝平均值叠加类。

    该进程叠加的self.merged_img其实是被拒绝的叠加结果。取值和输出时需要转换。

    Args:
        BaseMergerSubprocess (_type_): _description_
    """

    def __init__(self, img_loader_type: type, rej_high_img: np.ndarray,
                 rej_low_img: np.ndarray) -> None:
        super().__init__(img_loader_type)
        self.rej_high_img = rej_high_img
        self.rej_low_img = rej_low_img
        self.merged_img = None

    def merge_image(self, new_img) -> None:
        assert self.merged_img is not None, "Unexcepted None encoutered."
        new_img.mask((new_img.mu > self.rej_high_img)
                     | (new_img.mu < self.rej_low_img))
        self.merged_img = self.merged_img + new_img

    def post_process(self,
                     img: np.ndarray,
                     index: Optional[int] = None) -> FastGaussianParam:
        return FastGaussianParam(img)


@time_cost_warpper
def StarTrailMaster(
        fname_list: list[str],
        fin_ratio: float,
        fout_ratio: float,
        resize: Optional[str] = None,
        int_weight: bool = True,
        output_bits: Optional[int] = None,
        ground_mask: Optional[np.ndarray] = None) -> Optional[EasyDict]:
    """星轨最大值叠加的入口函数。

    Args:
        fname_list (list[str]): 图像名列表
        fin_ratio (float): 渐入效果比值
        fout_ratio (float): 渐出效果比值
        resize (Optional[int], optional): _description_. Defaults to None.
        output_bits (int, optional): _description_. Defaults to -1.
        ground_mask (Optional[np.ndarray], optional): _description_. Defaults to None.

    Returns:
        Optional[EasyDict]: 叠加完成的图像及其exif，颜色配置等信息。如果无法得到图像，返回None。
    """
    tot_length = len(fname_list)

    mp_num, sub_length = get_mp_num(tot_length)
    pool = mp.Pool(processes=mp_num)
    results = mp.Queue()

    # 获取EXIF信息和样本图像
    base_info: Any = load_info(fname_list[0])
    sample_img = load_sample_img(fname_list)
    # 如果无法获得样本图像，说明序列完全没有可用图像。直接退出。
    if sample_img is None:
        logger.error(
            "No sample image available: All reading attempt failed. " +
            "Check your input.")
        return None

    # 对于星轨，默认运行时dtype与输入一致
    input_dtype = sample_img.dtype
    runtime_dtype = input_dtype
    output_dtype = BITS2DTYPE[output_bits] if output_bits else input_dtype

    # DEBUG INFO
    logger.info(
        f"mp_num = {mp_num}; int_weight = {int_weight}; dtype = {input_dtype}")
    logger.info(f"ICC_profile = {get_color_profile(base_info.colorprofile)}")
    logger.info(f"EXIF TAG num = {len(base_info.exif)}")
    logger.debug(f"Raw basic infomation={base_info}")

    # 对于需要快速预览的场景，resize_opt被设置为具体的值；否则为None。
    resize_opt = get_resize(resize, sample_img.shape[:2])

    # 整形输入的位深度选项
    if (fin_ratio == 0 and fout_ratio == 0):
        # 1权重不进行类型转换，计算时按*1d运算
        int_weight = True
    elif int_weight and input_dtype in DTYPE_UPSCALE_MAP:
        #INT权重 -> base图像扩大范围
        runtime_dtype = DTYPE_UPSCALE_MAP[input_dtype]
        logger.info(
            f"Upscale datatype from {input_dtype} To {runtime_dtype} to apply int-weight."
        )
    else:
        runtime_dtype = float

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
            pool.apply_async(subprocessor.run,
                             kwds=dict(id=i,
                                       fname_list=fname_list[l:r],
                                       dtype=runtime_dtype,
                                       resize=resize_opt,
                                       weight_list=weight_list[l:r],
                                       debug = False),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: logger.error(error))
        # 合并多线程叠加结果
        base_img = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = np.max([base_img, cur_img], axis=0)

        # 精度转换
        # upscale_time代表输出需要放缩255的倍数
        upscale_time = DTYPE_REVERSE_MAP[output_dtype] - DTYPE_REVERSE_MAP[
            input_dtype]
        if int_weight and input_dtype != runtime_dtype:
            # 应用int_weight时，相当于已上放缩一次，此处还原。
            upscale_time -= 1

        if DTYPE_REVERSE_MAP[output_dtype] > DTYPE_REVERSE_MAP[runtime_dtype]:
            # 如果希望输出高于运行时精度的数据，警告（但仍然按需求输出）
            logger.warning(
                f"Expect output to be {output_dtype}, but runtime data type is {runtime_dtype}."
                + f"Accuracy is duplicated.")

        if upscale_time > 0:
            base_img = np.array(base_img, dtype=output_dtype)
            base_img *= (255**upscale_time)
        elif upscale_time:
            base_img = np.array(base_img // 255**(abs(upscale_time)),
                                dtype=output_dtype)

    except KeyboardInterrupt as e:
        pool.join()
        raise e
    # TODO: 异常处理
    return EasyDict(img=base_img,
                    exif=base_info.exif,
                    colorprofile=base_info.colorprofile)


@time_cost_warpper
def MeanStackMaster(fname_list: list[str],
                    resize: Optional[str] = None,
                    output_bits: Optional[int] = None,
                    ground_mask: Optional[np.ndarray] = None) -> EasyDict:
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

    base_info: Any = load_info(fname_list[0])
    sample_img = load_img(fname_list[0])
    if sample_img is None:
        return EasyDict(img=None)

    input_dtype = sample_img.dtype
    resize_opt = get_resize(resize, sample_img.shape[:2])
    # 默認提升1次範圍。如果jpg过多，需要升2次。
    # TODO: 设计上的优化：理论上子进程不超过255时不需要提升2次，仅合并时提升也可以达到相同效果。
    if len(fname_list) > 255 and input_dtype == np.dtype("uint8"):
        runtime_dtype = DTYPE_UPSCALE_MAP[DTYPE_UPSCALE_MAP[input_dtype]]
    else:
        runtime_dtype = DTYPE_UPSCALE_MAP[input_dtype]
    output_dtype = BITS2DTYPE[output_bits] if output_bits else input_dtype

    results = mp.Queue()
    logger.info(
        f"Processor Num = {mp_num}; Dtype = {input_dtype}; Resize = {resize_opt}"
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
                                       dtype=runtime_dtype,
                                       resize=resize_opt),
                             callback=lambda ret: results.put(ret),
                             error_callback=lambda error: print(error))
        # 合并多线程叠加结果
        base_img: FastGaussianParam = results.get()
        for i in range(mp_num - 1):
            cur_img = results.get()
            base_img = base_img + cur_img

    except KeyboardInterrupt as e:
        pool.join()
        raise e

    # 精度转换
    # upscale_time代表输出需要放缩255的倍数
    upscale_time = DTYPE_REVERSE_MAP[output_dtype] - DTYPE_REVERSE_MAP[
        input_dtype]
    ret_img = base_img.mu
    var_img = base_img.var
    if upscale_time > 0:
        ret_img = np.array(base_img.mu * (255**upscale_time),
                           dtype=output_dtype)
        var_img = base_img.var * (255**upscale_time)**2
    else:
        ret_img = np.array(base_img.mu // (255**abs(upscale_time)),
                           dtype=output_dtype)
        var_img = base_img.var // (255**abs(upscale_time))**2

    return EasyDict(img=ret_img,
                    var=var_img,
                    exif=base_info.exif,
                    colorprofile=base_info.colorprofile)


@time_cost_warpper
def SigmaClippingMaster(fname_list: list[str],
                        resize: Optional[str] = None,
                        output_bits: Optional[int] = None,
                        ground_mask: Optional[np.ndarray] = None,
                        rej_high: float = 3.0,
                        rej_low: float = 3.0,
                        max_iter: int = 5) -> EasyDict:
    """带SigmaClipping的平均值叠加的入口函数。

    Args:
        fname_list (list[str]): _description_
        resize_length (Optional[int], optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    tot_length = len(fname_list)
    mp_num, sub_length = get_mp_num(tot_length)
    pool = mp.Pool(processes=mp_num)

    base_info: Any = load_info(fname_list[0])
    sample_img = load_img(fname_list[0])
    if sample_img is None:
        return EasyDict(img=None)

    input_dtype = sample_img.dtype
    resize_opt = get_resize(resize, sample_img.shape[:2])
    # 默認提升1次範圍。如果jpg过多，需要升2次。
    # TODO: 设计上的优化：理论上子进程不超过255时不需要提升2次，仅合并时提升也可以达到相同效果。
    if tot_length > 255 and input_dtype == np.dtype("uint8"):
        post_runtime_dtype = DTYPE_UPSCALE_MAP[DTYPE_UPSCALE_MAP[input_dtype]]
    else:
        post_runtime_dtype = DTYPE_UPSCALE_MAP[input_dtype]
    if sub_length > 255 and input_dtype == np.dtype("uint8"):
        runtime_dtype = DTYPE_UPSCALE_MAP[DTYPE_UPSCALE_MAP[input_dtype]]
    else:
        runtime_dtype = DTYPE_UPSCALE_MAP[input_dtype]

    output_dtype = BITS2DTYPE[output_bits] if output_bits else input_dtype

    results = mp.Queue()
    logger.info(
        f"Processor Num = {mp_num}; Dtype = {input_dtype}; Resize = {resize_opt}"
    )
    try:
        base_subprocessor = MeanMergerSubprocessor(ImgSeriesLoader)
        # 多线程叠加
        # TODO: 出问题的话，pool似乎不会直接中断，也不抛出堆栈错误
        for i in range(mp_num):
            l, r = int(i * sub_length), int((i + 1) * sub_length)
            pool.apply_async(base_subprocessor.run,
                             kwds=dict(id=i,
                                       fname_list=fname_list[l:r],
                                       dtype=runtime_dtype,
                                       resize=resize_opt),
                             callback=lambda ret: results.put(ret),
                             error_callback=error_raiser)
        # 合并多线程叠加结果
        base_img = FastGaussianParam(sum_mu=np.zeros((1, ),
                                                     dtype=post_runtime_dtype),
                                     n=np.zeros((1, ),
                                                dtype=np.dtype("uint16")))
        for i in range(mp_num):
            cur_img = results.get()
            base_img = base_img + cur_img

    except KeyboardInterrupt as e:
        pool.join()
        raise e
    # Sigma Clip
    try:
        cur_iter = 0
        ref_img = base_img
        last_clip_num = ref_img.n
        reject_img = FastGaussianParam(sum_mu=np.zeros(
            (1, ), dtype=post_runtime_dtype),
                                       n=np.zeros((1, ),
                                                  dtype=np.dtype("uint16")))
        while cur_iter < max_iter:
            ref_mu = ref_img.mu
            ref_var = np.sqrt(ref_img.var)
            rej_high_img = np.array((ref_mu + ref_var * rej_high).clip(
                min=0, max=DTYPE_MAX_VALUE[input_dtype]),
                                    dtype=input_dtype)
            rej_low_img = np.array((ref_var - ref_var * rej_low).clip(
                min=0, max=DTYPE_MAX_VALUE[input_dtype]),
                                   dtype=input_dtype)
            cv2.imwrite("ref_high_img.jpg", rej_high_img)
            cur_iter += 1
            logger.info(f"SigmaClipping iter#{cur_iter}...")
            subprocessor = SigmaClipSubprocessor(ImgSeriesLoader,
                                                 rej_high_img=rej_high_img,
                                                 rej_low_img=rej_low_img)
            # 多线程叠加
            # TODO: 出问题的话，pool似乎不会直接中断，也不抛出堆栈错误
            # TODO: 如果是加载时resize，会多次计算。这是冗余的。
            for i in range(mp_num):
                l, r = int(i * sub_length), int((i + 1) * sub_length)
                pool.apply_async(subprocessor.run,
                                 kwds=dict(id=i,
                                           fname_list=fname_list[l:r],
                                           dtype=runtime_dtype,
                                           resize=resize_opt),
                                 callback=lambda ret: results.put(ret),
                                 error_callback=lambda error: print(error))
            # 合并多线程叠加结果
            reject_img = FastGaussianParam(sum_mu=np.zeros(
                (1, ), dtype=post_runtime_dtype),
                                           n=np.zeros(
                                               (1, ),
                                               dtype=np.dtype("uint16")))
            for i in range(mp_num):
                cur_img = results.get()
                reject_img = reject_img + cur_img
            ref_img = base_img - reject_img
            if ((last_clip_num - ref_img.n).all() == 0):
                break
            last_clip_num = ref_img.n

    except KeyboardInterrupt as e:
        pool.join()
        raise e

    # 精度转换
    # upscale_time代表输出需要放缩255的倍数
    upscale_time = DTYPE_REVERSE_MAP[output_dtype] - DTYPE_REVERSE_MAP[
        input_dtype]
    ret_img = ref_img.mu
    var_img = ref_img.var
    rej_img = reject_img
    if upscale_time > 0:
        ret_img = np.array(ref_img.mu * (255**upscale_time),
                           dtype=output_dtype)
        rej_img = np.array(reject_img.mu * (255**upscale_time),
                           dtype=output_dtype)
        var_img = ref_img.var * (255**upscale_time)**2
    else:
        ret_img = np.array(ref_img.mu // (255**abs(upscale_time)),
                           dtype=output_dtype)
        rej_img = np.array(reject_img.mu // (255**abs(upscale_time)),
                           dtype=output_dtype)
        var_img = ref_img.var // (255**abs(upscale_time))**2

    return EasyDict(img=ret_img,
                    var=var_img,
                    rej=reject_img,
                    exif=base_info.exif,
                    colorprofile=base_info.colorprofile)


@time_cost_warpper
def SimpleMixTrailMaster(fname_list: list[str],
                         fin_ratio: float,
                         fout_ratio: float,
                         resize: Optional[str] = None,
                         int_weight: bool = True,
                         output_bits: int = -1,
                         ground_mask: Optional[str] = None) -> EasyDict:
    """基于蒙版简单混合的平均值+星轨最大值叠加的入口函数。

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
    max_ezdict = StarTrailMaster(fname_list, fin_ratio, fout_ratio, resize,
                                 int_weight, output_bits, ground_mask)
    mean_ezdict = SigmaClippingMaster(fname_list, resize, output_bits,
                                      ground_mask)
    max_img = max_ezdict.img
    mean_img = mean_ezdict.img
    mean_var = mean_ezdict.var
    resize_opt = get_resize(resize, max_img.shape[:2])

    assert ground_mask, "this mode should launch with a ground mask!"
    mask = load_img(ground_mask, resize=resize_opt)
    if mask is None:
        logger.error("Fail to load mask.")
        return EasyDict(img=None)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.repeat(mask[..., None], max_img.shape[-1], axis=-1)
    div_num = DTYPE_MAX_VALUE[
        mask.dtype] if mask.dtype in DTYPE_MAX_VALUE else np.max(mask)
    float_mask = np.array(mask, dtype=float) / div_num
    ratio = get_max_expmean(len(fname_list))
    diff_img = ratio * np.sqrt(mean_var)
    #if diff_img.dtype == np.dtype("uint8"):
    #    diff_img = cv2.medianBlur(diff_img, 128)
    #    diff_img = cv2.cvtColor(cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY),
    #                            cv2.COLOR_GRAY2BGR)
    reg_max_img = max_img - diff_img
    logger.info(f"fix ratio = {ratio:.4f}")
    #cv2.imwrite("var.tiff", mean_var)
    return EasyDict(img=reg_max_img * float_mask + (1 - float_mask) * mean_img,
                    exif=mean_ezdict.exif,
                    colorprofile=mean_ezdict.colorprofile)