import multiprocessing as mp
import sys
from abc import ABCMeta, abstractclassmethod
from typing import Any, Optional, Union
import fractions

import cv2
import numpy as np
from easydict import EasyDict
from loguru import logger

from .imgfio import ImgSeriesLoader, get_color_profile, load_img, load_info
from .merger import (BaseMerger, MaxMerger, MeanMerger, MinMerger,
                     SigmaClippingMerger)
from .progressbar import (QueueProgressbar, TqdmProgressbar, SUCC_FLAG,
                          FAIL_FLAG, END_FLAG)
from .utils import (BITS2DTYPE, DTYPE_MAX_VALUE, DTYPE_REVERSE_MAP,
                    DTYPE_UPSCALE_MAP, SOFTWARE_NAME, FastGaussianParam,
                    dtype_scaler, error_raiser, get_max_expmean, get_mp_num,
                    get_resize, time_cost_warpper)


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


def run_merger_subprocess(proc_id: int,
                          img_loader_type: type[ImgSeriesLoader],
                          merger_type: type[BaseMerger],
                          progressbar=None,
                          debug=False,
                          on_error: str = "continue",
                          **kwargs):
    """通用的叠加子进程。
    使用时需要定义图像的加载函数和融合函数。

    Args:
        object (_type_): _description_
    """
    # reset logger level
    logger.remove()
    if debug:
        logger.add(sys.stdout, level="DEBUG")
        logger.info(f"Debug mode activated.")
    else:
        logger.add(sys.stdout, level="INFO")

    # init img_loader and merger
    stacked_num = 0
    failed_num = 0
    img_loader = img_loader_type(**kwargs)
    merger = merger_type(**kwargs)
    tot_num = img_loader.tot_num

    # main progress
    try:
        img_loader.start()
        for i in range(tot_num):
            raw_img = img_loader.pop()
            if raw_img is None:
                # TODO: 添加支持,对于可能预期外的叠加中间（读入失败，尺寸不匹配等）抛出额外错误
                logger.warning("Skip the failed frame.")
                failed_num += 1
                if progressbar:
                    progressbar.put(FAIL_FLAG)
                continue
            cur_img = merger.post_process(raw_img, index=i)
            merger.merge(cur_img)
            if progressbar:
                progressbar.put(SUCC_FLAG)
            stacked_num += 1
    except (KeyboardInterrupt, Exception) as e:
        logger.error(
            f"Fatal error:{e.__repr__()}. {merger_type.__name__}Subprocessor#{proc_id} will terminated."
            + "The final result cam be unexpected.")
        if progressbar:
            progressbar.put(END_FLAG)

    finally:
        img_loader.stop()
    if stacked_num == 0:
        logger.warning(f"No valid frames are loaded!")
        return None
    logger.info(
        f"{merger_type.__name__}Subprocessor#{proc_id} successfully stacked {stacked_num} "
        + f"images from {tot_num} images. ({failed_num} fail(s)).")
    return merger.merged_image


class DtypeRecorder(object):

    def __init__(self, input_dtype, rt_upscale_num,
                 output_dtype: Optional[np.dtype], int_weight_switch,
                 int_weight, fin_ratio, fout_ratio) -> None:
        # 目前dtype放缩是在后置步骤完成的。TODO: 是否可简化？
        #if int_weight_switch and int_weight:
        #    rt_upscale_num += 1
        self.input_dtype = input_dtype
        self.runtime_dtype = dtype_scaler(self.input_dtype, rt_upscale_num)
        self.output_dtype = output_dtype if output_dtype else self.input_dtype
        self.int_weight = int_weight
        # upscale_time代表输出相比于运算过程量需要放缩255的倍数
        self.upscale_time = DTYPE_REVERSE_MAP[
            self.output_dtype] - DTYPE_REVERSE_MAP[input_dtype]

        # 视情况对运行中数据类型做变更
        if int_weight_switch:
            self.apply_int_weight(fin_ratio, fout_ratio)
            if self.int_weight and self.input_dtype != self.runtime_dtype:
                # 应用int_weight时，相当于已上放缩一次，此处还原。
                self.upscale_time -= 1
        if DTYPE_REVERSE_MAP[self.output_dtype] > DTYPE_REVERSE_MAP[
                self.runtime_dtype]:
            # 如果希望输出高于运行时精度的数据，警告（但仍然按需求输出）
            logger.warning(
                f"Expect output to be {output_dtype}, but runtime data type is {self.runtime_dtype}."
                + f" Accuracy can be duplicated.")

    def apply_int_weight(self, fin_ratio, fout_ratio):
        """根据渐入渐出选项变更dtype。

        Args:
            fin_ratio (_type_): _description_
            fout_ratio (_type_): _description_
        """
        # 整形输入的位深度选项
        if (fin_ratio == 0 and fout_ratio == 0):
            # 1权重不进行类型转换，计算时按*1d运算
            self.int_weight = True
        elif self.int_weight and self.input_dtype in DTYPE_UPSCALE_MAP:
            #INT权重 -> base图像扩大范围
            self.runtime_dtype = DTYPE_UPSCALE_MAP[self.input_dtype]
            logger.info(
                f"Upscale datatype from {self.input_dtype} To {self.runtime_dtype} to apply int-weight."
            )
        else:
            self.runtime_dtype = float

    def rescale(self, image: np.ndarray, power: int = 1):
        if self.upscale_time > 0:
            image = np.array(image, dtype=self.output_dtype)
            image *= (255**self.upscale_time)**(power)
        elif self.upscale_time:
            image = np.array(image // (255**(abs(self.upscale_time)))**(power),
                             dtype=self.output_dtype)
        else:
            image = np.array(image, dtype=self.output_dtype)
        return image


class GenericMasterBase(object):
    """GenericMaster 的基类。
    MasterBase类是叠加算法的入口类。继承该方法以实现特定的叠加功能。
    
    MasterBase类包含了大多数叠加必须使用的一些通用方法，如初始化必要参数，通过参考图像获取exif，colorprofile等配置信息，数据放缩等。
    
    如果期望实现的叠加仅需要遍历一次数据，可基于继承自GenericMasterBase的SimpleMasterTemplate进行开发。反之，如果流程中包含若干个 SimpleMaster 或者 GenericMaster 流程，直接继承该类并在 run 方法中执行必要的初始化，并串行设置自定义的中间流程即可。

    ## Usage
    该类按照如下方式被调用：
    1. 初始化
    2. 调用run方法，并返回一个Easydict对象，其中需要至少包含img[np.ndarray], colorprofile, exif字段。
    
    要继承该方法，主要需要override run() method。
    
    
    Args:
        object (_type_): _description_
    """

    def __init__(self) -> None:
        self.rt_upscale_num = 0
        # int_weight_switch 表示该模式是否支持int_weight。
        # 该处的int_weight_switch特指255权重。
        self.int_weight_switch = False

    def init_base_param(self, fname_list, **kwargs):
        self.fname_list = fname_list
        self.tot_length = len(fname_list)
        self.mp_num, self.sub_length = get_mp_num(self.tot_length)

    @time_cost_warpper
    def init_base_and_exif(self,
                           fname_list,
                           base_info: Any,
                           sample_img: Optional[np.ndarray],
                           info_load_mode="detail",
                           **kwargs):
        assert info_load_mode in [
            "detail", "first_only"
        ], "info_load_mode only supports `detail` and `first-only`."
        # 直接继承的情况下不进行初始化
        self.base_info: Any = base_info
        self.sample_img = sample_img
        if self.base_info is not None and self.sample_img is not None:
            return
        self.sample_img = load_sample_img(fname_list)
        # 获取EXIF信息和样本图像
        if info_load_mode == "detail":
            # detail模式下，会从所有文件中读取EXIF信息，并加和曝光时间。
            # 同时会读取EXIF中的宽高信息。如果存在不匹配宽高，抛出警告（可以被ignore？）
            time_cumsum = fractions.Fraction(0)
            for fname in fname_list:
                cur_info: Any = load_info(fname)
                time = cur_info.exif.get("Exif.Photo.ExposureTime")
                if time is not None:
                    time_cumsum += fractions.Fraction(time)
                if self.base_info is None:
                    self.base_info = cur_info
            if self.base_info.exif.get("Exif.Photo.ExposureTime") is not None:
                self.base_info.exif["Exif.Photo.ExposureTime"] = "/".join(
                    map(str, time_cumsum.as_integer_ratio()))
        else:
            # 直接使用张数估算总曝光时间
            self.base_info = load_info(fname_list[0])
            self.base_info.exif.update(
                "Exif.Photo.ExposureTime", "/".join(
                    map(str, (fractions.Fraction(
                        self.base_info.exif.get("Exif.Photo.ExposureTime")) *
                              len(fname_list)).as_integer_ratio())))
        logger.info(
            f"Calculated total exposure time = {self.base_info.exif.get('Exif.Photo.ExposureTime')}."
        )
        logger.info(
            f"ICC_profile = {get_color_profile(self.base_info.colorprofile)}")
        logger.info(f"EXIF TAG num = {len(self.base_info.exif)}")
        logger.debug(f"Raw basic infomation={self.base_info}")
        # 设置软体名称
        self.base_info.exif["Exif.Image.Software"] = SOFTWARE_NAME

    def init_dtype_recorder(self,
                            sample_img: np.ndarray,
                            output_bits: Optional[int] = None,
                            int_weight: bool = False,
                            fin_ratio: Optional[float] = None,
                            fout_ratio: Optional[float] = None):
        # 计算各阶段使用的dtype。
        # 如果某种方式的提升逻辑不一样，Override该函数即可。
        self.dtype_recorder = DtypeRecorder(
            input_dtype=sample_img.dtype,
            rt_upscale_num=self.rt_upscale_num,
            output_dtype=BITS2DTYPE[output_bits] if output_bits else None,
            int_weight_switch=self.int_weight_switch,
            int_weight=int_weight,
            fin_ratio=fin_ratio,
            fout_ratio=fout_ratio)

    def run(self, **kwargs):
        raise NotImplementedError


class SimpleMasterTemplate(GenericMasterBase):
    """WIP.

    Args:
        object (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        # 如果需要做对应改动，则需要实例化，并对实例化的模板执行对应操作
        self.subprocessor = run_merger_subprocess
        self.rt_upscale_num = 0
        self.default_progressbar = TqdmProgressbar(
            desc=f"Executing {self.__class__.__name__}")
        self.gen_weight_list = False
        self.sub_merger_type = MaxMerger
        self.main_merger_type = MaxMerger
        self.main_upscale = False

    def construct_ret(self) -> EasyDict:
        if self.main_merger.merged_image is not None:
            rescaled_main_img = self.dtype_recorder.rescale(
                self.main_merger.merged_image)
        else:
            rescaled_main_img = None
        return EasyDict(img=rescaled_main_img,
                        exif=self.base_info.exif,
                        colorprofile=self.base_info.colorprofile)

    @time_cost_warpper
    def run(self,
            fname_list: list[str],
            fin_ratio: float,
            fout_ratio: float,
            resize: Optional[str] = None,
            int_weight: bool = True,
            output_bits: Optional[int] = None,
            ground_mask: Optional[str] = None,
            debug_mode: Optional[bool] = None,
            base_info: Optional[EasyDict] = None,
            sample_img: Optional[np.ndarray] = None,
            progressbar: Optional[QueueProgressbar] = None,
            **kwargs) -> Optional[EasyDict]:
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
        self.init_base_param(fname_list=fname_list, **kwargs)
        has_scaled = False
        pool = mp.Pool(processes=self.mp_num)
        results = mp.Manager().Queue()
        self.main_merger = self.main_merger_type()

        self.init_base_and_exif(fname_list=fname_list,
                                base_info=base_info,
                                sample_img=sample_img,
                                **kwargs)
        # 如果无法获得样本图像，说明序列完全没有可用图像。直接退出。
        if self.sample_img is None:
            logger.error(
                "No sample image available: All reading attempt failed. " +
                "Check your input.")
            return EasyDict(img=None)

        self.init_dtype_recorder(self.sample_img, output_bits, int_weight,
                                 fin_ratio, fout_ratio)
        logger.info(
            f"Data scaling: Runtime scaling = {self.rt_upscale_num}; Output scaling = {self.dtype_recorder.upscale_time}."
        )
        if self.gen_weight_list:
            weight_list = generate_weight(self.tot_length,
                                          fin_ratio,
                                          fout_ratio,
                                          int_weight=int_weight)
        else:
            # default. for filling param only.
            weight_list = np.ones((self.tot_length, ), dtype=np.int8)

        # 对于需要快速预览的场景，resize_opt被设置为具体的值；否则为None。
        resize_opt = get_resize(resize, self.sample_img.shape[:2])

        # DEBUG INFO
        logger.info(
            f"Processor num = {self.mp_num}; int_weight = {self.dtype_recorder.int_weight}; dtype = {self.dtype_recorder.input_dtype}; resize = {resize_opt};"
        )
        logger.debug(f"Full weight list:{weight_list}")

        # 进度条指示器
        if not progressbar:
            progressbar = self.default_progressbar
        try:
            progressbar.reset(tot_num=self.tot_length)
            progressbar.start()
            # 多线程叠加
            for i in range(self.mp_num):
                l, r = int(i * self.sub_length), int((i + 1) * self.sub_length)
                pool.apply_async(
                    self.subprocessor,
                    kwds=dict(proc_id=i,
                              img_loader_type=ImgSeriesLoader,
                              merger_type=self.sub_merger_type,
                              fname_list=fname_list[l:r],
                              progressbar=progressbar.queue,
                              debug=debug_mode,
                              dtype=self.dtype_recorder.runtime_dtype,
                              resize=resize_opt,
                              weight_list=weight_list[l:r],
                              **kwargs),
                    callback=lambda ret: results.put(ret),
                    error_callback=lambda error: error_raiser(error))
            pool.close()
            # 合并多线程叠加结果
            # refresh in case
            self.main_merger.merged_image = None
            for i in range(self.mp_num):
                cur_img = results.get()
                self.main_merger.merge(cur_img)
                # A temp fix for datascaleup of MeanStacker.
                # If this is used for main_merger that do not have upscale method,
                # this could raise an Exception.
                # TODO: Fix this in the future.
                if (not has_scaled) and self.main_upscale:
                    logger.info("Upscale main merger dtype...")
                    self.main_merger.upscale()
                    has_scaled = True
            # 结束进度条
            pool.join()
            progressbar.stop()
            result_dict = self.construct_ret()

        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"{e.__repr__()} is triggered.")
            pool.terminate()
            pool.join()
            progressbar.stop()
            raise e
        # TODO: 异常处理
        return result_dict


class StarTrailMaster(SimpleMasterTemplate):

    def __init__(self) -> None:
        super().__init__()
        self.sub_merger_type = MaxMerger
        self.gen_weight_list = True
        self.int_weight_switch = True


class MinStackMaster(SimpleMasterTemplate):

    def __init__(self) -> None:
        super().__init__()
        self.sub_merger_type = MinMerger
        self.main_merger_type = MinMerger


class MeanStackMaster(SimpleMasterTemplate):

    def __init__(self) -> None:
        super().__init__()
        self.sub_merger_type = MeanMerger
        self.main_merger_type = MeanMerger
        self.rt_upscale_num = 1

    def init_dtype_recorder(self,
                            sample_img: np.ndarray,
                            output_bits: Optional[int] = None,
                            int_weight: bool = False,
                            fin_ratio: Optional[float] = None,
                            fout_ratio: Optional[float] = None):
        # 处理uint8输入且总张数大于255的场景
        if self.tot_length > 255 and sample_img.dtype == np.dtype("uint8"):
            if self.tot_length / self.mp_num > 255:
                # 单进程也大于255的话，runtime数据需要2次提升
                self.rt_upscale_num = 2
            else:
                # 否则仅需要在主进程内提升一次即可
                self.main_upscale = True
        super().init_dtype_recorder(sample_img, output_bits, int_weight,
                                    fin_ratio, fout_ratio)

    def construct_ret(self) -> EasyDict:
        if self.main_merger.merged_image is not None:
            rescaled_main_img = self.dtype_recorder.rescale(
                self.main_merger.merged_image.mu)
            rescaled_var_img = self.dtype_recorder.rescale(
                self.main_merger.merged_image.var, power=2)
            num = self.main_merger.merged_image.n
        else:
            rescaled_main_img = None
            rescaled_var_img = None
            num = None
        return EasyDict(img=rescaled_main_img,
                        var=rescaled_var_img,
                        raw=self.main_merger.merged_image,
                        n=num,
                        exif=self.base_info.exif,
                        colorprofile=self.base_info.colorprofile)


class SingleSigmaClippingMaster(MeanStackMaster):

    def __init__(self) -> None:
        super().__init__()
        self.sub_merger_type = SigmaClippingMerger

    # sigmaClipping de ref img should be scaled before being using as the ref img.
    # It is not allowed to rescale the ref img, actually.
    def construct_ret(self) -> EasyDict:
        main_img_param = self.full_ref_img - self.main_merger.merged_image
        rescaled_main_img = self.dtype_recorder.rescale(main_img_param.mu)
        rescaled_var_img = self.dtype_recorder.rescale(main_img_param.var,
                                                       power=2)
        return EasyDict(img=rescaled_main_img,
                        var=rescaled_var_img,
                        raw=main_img_param,
                        n=main_img_param.n,
                        exif=self.base_info.exif,
                        colorprofile=self.base_info.colorprofile)

    def init_base_param(self, fname_list: list[str],
                        **kwargs) -> Optional[EasyDict]:
        self.ref_img: FastGaussianParam = kwargs["ref_img"]
        self.full_ref_img: FastGaussianParam = kwargs["full_ref_img"]
        super().init_base_param(fname_list)


class SigmaClippingMaster(GenericMasterBase):
    """若干个SimpleMaster或者ComplexMaster进程。因此其主要执行流除了需要处理必要的初始化以外，通常需要包含高度自定义的中间流程。

    Args:
        object (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.subprocessor = None
        # 对于复杂模板类，该变量实际不被使用，但可用于指示中间支持的upscale数目。
        self.rt_upscale_num = 1
        self.progressbar = None
        self.gen_weight_list = False
        self.sub_merger_type = None
        self.main_merger_type = None
        self.main_upscale = False
        self.int_weight_switch = False

    @time_cost_warpper
    def run(self,
            fname_list: list[str],
            fin_ratio: float,
            fout_ratio: float,
            resize: Optional[str] = None,
            int_weight: bool = True,
            output_bits: Optional[int] = None,
            ground_mask: Optional[str] = None,
            debug_mode: Optional[bool] = None,
            base_info: Optional[EasyDict] = None,
            sample_img: Optional[np.ndarray] = None,
            progressbar: Optional[QueueProgressbar] = None,
            rej_high: float = 3.0,
            rej_low: float = 3.0,
            max_iter: int = 5,
            **kwargs):
        assert max_iter > 0, "max_iter must >0 !"
        self.init_base_param(fname_list)
        self.init_base_and_exif(fname_list,
                                base_info=base_info,
                                sample_img=sample_img)
        # 如果无法获得样本图像，说明序列完全没有可用图像。直接退出。
        if self.sample_img is None:
            logger.error(
                "No sample image available: All reading attempt failed. " +
                "Check your input.")
            return EasyDict(img=None)
        self.init_dtype_recorder(self.sample_img,
                                 output_bits,
                                 int_weight=False,
                                 fin_ratio=None,
                                 fout_ratio=None)

        base_mean_stacker = MeanStackMaster()
        base_result_dict = base_mean_stacker.run(fname_list=fname_list,
                                                 fin_ratio=fin_ratio,
                                                 fout_ratio=fout_ratio,
                                                 resize=resize,
                                                 int_weight=int_weight,
                                                 output_bits=None,
                                                 ground_mask=ground_mask,
                                                 debug_mode=debug_mode,
                                                 base_info=self.base_info,
                                                 sample_img=self.sample_img,
                                                 progressbar=progressbar)
        single_sigmaclipping = SingleSigmaClippingMaster()
        cur_iter = 0
        last_clip_num = base_result_dict.n
        iter_result = base_result_dict
        while cur_iter < max_iter:
            cur_iter += 1
            logger.info(f"SigmaClipping iter#{cur_iter}...")
            iter_result = single_sigmaclipping.run(
                fname_list=fname_list,
                fin_ratio=fin_ratio,
                fout_ratio=fout_ratio,
                resize=resize,
                int_weight=int_weight,
                output_bits=None,
                ground_mask=ground_mask,
                debug_mode=debug_mode,
                base_info=self.base_info,
                sample_img=self.sample_img,
                progressbar=progressbar,
                full_ref_img=base_result_dict.raw,
                ref_img=iter_result.raw,
                rej_input_dtype=self.dtype_recorder.input_dtype,
                rej_high=rej_high,
                rej_low=rej_low)
            if ((last_clip_num - iter_result.n).all() == 0):
                logger.info("Early convergence detected.")
                break
            last_clip_num = iter_result.n
        rescaled_main_img = self.dtype_recorder.rescale(iter_result.img)
        rescaled_var_img = self.dtype_recorder.rescale(iter_result.var,
                                                       power=2)
        return EasyDict(img=rescaled_main_img,
                        var=rescaled_var_img,
                        raw=iter_result,
                        n=iter_result.n,
                        exif=self.base_info.exif,
                        colorprofile=self.base_info.colorprofile)


class SimpleMixTrailMaster(GenericMasterBase):
    """ComplexMasterTemplate通常包含若干个SimpleMaster或者ComplexMaster进程。因此其主要执行流除了需要处理必要的初始化以外，通常需要包含高度自定义的中间流程。

    Args:
        object (_type_): _description_
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @time_cost_warpper
    def run(self,
            fname_list: list[str],
            fin_ratio: float,
            fout_ratio: float,
            resize: Optional[str] = None,
            int_weight: bool = True,
            output_bits: Optional[int] = None,
            ground_mask: Optional[str] = None,
            debug_mode: Optional[bool] = None,
            base_info: Optional[EasyDict] = None,
            sample_img: Optional[np.ndarray] = None,
            progressbar: Optional[QueueProgressbar] = None,
            rej_high: float = 3.0,
            rej_low: float = 3.0,
            max_iter: int = 5,
            **kwargs):
        assert max_iter > 0, "max_iter must >0 !"
        self.init_base_param(fname_list)
        self.init_base_and_exif(fname_list,
                                base_info=base_info,
                                sample_img=sample_img)
        # 如果无法获得样本图像，说明序列完全没有可用图像。直接退出。
        if self.sample_img is None:
            logger.error(
                "No sample image available: All reading attempt failed. " +
                "Check your input.")
            return EasyDict(img=None)
        self.init_dtype_recorder(self.sample_img, output_bits)
        # 对于需要快速预览的场景，resize_opt被设置为具体的值；否则为None。
        resize_opt = get_resize(resize, self.sample_img.shape[:2])
        assert ground_mask is not None, "this mode should launch with a ground mask!"
        mask = load_img(ground_mask, resize=resize_opt)
        if mask is None:
            logger.error("Fail to load mask.")
            return EasyDict(img=None)

        sigma_mean_stacker = SigmaClippingMaster()
        mean_result_dict = sigma_mean_stacker.run(fname_list=fname_list,
                                                  fin_ratio=fin_ratio,
                                                  fout_ratio=fout_ratio,
                                                  resize=resize,
                                                  int_weight=int_weight,
                                                  output_bits=output_bits,
                                                  ground_mask=ground_mask,
                                                  debug_mode=debug_mode,
                                                  base_info=self.base_info,
                                                  sample_img=self.sample_img,
                                                  progressbar=progressbar,
                                                  rej_high=rej_high,
                                                  rej_low=rej_low,
                                                  max_iter=max_iter)
        max_stacker = StarTrailMaster()
        max_result_dict = max_stacker.run(fname_list=fname_list,
                                          fin_ratio=fin_ratio,
                                          fout_ratio=fout_ratio,
                                          resize=resize,
                                          int_weight=int_weight,
                                          output_bits=output_bits,
                                          ground_mask=ground_mask,
                                          debug_mode=debug_mode,
                                          base_info=self.base_info,
                                          sample_img=self.sample_img,
                                          progressbar=progressbar)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.repeat(mask[..., None], 3, axis=-1)
        div_num = DTYPE_MAX_VALUE[
            mask.dtype] if mask.dtype in DTYPE_MAX_VALUE else np.max(mask)
        float_mask = np.array(mask, dtype=float) / div_num
        ratio = get_max_expmean(self.tot_length)
        diff_img = ratio * np.sqrt(mean_result_dict.var)
        reg_max_img = max_result_dict.img - diff_img
        logger.info(f"fix ratio = {ratio:.4f}")
        merged_img = reg_max_img * float_mask + (
            1 - float_mask) * mean_result_dict.img
        # 不需要处理放缩，仅需要转换格式
        merged_img = np.array(merged_img, dtype = self.dtype_recorder.output_dtype)
        return EasyDict(img=merged_img,
                        exif=self.base_info.exif,
                        colorprofile=self.base_info.colorprofile)