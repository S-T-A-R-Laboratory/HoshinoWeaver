from typing import Optional

from easydict import EasyDict
from loguru import logger

from ezlib.utils import time_cost_warpper

from .imgfio import analyze_attr, get_img_attrs_by_pil, save_img
from .progressbar import QueueProgressbar
from .trailstacker import (DataArrayMaster, MeanStackMaster, MinStackMaster,
                           SigmaClippingMaster, SimpleMixTrailMaster,
                           StarTrailMaster)

modestr2func = {
    "max": StarTrailMaster,
    "min": MinStackMaster,
    "mean": MeanStackMaster,
    "sigmaclip-mean": SigmaClippingMaster,
    "mask-mix": SimpleMixTrailMaster,
}


def launch(img_files: list,
           mode: str,
           output_fname: str,
           fin_ratio: float = 0,
           fout_ratio: float = 0,
           int_weight: bool = False,
           resize: Optional[str] = None,
           output_bits: Optional[int] = None,
           ground_mask: Optional[str] = None,
           debug_mode: bool = False,
           progressbar: Optional[QueueProgressbar] = None,
           num_processor: Optional[int] = None,
           rej_high: float = 3.0,
           rej_low: float = 3.0,
           max_iter: int = 5,
           png_compressing: int = 0,
           jpg_quality: int = 90,
           check_exif: bool = False,
           cache: Optional[EasyDict] = None,
           **kwargs) -> dict:
    """端到端的请求接口。将按照模式和参数设置叠加图像，并输出图像到给定路径下。
    
    并非所有参数都在给定模式下生效。如果指定的叠加模式不需要某些参数，可以不传入参数或者传入None。

    Args:
        img_files (list): 图像文件名列表。建议表示为绝对路径形式。
        mode (str): 叠加模式。目前可选"max"（最大值）、"min"（最小值）、"mean"（平均值）、
            "sigmaclip-mean"（Sigma裁剪均值）、"mask-mix"（使用掩膜混合的最大值+平均值）。
        output_fname (str): 输出文件名。建议表示为绝对路径形式。
        fin_ratio (Optional[float], optional): 星轨渐入比例，取值范围[0,1]。仅在包含星轨的模式下生效。 Defaults to None.
        fout_ratio (Optional[float], optional): 星轨渐出比例，取值范围[0,1]。仅在包含星轨的模式下生效。 Defaults to None.
        int_weight (bool, optional): 是否启用为星轨整形权重代替浮点权重。仅在包含星轨的模式下生效。
            大多数情况下，该开关以可接受的精度损失换取更快的叠加速度。 Defaults to False.
        resize (Optional[str], optional): 图像缩放大小。支持以"1920x1080"形式同时指定长宽（顺序为宽-高），
            或单独的数字"1920"指定长边。不配置时，默认不放缩。 Defaults to None.
        output_bits (Optional[int], optional): 输出图像位深度。支持[8,16,32]。不配置时，
            默认与输入图像位深度一致，除非指定了必须为8位深度的输出格式（如jpg）。 Defaults to None.
        ground_mask (Optional[str], optional): 掩模文件路径。建议表示为绝对路径。仅在需要掩模的模式下使用。 Defaults to None.
        debug_mode (bool, optional): 是否在debug级别打印日志. Defaults to False.
        progressbar (QueueProgressbar, Optional): 進度條實例。如果期望使用GUI或者其他形勢進度條，需要傳入實例。否則，使用默認的命令行進度條。Defaults to None.
        num_processor (Optional[int], optional): 期望的并行进程数量. Defaults to None.
        rej_high (float, optional): Sigma裁剪均值的拒绝上界倍率。仅在使用了Sigma裁剪均值的模式下生效。 Defaults to 3.0.
        rej_low (float, optional): Sigma裁剪均值的拒绝下界倍率。仅在使用了Sigma裁剪均值的模式下生效。 Defaults to 3.0.
        max_iter (int, optional): Sigma裁剪均值的最大迭代轮数。 Defaults to 5.
        png_compressing (int): PNG compressing arguments, ranges from 0 (no compressing) to 9. Defaults to 0.
        jpg_quality (int): JPG quality parameter, ranges from 0 to 100. Defaults to 90.
        check_exif (bool, optional): 是否在运行前检查EXIF. Defaults to False.
        cache (Optional[EasyDict]): 缓存图像。当该项非空时，将直接使用缓存进行叠加。Defaults to None。

    Returns:
        dict: 该函数返回一个包含"status"与"message"两个字段字典。其中，"status"字段取值为bool，代表是否正常输出了结果；
            如果失败，"message"中会带有失败原因的字符串。
    """
    try:
        assert mode in modestr2func, f" Got unsupport mode `{mode}`. Should be selected from {modestr2func.keys()}"
        # 前置检查：如果--output-bits选择非8位的情况下输出jpg，需要校正并警告。
        if output_fname is not None and (output_fname.lower().split(".")[-1]
                                         in ["jpg", "jpeg"]
                                         and output_bits != 8):
            logger.warning(
                "JPEG only supports 8-bit image output. output bits are forced fixing to 8."
            )
            output_bits = 8
        if check_exif:
            logger.info(scan_all_exif(img_files))
        runner = modestr2func[mode]()

        if cache is not None:
            # 缓存模式，从内存直接进行叠加
            logger.info("Got cache, run in `run_in_memory` mode.")
            res = runner.run_in_memory(cache=cache,
                                       fin_ratio=fin_ratio,
                                       fout_ratio=fout_ratio,
                                       int_weight=int_weight,
                                       resize=resize,
                                       output_bits=output_bits,
                                       ground_mask=ground_mask,
                                       debug_mode=debug_mode)
        else:
            # 非缓存模式，从硬盘读取的多进程叠加
            res = runner.run(fname_list=img_files,
                             fin_ratio=fin_ratio,
                             fout_ratio=fout_ratio,
                             int_weight=int_weight,
                             resize=resize,
                             output_bits=output_bits,
                             ground_mask=ground_mask,
                             debug_mode=debug_mode,
                             progressbar=progressbar,
                             num_processor=num_processor,
                             rej_high=rej_high,
                             rej_low=rej_low,
                             max_iter=max_iter)
        if res.img is None:
            return {"status": False, "message": "空结果"}
        elif output_fname is not None:
            save_img(output_fname,
                     res.img,
                     png_compressing=png_compressing,
                     jpg_quality=jpg_quality,
                     exif=res.exif,
                     colorprofile=res.colorprofile)
            return {"status": True, "message": None}
    except (KeyboardInterrupt, Exception) as e:
        return {
            "status": False,
            "message": f"程序因为以下原因终止：{e.__repr__()}",
            "exception": e
        }


@time_cost_warpper
def scan_all_exif(fname_list: list[str]) -> list:
    """
    快速检查输入，并给出一系列可能会导致叠加任务无法正常进行的风险提示。
    
    目前有后缀名检查suffix，图像尺寸检查size_str，位数检查bits。位数检查有一定局限性，tiff不支持（pillow的底层问题，对tiff支持弱）

    由于部分数值不一定能够读取到，不推荐作为强制卡控。

    返回一个dict的list。每个dict包含5个字段：
    
    1. 检查的属性名 attr_name (str)
    2. 该属性最主要的模式 mode_attr (str)
    3. 主要模式的数量 mode_num (int)
    4. 其他模式的及数量分布 other_dist (Optional[list[tuple[str,int]]])
    5. 非主要模式的文件名列表 other_fname_list (Optional[list])

    Args:
        fname_list (list[str]): 文件名列表

    Returns:
        list[dict]: 返回风险提示列表。
    """
    attr_list = list(map(get_img_attrs_by_pil, fname_list))
    # 后缀名检查
    suffix_dict = analyze_attr(attr_list, "suffix")
    # 尺寸检查
    size_dict = analyze_attr(attr_list, "size")
    # 位数检查
    bits_dict = analyze_attr(attr_list, "bits")
    return [suffix_dict, size_dict, bits_dict]


def create_cache(img_files: list,
                 resize: Optional[str] = None,
                 output_bits: Optional[int] = None,
                 debug_mode: bool = False,
                 progressbar: Optional[QueueProgressbar] = None,
                 num_processor: Optional[int] = None,
                 check_exif: bool = False,
                 **kwargs):
    """将给定图像文件序列加载到内存中。可用于运行直接加载到内存的叠加，或者用于预览前的图像加载。

    Args:
        img_files (list): 图像文件名列表。建议表示为绝对路径形式。
        resize (Optional[str], optional): 图像缩放大小。支持以"1920x1080"形式同时指定长宽（顺序为宽-高），
            或单独的数字"1920"指定长边。不配置时，默认不放缩。 Defaults to None. 预览时推荐配置为具体值，如"960".
        output_bits (Optional[int], optional): 输出图像位深度。支持[8,16,32]。Defaults to None. 预览时推荐配置为具体值，如8。
        debug_mode (bool, optional): 是否在debug级别打印日志. Defaults to False.
        progressbar (QueueProgressbar, Optional): 進度條實例。如果期望使用GUI或者其他形勢進度條，需要傳入實例。否則，使用默認的命令行進度條。Defaults to None.
        num_processor (Optional[int], optional): 期望的并行进程数量. Defaults to None.
        check_exif (bool, optional): 是否在运行前检查EXIF. Defaults to False.
    """
    try:
        if check_exif:
            logger.info(scan_all_exif(img_files))
        cache_runner = DataArrayMaster()
        # TODO: 缺省参数优化
        res = cache_runner.run(fname_list=img_files,
                               fin_ratio=0,
                               fout_ratio=0,
                               int_weight=False,
                               resize=resize,
                               output_bits=output_bits,
                               debug_mode=debug_mode,
                               progressbar=progressbar,
                               num_processor=num_processor)
    except (KeyboardInterrupt, Exception) as e:
        return None
    return res