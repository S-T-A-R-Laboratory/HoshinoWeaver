import argparse
import os
import sys
from ezlib.progressbar import QueueProgressbar
from ezlib.trailstacker import SigmaClippingMaster, StarTrailMaster, MeanStackMaster, SimpleMixTrailMaster, MinStackMaster
from ezlib.imgfio import save_img
from loguru import logger
from typing import Optional

from ezlib.utils import is_support_format

modestr2func = {
    "max": StarTrailMaster,
    "min": MinStackMaster,
    "mean": MeanStackMaster,
    "sigmaclip-mean": SigmaClippingMaster,
    "mask-mix": SimpleMixTrailMaster
}


def launch(img_files: list,
                mode: str,
                output_fname: str,
                fin_ratio: Optional[float] = None,
                fout_ratio: Optional[float] = None,
                int_weight: bool = False,
                resize: Optional[str] = None,
                output_bits: Optional[int] = None,
                ground_mask: Optional[str] = None,
                debug_mode: bool = False,
                progressbar: Optional[QueueProgressbar] = None,
                rej_high: float = 3.0,
                rej_low: float = 3.0,
                max_iter: int = 5,
                png_compressing: int = 0,
                jpg_quality: int = 90,
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
        rej_high (float, optional): Sigma裁剪均值的拒绝上界倍率。仅在使用了Sigma裁剪均值的模式下生效。 Defaults to 3.0.
        rej_low (float, optional): Sigma裁剪均值的拒绝下界倍率。仅在使用了Sigma裁剪均值的模式下生效。 Defaults to 3.0.
        max_iter (int, optional): Sigma裁剪均值的最大迭代轮数。 Defaults to 5.
        png_compressing (int): PNG compressing arguments, ranges from 0 (no compressing) to 9. Defaults to 0.
        jpg_quality (int): JPG quality parameter, ranges from 0 to 100. Defaults to 90.

    Returns:
        dict: 该函数返回一个包含"success"与"message"两个字段字典。其中，"success"字段取值为bool，代表是否正常输出了结果；
            如果失败，"message"中会带有失败原因的字符串。
    """
    try:
        assert mode in modestr2func, f" Got unsupport mode `{mode}`. Should be selected from {modestr2func.keys()}"
        runner = modestr2func[mode]()
        res = runner.run(fname_list=img_files,
                         fin_ratio=fin_ratio,
                         fout_ratio=fout_ratio,
                         int_weight=int_weight,
                         resize=resize,
                         output_bits=output_bits,
                         ground_mask=ground_mask,
                         debug_mode=debug_mode,
                         progressbar=progressbar,
                         rej_high=rej_high,
                         rej_low=rej_low,
                         max_iter=max_iter)
        if res.img is None:
            return {"success": False, "message": "空结果"}
        else:
            save_img(output_fname,
                     res.img,
                     png_compressing=png_compressing,
                     jpg_quality=jpg_quality,
                     exif=res.exif,
                     colorprofile=res.colorprofile)
            return {"success":True, "message": None}
    except Exception as e:
        raise e
        return {"success": False, "message": f"程序因为以下原因失败：{e.__repr__()}"}


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dirname", help="dir of images")
    arg_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["mean", "max", "min", "mask-mix", "sigmaclip-mean"],
        help="stack mode")
    arg_parser.add_argument("--ground-mask",
                            type=str,
                            help="/path/to/the/mask.file",
                            default=None)
    arg_parser.add_argument("--fade-in", type=float, default=0.1)
    arg_parser.add_argument("--fade-out", type=float, default=0.1)
    arg_parser.add_argument("--int-weight", action="store_true")
    arg_parser.add_argument("--jpg-quality", type=int, default=90)
    arg_parser.add_argument("--png-compressing", type=int, default=0)
    arg_parser.add_argument("--output",
                            type=str,
                            required=False,
                            default="default.jpg")
    arg_parser.add_argument("--output-bits",
                            type=int,
                            choices=[8, 16, 32],
                            help="the bit of output image.")
    arg_parser.add_argument("--resize", type=str, default=None)
    arg_parser.add_argument("--num-processor",
                            type=int,
                            default=None,
                            help="max available processor num.")
    arg_parser.add_argument("--debug",
                            action="store_true",
                            help="print logs with debug level.")
    args = arg_parser.parse_args()

    # TODO: 增加文件中的日志
    logger.remove()
    if args.debug:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")

    dir_name = args.dirname
    fin_ratio, fout_ratio = float(args.fade_in), float(args.fade_out)
    output_file = args.output

    # get filename list in the directory
    img_files = os.listdir(dir_name)
    img_files.sort()
    img_files = [
        os.path.join(dir_name, x) for x in img_files if is_support_format(x)
    ]
    launch(img_files, 
           args.mode,
           output_file,
           fin_ratio=fin_ratio,
           fout_ratio=fout_ratio,
           int_weight=args.int_weight,
           resize=args.resize,
           output_bits=args.output_bits,
           ground_mask=args.ground_mask,
           debug_mode=args.debug,
           rej_high=3.0,
           rej_low=3.0,
           max_iter=5)