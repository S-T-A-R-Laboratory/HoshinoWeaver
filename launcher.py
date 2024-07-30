import argparse
import os
import sys
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


def gui_request(img_files: list,
                mode: str,
                output_fname: str,
                fin_ratio: Optional[float] = None,
                fout_ratio: Optional[float] = None,
                int_weight: bool = False,
                resize: Optional[str] = None,
                output_bits: Optional[int] = None,
                ground_mask: Optional[str] = None,
                debug_mode: bool = False,
                rej_high: float = 3.0,
                rej_low: float = 3.0,
                max_iter: int = 5,
                png_compressing: int = 0,
                jpg_quality: int = 90,
                **kwargs) -> dict:
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

    assert args.mode in modestr2func, f" Got unsupport mode `{args.mode}`. Should be selected from {modestr2func.keys()}"
    runner = modestr2func[args.mode]()

    res = runner.run(fname_list=img_files,
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

    if res.img is None:
        logger.error(f"Got empty result.")
        exit()
    save_img(output_file,
             res.img,
             png_compressing=args.png_compressing,
             jpg_quality=args.jpg_quality,
             exif=res.exif,
             colorprofile=res.colorprofile)
