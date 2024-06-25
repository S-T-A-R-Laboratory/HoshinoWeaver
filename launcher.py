import argparse
import os
import sys
from ezlib.trailstacker import StarTrailMaster, MeanTrackMaster
from ezlib.imgfio import save_img
from loguru import logger

from ezlib.utils import is_support_format

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dirname", help="dir of images")
    arg_parser.add_argument("--mode",
                            type=str,
                            required=True,
                            choices=["mean", "max"],
                            help="stack mode")
    arg_parser.add_argument("--fade-in", type=float, default=0.1)
    arg_parser.add_argument("--fade-out", type=float, default=0.1)
    arg_parser.add_argument("--int-weight", action="store_true")
    arg_parser.add_argument("--jpg-quality", type=int, default=90)
    arg_parser.add_argument("--png-compressing", type=int, default=1)
    arg_parser.add_argument("--output", type=str)
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

    if args.mode == "max":
        res = StarTrailMaster(img_files,
                              fin_ratio=fin_ratio,
                              fout_ratio=fout_ratio,
                              int_weight=args.int_weight)
    elif args.mode == "mean":
        res = MeanTrackMaster(img_files)

    save_img(output_file,
             res.img,
             png_compressing=args.png_compressing,
             jpg_quality=args.jpg_quality,
             exif=res.exif,
             colorprofile=res.colorprofile)
