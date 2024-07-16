import argparse
import os
import sys
from ezlib.trailstacker import SigmaClippingMaster, StarTrailMaster, MeanStackMaster, SimpleMixTrailMaster
from ezlib.imgfio import save_img
from loguru import logger

from ezlib.utils import is_support_format

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dirname", help="dir of images")
    arg_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["mean", "max", "mask-mix", "sigmaclip-mean"],
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
    arg_parser.add_argument("--output", type=str, required=True)
    arg_parser.add_argument("--output-bits",
                            type=int,
                            choices=[8, 16, 32],
                            help="the bit of output image.")
    arg_parser.add_argument("--resize", type=str, default=None)
    arg_parser.add_argument("--num_processor",
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

    if args.mode == "max":
        res = StarTrailMaster(img_files,
                              fin_ratio=fin_ratio,
                              fout_ratio=fout_ratio,
                              resize=args.resize,
                              int_weight=args.int_weight,
                              output_bits=args.output_bits,
                              ground_mask=args.ground_mask)
    elif args.mode == "mean":
        res = MeanStackMaster(img_files,
                              resize=args.resize,
                              output_bits=args.output_bits,
                              ground_mask=args.ground_mask)
    elif args.mode == "mask-mix":
        res = SimpleMixTrailMaster(img_files,
                                   fin_ratio=fin_ratio,
                                   fout_ratio=fout_ratio,
                                   int_weight=args.int_weight,
                                   resize=args.resize,
                                   output_bits=args.output_bits,
                                   ground_mask=args.ground_mask)
    elif args.mode == "sigmaclip-mean":
        res = SigmaClippingMaster(img_files, output_bits=args.output_bits)
    else:
        raise NotImplementedError(
            f"NotImplementedMode: {args.mode}. Only \"max\" and \"mean\" are supported."
        )
    if res.img is None:
        logger.error(f"Got empty result.")
        exit()
    save_img(output_file,
             res.img,
             png_compressing=args.png_compressing,
             jpg_quality=args.jpg_quality,
             exif=res.exif,
             colorprofile=res.colorprofile)
