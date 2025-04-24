from __future__ import annotations

import argparse
import datetime
import os
import sys

from loguru import logger

from ezlib import launch
from ezlib.trailstacker import ON_ERR_CONTINUE, ON_ERR_STOP
from ezlib.utils import SOFTWARE_NAME, VERSION, is_support_format

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
    arg_parser.add_argument("--fade-in", type=float, default=0)
    arg_parser.add_argument("--fade-out", type=float, default=0)
    # int weight serve as the default option since v0.5.0.
    # this argument is now only kept for backward compatibility only.
    weight_group = arg_parser.add_mutually_exclusive_group()
    weight_group.add_argument("--int-weight",
                              action="store_true",
                              default=True,
                              help="Use integer weights (default).")
    weight_group.add_argument("--float-weight",
                              action="store_true",
                              help="Use floating-point weights.")
    arg_parser.add_argument("--jpg-quality", type=int, default=90)
    arg_parser.add_argument("--png-compressing", type=int, default=0)
    arg_parser.add_argument("--output", type=str, required=False)
    arg_parser.add_argument("--output-bits",
                            type=int,
                            choices=[8, 16],
                            help="the bit of output image.")
    arg_parser.add_argument("--resize", type=str, default=None)
    arg_parser.add_argument("--num-processor",
                            type=int,
                            default=None,
                            help="max available processor num.")
    arg_parser.add_argument("--filter",
                            type=str,
                            nargs="*",
                            help="filters for every single images.")
    arg_parser.add_argument("--debug",
                            action="store_true",
                            help="print logs with debug level.")
    arg_parser.add_argument("--log-path",
                            type=str,
                            help="print logs to the given path.")
    arg_parser.add_argument("--on-error",
                            type=str,
                            choices=[ON_ERR_STOP, ON_ERR_CONTINUE],
                            default=ON_ERR_STOP,
                            help="define the action when an error occurs.")
    args = arg_parser.parse_args()

    dir_name = args.dirname
    fin_ratio, fout_ratio = float(args.fade_in), float(args.fade_out)
    output_file = args.output

    # get filename list in the directory
    img_files = os.listdir(dir_name)
    img_files.sort()
    img_files = [
        os.path.join(dir_name, x) for x in img_files if is_support_format(x)
    ]

    # 命令行模式下，日志文件名称直接通过时间戳生成
    log_path = None
    if args.log_path is not None:
        log_filename = f"{SOFTWARE_NAME}_{VERSION}_LOG_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        log_path = f"{args.log_path}\\{log_filename}"

    ret_json = launch(img_files,
                      args.mode,
                      output_file,
                      fin_ratio=fin_ratio,
                      fout_ratio=fout_ratio,
                      int_weight=(not args.float_weight),
                      resize=args.resize,
                      output_bits=args.output_bits,
                      ground_mask=args.ground_mask,
                      filter_list=args.filter,
                      num_processor=args.num_processor,
                      debug_mode=args.debug,
                      on_err_action=args.on_error,
                      rej_high=3.0,
                      rej_low=3.0,
                      max_iter=5,
                      check_exif=True,
                      log_path=log_path)
    if not ret_json["status"]:
        logger.error(ret_json)
        if ret_json["exception"]:
            raise ret_json["exception"]
