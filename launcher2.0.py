from hoshicore.engine.wiring import run_from_yaml
import asyncio
import argparse
import os
import sys

from loguru import logger
from hoshicore.component.utils import is_support_format, init_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run the star trail stacking process.")
    parser.add_argument("config",
                        type=str,
                        help="Path to the YAML configuration file.")
    parser.add_argument("dir",
                        type=str,
                        help="Directory containing the input images.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=1,
                        help="Number of worker processes to use (default: 1).")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--debug",
                           action="store_true",
                           help="Enable DEBUG level logging.")
    log_group.add_argument(
        "--trace",
        action="store_true",
        help="Enable TRACE level logging (includes [MEM] diagnostics).")

    args = parser.parse_args()

    logger = init_logger(logger, args.debug, args.trace, None)

    yaml_path = args.config
    if not os.path.isfile(yaml_path):
        logger.error(f"Config file does not exist: {yaml_path}")
        sys.exit(1)
    dir_name = args.dir
    num_workers = args.num_workers

    # get filename list in the directory
    img_files = os.listdir(dir_name)
    img_files.sort()
    img_files = [
        os.path.join(dir_name, x) for x in img_files if is_support_format(x)
    ]

    global_inputs = {"fnames": img_files}
    global_configs = {
        "fin": 0.2,
        "fout": 0.2,
        "int_weight": True,
        "output_dtype": "uint16",
        "buffer_mode": "auto"
    }

    asyncio.run(
        run_from_yaml(yaml_path,
                      global_inputs,
                      global_configs,
                      num_workers=num_workers))


if __name__ == "__main__":
    main()
