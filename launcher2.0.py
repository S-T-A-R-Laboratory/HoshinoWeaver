from hoshicore.engine.wiring import run_from_yaml
import asyncio
import argparse
import os
from hoshicore.component.utils import is_support_format

parser = argparse.ArgumentParser(
    description="Run the star trail stacking process.")
parser.add_argument("dir",
                    type=str,
                    help="Directory containing the input images.")

args = parser.parse_args()
dir_name = args.dir

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
    "output_dtype": "uint8",
}

asyncio.run(
    run_from_yaml("./hoshicore/dag/fix_startrail.yaml", global_inputs,
                  global_configs))
