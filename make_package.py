from __future__ import annotations
import argparse
import os
import platform as pf
import shutil
import subprocess
import sys
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import Any
from loguru import logger

from ezlib.utils import VERSION, SOFTWARE_NAME, time_cost_warpper

# alias
join_path = os.path.join

CLI_FILENAME = "launcher"
GUI_FILENAME = "HoshinoWeaver desktop"

logger.remove()
logger.add(sys.stdout, level="INFO")

@time_cost_warpper
def nuitka_compile(header, options, target):
    """使用nuitka编译打包的API

    Args:
        header (str): 启动的指令，如python -m nuitka 或 nuitka （取决于平台）。
        option_list (dict): 编译选项列表。
        tgt (str): 待编译目标。
    """
    options_list = [
        key if value == True else f'{key}={value}'
        for key, value in options.items() if value
    ]

    merged = header + options_list + [
        target,
    ]

    ret_code = subprocess.run(merged).returncode
    logger.info(f"Compiled {target} finished with return code = {ret_code}.")

    # 异常提前终止
    if ret_code != 0:
        logger.error(
            f"Fatal compile error occured when compiling {target}. Compile terminated."
        )
        exit(-1)


def file_to_zip(path_original, z):
    '''
    Copied from https://blog.51cto.com/lanzao/4994053
     作用：压缩文件到指定压缩包里
     参数一：压缩文件的位置
     参数二：压缩后的压缩包
    '''
    f_list = list(Path(path_original).glob("**/*"))
    for f in f_list:
        z.write(f, str(f)[len(path_original):])


platform_mapping = {
    "win32": "win",
    "cygwin": "win",
    "darwin": "macos",
    "linux2": "linux",
    "linux": "linux"
}

platform2icon_option = {
    "win": "--windows-icon-from-ico",
    "linux": "--linux-icon",
    "macos13+": "--macos-app-icon"
}

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--mingw64",
    action="store_true",
    help=
    "Use mingw64 as compiler. This option only works for nuitka under Windows.",
    default=False)
argparser.add_argument(
    "--apply-upx",
    action="store_true",
    help="Apply UPX to squeeze the size of executable program.",
)
argparser.add_argument(
    "--apply-zip",
    action="store_true",
    help="Generate .zip files after packaging.",
)
argparser.add_argument(
    "--debug-gui",
    action="store_true",
    help="Generate gui version with console.",
)

args = argparser.parse_args()
apply_upx = args.apply_upx
apply_zip = args.apply_zip

# 根据平台/版本决定确定编译/打包后的程序后缀

platform = platform_mapping[sys.platform]
exec_suffix = ""
if (platform == "win"):
    exec_suffix = ".exe"
if (platform == "macos"):
    mac_main_ver = int(pf.mac_ver()[0].split(".")[0])
    if mac_main_ver >= 13:
        exec_suffix = ".bin"
        platform += "13+"

# 设置工作路径，避免出现相对路径引用错误
work_path = os.path.dirname(os.path.abspath(__file__))
compile_path = join_path(work_path, "dist")

t0 = time.time()

# 检查python版本 必要时启用alias python3
compile_tool = [sys.executable, "-m", "nuitka"]
# 将header作为偏函数打包，简化后续传参
nuitka_compile = partial(nuitka_compile, compile_tool)

# 构建共用的打包选项，根据编译平台选择是否启用mingw64
nuitka_base: dict[str, Any] = {
    "--no-pyi-file": True,
    "--remove-output": True,
}
if platform == "win" and args.mingw64:
    logger.info("Apply mingw64 as compiler.")
    nuitka_base["--mingw64"] = True

# upx启用时，利用which获取upx路径
if apply_upx:
    upx_cmd = subprocess.run(["which", "upx"])
    if upx_cmd.returncode == 0:
        nuitka_base["--plugin-enable"] = "upx"
        nuitka_base["--upx-binary"] = upx_cmd.stdout

if not platform.startswith("macos"):
    nuitka_base["--include-package"] = "pyexiv2"
nuitka_base["--standalone"] = True

# 编译GUI
gui_cfg = {
    "--output-dir": compile_path,
    "--enable-plugin": "pyside6",
    "--disable-console": (not args.debug_gui),
    "--nofollow-import-to": "opencv,matplotlib",
    platform2icon_option[platform]: "./imgs/HNW.jpg"
}
if platform.startswith("macos") and (not args.debug_gui):
    gui_cfg["--macos-create-app-bundle"] = True

gui_cfg.update(nuitka_base)
nuitka_compile(gui_cfg, target=join_path(work_path, f"{GUI_FILENAME}.py"))

# 但基本还是要把pyexiv2复制到输出目录下
try:
    import pyexiv2
    pyexiv_path, init_file = os.path.split(pyexiv2.__file__)
    shutil.copytree(
        pyexiv_path,
        join_path(compile_path, os.path.join(f"{GUI_FILENAME}.dist",
                                             "pyexiv2")))
except Exception as e:
    logger.warning(
        "Failed to load pyexiv2. Related function can be unavailable in release verion."
    )

# 编译CLI

cli_cfg = {
    "--standalone": True,
    "--output-dir": compile_path,
    "--nofollow-import-to": "pyside6,opencv,matplotlib",
}
cli_cfg.update(nuitka_base)
nuitka_compile(cli_cfg, target=join_path(work_path, f"{CLI_FILENAME}.py"))

# postprocessing

FINAL_DIR_NAME = f"{GUI_FILENAME}_{platform}_{VERSION}{'-debug' if args.debug_gui else ''}"

# remove duplicate files of launcher
if (not platform.startswith("macos")) or args.debug_gui:
    print("Merging...", end="", flush=True)
    shutil.move(
        join_path(compile_path, f"{CLI_FILENAME}.dist",
                f"{CLI_FILENAME}{exec_suffix}"),
        join_path(compile_path, f"{GUI_FILENAME}.dist"))
    shutil.rmtree(join_path(compile_path, f"{CLI_FILENAME}.dist"))
    print("Done.")
    # rename executable file and folder
    print("Renaming executable files...", end="", flush=True)
    shutil.move(join_path(compile_path, f"{GUI_FILENAME}.dist"),
                join_path(compile_path, FINAL_DIR_NAME))
    print("Done.")

# shared postprocessing
# copy configuration file
# package codes with zip(if applied).
if apply_zip:
    zip_fname = join_path(compile_path, f"{SOFTWARE_NAME}.zip")
    print(f"Zipping files to {zip_fname} ...", end="", flush=True)
    with zipfile.ZipFile(zip_fname, mode='w') as zipfile_op:
        file_to_zip(join_path(compile_path, FINAL_DIR_NAME), zipfile_op)
    print("Done.")

logger.info(f"Package script finished. Total time cost {(time.time()-t0):.2f}s.")
