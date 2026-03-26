from .base import ParallelBaseOp, BaseOp
import fractions
from typing import Union
from pathlib import Path
import piexif
from PIL import Image
from loguru import logger


def read_exif(file_path: Union[str, Path]) -> dict:
    """
    读取文件的 EXIF 信息和 ICC 色彩配置文件
    
    使用 piexif 读取 EXIF 信息以确保获得完整的元数据。
    
    Args:
        file_path: 图像文件路径（JPEG 或 TIFF）
        
    Returns:
        ExifData 对象，包含 EXIF 信息和 ICC 配置文件
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取 EXIF 信息（使用 piexif）
    exif_dict = {}
    try:
        exif_dict = piexif.load(str(file_path))
    except Exception as e:
        # piexif 可能无法读取某些文件，返回空字典
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        }

    # 读取 ICC 色彩配置文件（使用 Pillow）
    icc_profile = None
    try:
        with Image.open(file_path) as img:
            if hasattr(img, 'info') and 'icc_profile' in img.info:
                icc_profile = img.info['icc_profile']
    except Exception:
        # 如果无法读取 ICC，继续执行
        pass

    return {"exif_dict": exif_dict, "icc_profile": icc_profile}


class ExifReadOp(ParallelBaseOp):
    INPUTS = {"fnames": {"type": "sequence", "description": "File names"}}
    OUTPUTS = {"result": {"type": "sequence", "description": "Exif sequence"}}
    PARALLEL_ARGS_LIST = ["fnames"]

    def _execute_single(self, single_arg_tuple: tuple, args: dict):
        fname: str = single_arg_tuple[0]
        exif = read_exif(fname)
        return exif


class ExifMergeOp(BaseOp):
    INPUTS = {
        "exifs": {
            "type": "sequence",
            "description": "Exif sequence"
        },
        "merge_method": {
            "type": "str",
            "description": "Merge method"
        }
    }
    OUTPUTS = {"result": {"type": "exif", "description": "Exif"}}

    def execute(self, args: dict):
        exifs = args["exifs"]
        merge_method = args["merge_method"]
        if merge_method == "sum":
            time_cumsum = 0
            exif_dict = None 
            for exif in exifs:
                time = exif.get("Exif.Photo.ExposureTime")
                if time is not None:
                    time_cumsum += fractions.Fraction(time)
                    if exif_dict is None:
                        exif_dict = exif
            exif_dict["Exif.Photo.ExposureTime"] = "/".join(
                    map(str, time_cumsum.as_integer_ratio()))
            logger.info(
                f"Calculated total exposure time = {exif_dict.get('Exif.Photo.ExposureTime')}."
            )
            return exif_dict
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")
