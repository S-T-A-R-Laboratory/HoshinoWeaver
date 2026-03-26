"""参数类定义。部分参数可能支持一些复杂的读写方法。
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

import numpy as np
from easydict import EasyDict
from loguru import logger

from ..component.utils import SUPPORT_COLOR_SPACE


@dataclass
class StackConfigArg(object):
    """叠加配置参数类。
    
    Args:
        fin_ratio (float): 渐入比例
    """
    fin_ratio: float = 0
    # 渐出比例
    fout_ratio: float = 0
    # resize参数（字符串）
    resize: Optional[str] = None
    # 整形叠加工作流
    int_weight: bool = True
    # 输出位数
    output_bits: Optional[int] = None
    # 地面掩模图像路径
    ground_mask_fname: Optional[str] = None
    # 应用在本次处理的滤镜列表(暂定)。
    filter_list: Optional[list[str]] = None
    # SigmaClipping时，接受的方差上界倍数。
    rej_high: float = 3.0
    # SigmaClipping时，接受的方差下界倍数。
    rej_low: float = 3.0
    # SigmaClipping时，最大迭代次数。
    max_iter: int = 5
    # SigmaClipping时，提前收敛的比率阈值。
    coverage_prec: float = 0.99


@dataclass
class ImgInfo(object):
    """储存图像的色彩配置文件和EXIF的图像数据结构。      
    """
    # 图像的EXIF信息字典
    exif: Union[EasyDict, dict] = None
    # 图像色彩配置文件
    colorprofile: Optional[bytes] = b""
    # EXIF 曝光时间key，常量。
    EXIF_EXPOSURE_TIME_KEY: str = "Exif.Photo.ExposureTime"
    # EXIF 软件名称key，常量。
    EXIF_SOFTWARE_NAME: str = "Exif.Image.Software"
    # 不支持的色彩空间名称
    UNSUPPORTED_COLORSPACE_NAME: str = "UnsupportedColorSpaceName"

    def exif_valid_chk(self):
        """
        检查 exif 的合法性。
        """
        assert isinstance(
            self.exif,
            (dict, EasyDict)), f"invalid exif object: got {type(self.exif)}."

    @property
    def color_profile_name(self) -> str:
        """
        Returns:
            str: 当前Info的实际色彩空间名称（如果可用）
        """
        if not self.colorprofile: return "None"
        color_profile = self.colorprofile.decode("latin-1", errors="ignore")
        for color_space in SUPPORT_COLOR_SPACE:
            if color_space in color_profile:
                return color_space
        logger.warning(
            "Unsupported color space name. For now only these color spaces"
            f" are supported: {SUPPORT_COLOR_SPACE}")
        return self.UNSUPPORTED_COLORSPACE_NAME

    def get_exposure_time(self) -> Fraction:
        """获取该图像信息的曝光时间。如果无合法值，则返回值为0的分数。

        Returns:
            Fraction: 曝光时间
        """
        if self.exif is not None:
            return Fraction(
                self.exif.get(self.EXIF_EXPOSURE_TIME_KEY, Fraction(0, 1)))
        return Fraction(0, 1)

    def set_exposure_time(self,
                          exposure_time: Fraction,
                          to_str: bool = True) -> None:
        """设置该图像信息的曝光时间。
        没有exif的场合，该方法会创建一个新的exif信息。

        Args:
            exposure_time (Fraction): 曝光时间 in Fraction
            to_str (bool): 填充进exif时是否转换为str。Default to True.
        """
        self.exif_valid_chk()
        if to_str:
            exposure_time_value = "/".join(
                map(str, exposure_time.as_integer_ratio()))
        else:
            exposure_time = exposure_time

        self.exif.update({self.EXIF_EXPOSURE_TIME_KEY: exposure_time_value})

    def set_software_name(self, name: str) -> None:
        self.exif_valid_chk()
        self.exif.update({self.EXIF_SOFTWARE_NAME: name})
