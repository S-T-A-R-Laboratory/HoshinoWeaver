"""
EXIF 元数据读写工具

使用 piexif 和 tifffile 来处理 JPEG 和 TIFF 格式文件的 EXIF 信息和 ICC 色彩配置文件。

主要功能：
- 使用 piexif 读取完整的 EXIF 元数据信息
- 支持 JPEG 和 TIFF 格式文件
- 处理 ICC 色彩配置文件
- 数字标签转换为明文标签（人类可读）
- JPEG 文件使用 piexif 直接写入
- TIFF 文件使用 tifffile 作为候补方案写入
- 提供 ExifData 中间数据结构，支持结构化打印和跨格式转存

使用示例：
    >>> from norma.exif import read_exif, write_exif, ExifData
    >>> 
    >>> # 读取 EXIF 信息
    >>> exif_data = read_exif("image.jpg")
    >>> exif_data.print_summary()
    >>> 
    >>> # 修改 EXIF 信息
    >>> exif_data.set_tag_value("0th", 271, "Canon")  # Make
    >>> 
    >>> # 写入 EXIF 信息
    >>> write_exif("output.jpg", exif_data)
    >>> 
    >>> # 移植 EXIF 信息
    >>> from norma.exif import transplant_exif
    >>> transplant_exif("source.jpg", "target.jpg", "output.jpg")
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ExifTags
from PIL.Image import Image as PILImage
import piexif
import tifffile

COMMON_EXIF_TAGS: dict[str, tuple] = {
    "ImageWidth": ("0th", 256),
    "ImageLength": ("0th", 257),
    "BitsPerSample": ("0th", 258),
    'Make': ("0th", 271),
    'Model': ("0th", 272),
    'XResolution': ("0th", 282),
    'YResolution': ("0th", 283),
    'Software': ("0th", 305),
    'InterColorProfile': ("0th", 34675),
    'ExposureTime': ('Exif', 33434),
    'FNumber': ('Exif', 33437),
    'ISOSpeedRatings': ('Exif', 34855),
    'ExifVersion': ('Exif', 36864),
    'DateTimeOriginal': ('Exif', 36867),
    'DateTimeDigitized': ('Exif', 36868),
    'OffsetTime': ('Exif', 36880),
    'FocalLength': ('Exif', 37386),
    'FocalPlaneXResolution': ('Exif', 41486),
    'FocalPlaneYResolution': ('Exif', 41487),
    'FocalPlaneResolutionUnit': ('Exif', 41488),
    'LensSpecification': ('Exif', 42034),
    'LensModel': ('Exif', 42036),
}


@dataclass
class ExifData:
    """
    EXIF 元数据中间数据结构
    
    用于存储、操作和转换 EXIF 信息，支持结构化打印和跨格式转存。
    """
    # IFD 字典：piexif 的标准结构
    # 包含 "0th", "Exif", "GPS", "Interop", "1st" 等键
    exif_dict: Dict[str, Dict[int, Any]] = field(default_factory=dict)

    # ICC 色彩配置文件（字节数据）
    icc_profile: Optional[bytes] = None

    # 文件格式信息
    file_format: Optional[str] = None  # 'JPEG' 或 'TIFF'

    def __post_init__(self):
        """初始化时确保 exif_dict 有正确的结构"""
        if not self.exif_dict:
            self.exif_dict = {
                "0th": {},
                "Exif": {},
                "GPS": {},
                "Interop": {},
                "1st": {},
            }

    def get_tag_name(self, ifd_name: str, tag_id: int) -> Optional[str]:
        """
        获取标签的明文名称
        
        Args:
            ifd_name: IFD 名称（"0th", "Exif", "GPS" 等）
            tag_id: 标签 ID
            
        Returns:
            标签的明文名称，如果不存在则返回 None
        """
        try:
            if ifd_name in piexif.TAGS:
                if tag_id in piexif.TAGS[ifd_name]:
                    return piexif.TAGS[ifd_name][tag_id]["name"]
        except (KeyError, TypeError):
            pass

        # 尝试使用 Pillow 的 ExifTags
        try:
            if tag_id in ExifTags.TAGS:
                return ExifTags.TAGS[tag_id]
        except (KeyError, TypeError):
            pass

        return None

    def get_tag_value(self, ifd_name: str, tag_id: int) -> Any:
        """
        获取标签值
        
        Args:
            ifd_name: IFD 名称
            tag_id: 标签 ID
            
        Returns:
            标签值，如果不存在则返回 None
        """
        if ifd_name in self.exif_dict and tag_id in self.exif_dict[ifd_name]:
            return self.exif_dict[ifd_name][tag_id]
        return None

    def set_tag_value(self, ifd_name: str, tag_id: int, value: Any) -> None:
        """
        设置标签值
        
        Args:
            ifd_name: IFD 名称
            tag_id: 标签 ID
            value: 标签值
        """
        if ifd_name not in self.exif_dict:
            self.exif_dict[ifd_name] = {}
        self.exif_dict[ifd_name][tag_id] = value

    def to_human_readable(self) -> Dict[str, Dict[str, Any]]:
        """
        转换为人类可读的字典格式（标签 ID 转换为明文名称）
        
        Returns:
            包含明文标签名称的字典
        """
        result = {}
        for ifd_name, tags in self.exif_dict.items():
            result[ifd_name] = {}
            if not isinstance(tags, dict):
                result[ifd_name] = tags
                continue
            for tag_id, value in tags.items():
                tag_name = self.get_tag_name(ifd_name, tag_id)
                if tag_name:
                    result[ifd_name][tag_name] = value
                else:
                    result[ifd_name][f"Unknown_{tag_id}"] = value
        return result

    def print_summary(self) -> None:
        """打印 EXIF 信息摘要（人类可读格式）"""
        readable = self.to_human_readable()
        print(f"文件格式: {self.file_format}")
        print(f"ICC 配置文件: {'存在' if self.icc_profile else '不存在'}")
        print("\nEXIF 信息:")
        for ifd_name, tags in readable.items():
            if tags:
                print(f"\n[{ifd_name}]")
                for tag_name, value in tags.items():
                    # 处理字节数据
                    if isinstance(value, bytes):
                        try:
                            value_str = value.decode('utf-8', errors='ignore')
                            if len(value_str) > 50:
                                value_str = value_str[:50] + "..."
                        except:
                            value_str = f"<bytes: {len(value)} bytes>"
                    elif isinstance(value, tuple) and len(value) > 0:
                        # 处理有理数元组
                        if len(value) == 2:
                            value_str = f"{value[0]}/{value[1]}"
                        else:
                            value_str = str(value)
                    else:
                        value_str = str(value)
                    print(f"  {tag_name}: {value_str}")

    def copy(self) -> 'ExifData':
        """创建当前对象的深拷贝"""
        import copy
        new_exif_dict = copy.deepcopy(self.exif_dict)
        new_icc = self.icc_profile.copy() if self.icc_profile else None
        return ExifData(exif_dict=new_exif_dict,
                        icc_profile=new_icc,
                        file_format=self.file_format)

        
    @classmethod
    def initialize_empty(cls, file_format: Optional[str] = None):
        """创建一个空的 ExifData 对象"""
        return cls(exif_dict={
            "0th": {},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
        },
                   icc_profile=None,
                   file_format=file_format)


def read_exif(file_path: Union[str, Path]) -> ExifData:
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

    # 确定文件格式
    suffix = file_path.suffix.lower()
    if suffix in ['.jpg', '.jpeg']:
        file_format = 'JPEG'
    elif suffix in ['.tif', '.tiff']:
        file_format = 'TIFF'
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

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

    return ExifData(exif_dict=exif_dict,
                    icc_profile=icc_profile,
                    file_format=file_format)


def write_exif(file_path: Union[str, Path],
               exif_data: ExifData,
               preserve_image: bool = True) -> None:
    """
    将 EXIF 信息和 ICC 色彩配置文件写入文件
    
    JPEG 文件使用 piexif 直接写入，TIFF 文件使用 tifffile 作为候补方案。
    
    Args:
        file_path: 目标文件路径
        exif_data: ExifData 对象，包含要写入的 EXIF 信息
        preserve_image: 是否保留原始图像数据（默认 True）
        
    Raises:
        ValueError: 不支持的文件格式或 piexif 无法处理 TIFF
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ['.jpg', '.jpeg']:
        _write_exif_jpeg(file_path, exif_data, preserve_image)
    elif suffix in ['.tif', '.tiff']:
        _write_exif_tiff(file_path, exif_data, preserve_image)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def _write_exif_jpeg(file_path: Path, exif_data: ExifData,
                     preserve_image: bool) -> None:
    """
    写入 JPEG 文件的 EXIF 信息（使用 piexif）
    
    Args:
        file_path: 目标文件路径
        exif_data: ExifData 对象
        preserve_image: 是否保留原始图像数据
    """
    # 准备 EXIF 数据
    exif_bytes = None
    try:
        # 验证并转储 EXIF 数据
        exif_bytes = piexif.dump(exif_data.exif_dict)
    except Exception as e:
        raise ValueError(f"无法生成 EXIF 数据: {e}")

    if preserve_image:
        # 保留原始图像数据
        try:
            # 读取原始图像
            with Image.open(file_path) as img:
                # 转换为 RGB（如果需要）
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # 准备保存参数
                save_kwargs = {'format': 'JPEG', 'quality': 95}
                if exif_data.icc_profile:
                    save_kwargs['icc_profile'] = exif_data.icc_profile

                # 使用 piexif.insert 插入 EXIF
                img_bytes = io.BytesIO()
                img.save(img_bytes, **save_kwargs)
                img_bytes.seek(0)

                # 插入 EXIF 数据
                exif_inserted = piexif.insert(exif_bytes, img_bytes.getvalue())

                # 写入文件
                with open(file_path, 'wb') as f:
                    f.write(exif_inserted)
        except Exception as e:
            raise IOError(f"写入 JPEG 文件时出错: {e}")
    else:
        # 直接插入 EXIF（会覆盖原文件）
        try:
            with open(file_path, 'rb') as f:
                img_data = f.read()
            exif_inserted = piexif.insert(exif_bytes, img_data)
            with open(file_path, 'wb') as f:
                f.write(exif_inserted)
        except Exception as e:
            raise IOError(f"写入 JPEG 文件时出错: {e}")


def _write_exif_tiff(file_path: Path, exif_data: ExifData,
                     preserve_image: bool) -> None:
    """
    写入 TIFF 文件的 EXIF 信息（使用 tifffile 作为候补方案）
    
    注意：piexif 对 TIFF 的支持有限，因此使用 tifffile 来处理。
    
    Args:
        file_path: 目标文件路径
        exif_data: ExifData 对象
        preserve_image: 是否保留原始图像数据
    """
    if preserve_image:
        # 读取原始图像数据
        try:
            # 使用 tifffile 读取图像
            img_data = tifffile.imread(str(file_path))

            # 准备元数据
            metadata = {}
            if exif_data.icc_profile:
                metadata['icc_profile'] = exif_data.icc_profile

            # 尝试将 EXIF 数据转换为 tifffile 可用的格式
            # tifffile 使用不同的元数据结构
            if exif_data.exif_dict:
                # 将 piexif 格式转换为 tifffile 的 extratags
                extratags = _convert_exif_to_tifftags(exif_data.exif_dict)
                if extratags:
                    metadata['extratags'] = extratags

            # 写入文件
            tifffile.imwrite(str(file_path), img_data, **metadata)
        except Exception as e:
            # 如果 tifffile 方法失败，尝试使用 Pillow
            try:
                with Image.open(file_path) as img:
                    # 保存图像和 ICC 配置文件
                    save_kwargs = {'format': 'TIFF', 'save_all': True}
                    if exif_data.icc_profile:
                        save_kwargs['icc_profile'] = exif_data.icc_profile

                    # Pillow 对 TIFF 的 EXIF 支持有限，主要保存 ICC
                    img.save(file_path, **save_kwargs)
            except Exception as e2:
                raise IOError(f"写入 TIFF 文件时出错: {e2}")
    else:
        raise ValueError("TIFF 文件写入时必须保留图像数据")


def _convert_exif_to_tifftags(exif_dict: Dict[str, Dict[int, Any]]) -> list:
    """
    将 piexif 格式的 EXIF 字典转换为 tifffile 的 extratags 格式
    
    Args:
        exif_dict: piexif 格式的 EXIF 字典
        
    Returns:
        tifffile extratags 列表
    """
    extratags = []

    # tifffile 使用 (code, type, count, value, writeonce) 格式
    # 类型码映射：'B'=byte, 's'=string, 'H'=short, 'I'=long, '2I'=rational
    # 简化处理：只转换部分关键标签
    # 完整的转换需要更复杂的类型映射
    for ifd_name, tags in exif_dict.items():
        if ifd_name == "0th" and tags:
            # 处理一些基本标签
            for tag_id, value in tags.items():
                try:
                    # 根据标签类型确定 tifffile 的类型码
                    if isinstance(value, str):
                        # 字符串类型
                        extratags.append(
                            (tag_id, 's', len(value), value, False))
                    elif isinstance(value, int):
                        # 整数类型
                        if value < 65536:
                            extratags.append((tag_id, 'H', 1, value, False))
                        else:
                            extratags.append((tag_id, 'I', 1, value, False))
                    elif isinstance(value, tuple) and len(value) == 2:
                        # 有理数类型
                        extratags.append((tag_id, '2I', 1, value, False))
                    elif isinstance(value, bytes):
                        # 字节类型
                        extratags.append(
                            (tag_id, 'B', len(value), value, False))
                except Exception:
                    # 跳过无法转换的标签
                    continue

    return extratags


def set_common_tag(exif_data: ExifData,
                   tag_name: str,
                   value: Any,
                   ifd_name: str = "0th") -> None:
    """
    设置常用 EXIF 标签的便捷方法
    
    Args:
        exif_data: ExifData 对象
        tag_name: 标签名称（如 "Make", "Model" 等）
        value: 标签值
        ifd_name: IFD 名称（"0th", "Exif", "GPS" 等）
    """
    # 尝试从 piexif.TAGS 中查找标签 ID
    tag_id = None
    if ifd_name in piexif.TAGS:
        for tid, tag_info in piexif.TAGS[ifd_name].items():
            if tag_info.get("name") == tag_name:
                tag_id = tid
                break

    if tag_id is None:
        raise ValueError(f"未找到标签: {tag_name} 在 IFD: {ifd_name}")

    exif_data.set_tag_value(ifd_name, tag_id, value)
