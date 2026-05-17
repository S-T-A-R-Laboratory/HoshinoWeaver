"""
和天球坐标系，以及世界坐标系和天球坐标系的变换相关的方法。
"""
from datetime import datetime
from typing import Union

import numpy as np


def compute_gmst(jd: float):
    """计算格林尼治平恒星时 (Greenwich Mean Sidereal Time)
    
    参数:
        jd: 儒略日
    
    返回:
        GMST (度)
    """
    # 从 J2000.0 起算的世纪数
    T = (jd - 2451545.0) / 36525.0

    # GMST 公式 (度)
    gmst_hours = 18.697374558 + 24.06570982441908 * (jd - 2451545.0) + \
                 0.000026 * T * T

    # 转换为度并归一化到 [0, 360)
    gmst_deg = (gmst_hours % 24.0) * 15.0

    return gmst_deg


def compute_julian_day(dt: datetime) -> float:
    """计算儒略日 (Julian Day)。简化公式，适用于 1900-2100 年。
    
    参数:
        dt: datetime 对象 (UTC)
    
    返回:
        儒略日 (浮点数)
    """
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 *
                                           (month + 1)) + day + b - 1524.5
    jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd


def compute_parallactic_angle(azimuth_deg, elevation_deg, latitude_deg):
    """计算视差角（parallactic angle）。
    
    视差角是地平坐标系和赤道坐标系之间的"上方"方向的夹角。
    当把地平坐标系的滚转角转换到赤道坐标系时，需要加上视差角。
    
    视差角公式: tan(q) = sin(az) / (tan(lat) * cos(alt) - sin(alt) * cos(az))
    
    Arguments:
        azimuth_deg: 方位角（度），从北顺时针
        elevation_deg: 高度角/仰角（度）
        latitude_deg: 观测点纬度（度），北纬为正
    
    返回:
        parallactic_angle_deg: 视差角（度）
    """
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(elevation_deg)
    lat = np.deg2rad(latitude_deg)
    denominator = np.tan(lat) * np.cos(alt) - np.sin(alt) * np.cos(az)
    parallactic_angle = np.arctan2(np.sin(az), denominator)

    return np.rad2deg(parallactic_angle)


def altaz_to_radec(azimuth_deg: float, elevation_deg: float,
                   latitude_deg: float, longitude_deg: float, jd: float):
    """将地平坐标 (方位角, 高度角) 转换为赤道坐标 (RA, Dec)
    
    参数:
        azimuth_deg: 方位角 (度)，从正北顺时针
        elevation_deg: 高度角/仰角 (度)
        latitude_deg: 观测点纬度 (度)，北纬为正
        longitude_deg: 观测点经度 (度)，东经为正
        jd: 儒略日
    
    返回:
        (ra_deg, dec_deg): 赤经和赤纬 (度)
    """
    # 转换为弧度
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(elevation_deg)
    lat = np.deg2rad(latitude_deg)

    # 计算赤纬
    sin_dec = np.sin(alt) * np.sin(lat) + np.cos(alt) * np.cos(lat) * np.cos(
        az)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    # 计算时角
    cos_ha = (np.sin(alt) - np.sin(lat) * np.sin(dec)) / (np.cos(lat) *
                                                          np.cos(dec))
    cos_ha = np.clip(cos_ha, -1.0, 1.0)

    sin_ha = -np.sin(az) * np.cos(alt) / np.cos(dec)

    ha = np.arctan2(sin_ha, cos_ha)

    # 计算当地恒星时 (LST)
    gmst_deg = compute_gmst(jd)
    lst_deg = gmst_deg + longitude_deg

    # 计算赤经: RA = LST - HA
    ra_rad = np.deg2rad(lst_deg) - ha

    # 转换为度并归一化
    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec)

    # 归一化 RA 到 [0, 360)
    ra_deg = ra_deg % 360.0

    return ra_deg, dec_deg
