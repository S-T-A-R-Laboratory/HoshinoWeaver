"""Pure functions for coordinate conversions and rotation matrix construction."""
import math
from enum import Enum
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class CoordSystem(Enum):
    ALTAZ = "altaz"
    RADEC = "radec"
    CAMERA = "camera"


def az_alt_to_vector(az_deg: float, alt_deg: float) -> NDArray[np.float64]:
    az = math.radians(az_deg)
    alt = math.radians(alt_deg)
    x = math.cos(alt) * math.cos(az)
    y = math.cos(alt) * math.sin(az)
    z = math.sin(alt)
    return np.array([x, y, z])


def vector_to_az_alt(v: Tuple[float, float, float]) -> Tuple[float, float]:
    x, y, z = v
    r = math.sqrt(x * x + y * y + z * z)
    x /= r
    y /= r
    z /= r
    alt = math.asin(z)
    az = math.atan2(y, x)
    return az, alt


def make_cross_matrix(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def build_rotation_matrix(lon_deg: float, lat_deg: float,
                          roll_deg: float) -> NDArray[np.float64]:
    """Build a world-to-camera rotation matrix from spherical pointing + roll.

    Works for both AltAz (lon=az, lat=alt) and RA/Dec (lon=ra, lat=dec)
    because both share the same spherical geometry with z-axis as pole.

    Returns:
        3x3 rotation matrix (world → camera, OpenCV convention: X-right, Y-down, Z-forward).
    """
    ra = np.deg2rad(lon_deg)
    dec = np.deg2rad(lat_deg)
    roll = np.deg2rad(roll_deg)

    forward = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    north_pole = np.array([0.0, 0.0, 1.0])

    if abs(lat_deg) > 89.99:
        up_raw = np.array([np.cos(ra), np.sin(ra), 0.0])
        if lat_deg > 0:
            up_raw = -up_raw
    else:
        up_raw = north_pole - np.dot(north_pole, forward) * forward

    up = up_raw / np.linalg.norm(up_raw)
    down = -up

    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    R_no_roll = np.array([right, down, forward])

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    R_roll = np.array([
        [cos_roll, sin_roll, 0],
        [-sin_roll, cos_roll, 0],
        [0, 0, 1],
    ])

    return R_roll @ R_no_roll
