"""Immutable value types for the norma package."""
import abc
import dataclasses
from functools import cached_property
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from hoshicore._custom_op import camera_model_remap as custom_camera_model_remap

from .geometry import CoordSystem, build_rotation_matrix
from .projection import (make_intrinsic_matrix, project_vectors,
                         unproject_pixels)
from .sky_model import altaz_to_radec, compute_parallactic_angle


class BaseCameraModel(abc.ABC):
    """Abstract base for all camera models. Requires unproject()."""

    @abc.abstractmethod
    def unproject(self, pts: "NDArray[np.float64]") -> "NDArray[np.float64]":
        """Map pixel coordinates (n, 2) to unit vectors (n, 3)."""
        ...


@dataclasses.dataclass(frozen=True)
class Distortion:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def is_zero(self) -> bool:
        return (self.k1 == 0 and self.k2 == 0 and self.k3 == 0
                and self.p1 == 0 and self.p2 == 0)

    def to_cv2(self) -> NDArray[np.float64]:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=np.float64)

    @classmethod
    def from_cv2(cls, arr) -> "Distortion":
        if len(arr) == 5:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3], k3=arr[4])
        elif len(arr) == 4:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3])
        elif len(arr) == 2:
            return cls(k1=arr[0], k2=arr[1])
        else:
            raise ValueError(f"Unexpected distortion array length: {len(arr)}")


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int

    @cached_property
    def K(self) -> NDArray[np.float64]:
        return make_intrinsic_matrix(self.focal_length_mm, self.sensor_width_mm,
                                     self.sensor_height_mm, self.image_width_px,
                                     self.image_height_px)

    def with_focal_length(self, focal_length_mm: float) -> "Intrinsics":
        return dataclasses.replace(self, focal_length_mm=focal_length_mm)


@dataclasses.dataclass(frozen=True)
class Pointing:
    lon_deg: float
    lat_deg: float
    roll_deg: float
    system: CoordSystem

    @cached_property
    def R(self) -> NDArray[np.float64]:
        return build_rotation_matrix(self.lon_deg, self.lat_deg, self.roll_deg)


@dataclasses.dataclass(frozen=True)
class CameraModel(BaseCameraModel):
    intrinsics: Intrinsics
    distortion: Distortion = dataclasses.field(default_factory=Distortion)
    pointing: Optional[Pointing] = None

    @cached_property
    def K(self) -> NDArray[np.float64]:
        return self.intrinsics.K

    @cached_property
    def R(self) -> Optional[NDArray[np.float64]]:
        return self.pointing.R if self.pointing else None

    @property
    def dist_coeffs(self) -> Optional[NDArray[np.float64]]:
        if self.distortion.is_zero:
            return None
        return self.distortion.to_cv2()

    def unproject(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return unproject_pixels(pts, self.K, self.dist_coeffs, self.R)

    def project(self, vecs: NDArray[np.float64]) -> NDArray[np.float64]:
        return project_vectors(vecs, self.K, self.dist_coeffs, self.R)

    def with_intrinsics(self, intrinsics: Intrinsics) -> "CameraModel":
        return dataclasses.replace(self, intrinsics=intrinsics)

    def with_distortion(self, distortion: Distortion) -> "CameraModel":
        return dataclasses.replace(self, distortion=distortion)

    def with_pointing(self, pointing: Pointing) -> "CameraModel":
        return dataclasses.replace(self, pointing=pointing)

    def with_focal_length(self, focal_length_mm: float) -> "CameraModel":
        return self.with_intrinsics(
            self.intrinsics.with_focal_length(focal_length_mm))

    def _rotation_dst_to_src(
            self, camera: "CameraModel") -> NDArray[np.float32]:
        src_rotation = np.eye(3, dtype=np.float32) if camera.R is None else np.asarray(
            camera.R, dtype=np.float32)
        dst_rotation = np.eye(3, dtype=np.float32) if self.R is None else np.asarray(
            self.R, dtype=np.float32)
        return src_rotation @ dst_rotation.T

    def _project_image_from_camera_custom_fused(
            self,
            camera: "CameraModel",
            img: NDArray[np.uint8],
            output_size: tuple[int, int],
            roi=None,
            interpolation=cv2.INTER_LINEAR) -> NDArray[np.uint8] | None:
        if roi is not None or interpolation != cv2.INTER_LINEAR:
            return None

        target_width, target_height = output_size
        return custom_camera_model_remap(
            image=img,
            out_height=target_height,
            out_width=target_width,
            fx_src=float(camera.K[0, 0]),
            fy_src=float(camera.K[1, 1]),
            cx_src=float(camera.K[0, 2]),
            cy_src=float(camera.K[1, 2]),
            fx_dst=float(self.K[0, 0]),
            fy_dst=float(self.K[1, 1]),
            cx_dst=float(self.K[0, 2]),
            cy_dst=float(self.K[1, 2]),
            rotation_dst_to_src=self._rotation_dst_to_src(camera),
            src_dist_coeffs=camera.dist_coeffs,
            dst_dist_coeffs=self.dist_coeffs,
        )

    @classmethod
    def from_view(cls, view: "View", mode: str = "auto") -> "CameraModel":
        if mode == "auto":
            if view.az_deg is not None and view.alt_deg is not None and view.world_roll_deg is not None:
                mode = CoordSystem.ALTAZ.value
            elif view.ra_deg is not None and view.dec_deg is not None and view.sky_roll_deg is not None:
                mode = CoordSystem.RADEC.value
            else:
                mode = CoordSystem.CAMERA.value

        intrinsics = Intrinsics(
            focal_length_mm=view.focal_length,
            sensor_width_mm=view.sensor_width_mm,
            sensor_height_mm=view.sensor_height_mm,
            image_width_px=view.img_width,
            image_height_px=view.img_height,
        )

        pointing = None
        if mode == CoordSystem.ALTAZ.value:
            pointing = Pointing(lon_deg=view.az_deg, lat_deg=view.alt_deg,
                                roll_deg=view.world_roll_deg,
                                system=CoordSystem.ALTAZ)
        elif mode == CoordSystem.RADEC.value:
            pointing = Pointing(lon_deg=view.ra_deg, lat_deg=view.dec_deg,
                                roll_deg=view.sky_roll_deg,
                                system=CoordSystem.RADEC)

        return cls(intrinsics=intrinsics, pointing=pointing)

    build_from_view = from_view

    def project_image_from_camera(self, camera: "CameraModel",
                                  img: NDArray[np.uint8],
                                  output_size: tuple[int, int],
                                  roi=None,
                                  interpolation=cv2.INTER_LINEAR):
        """Projects an image from `camera` into this camera's frame via remap."""
        target_width, target_height = output_size
        fused = self._project_image_from_camera_custom_fused(
            camera,
            img,
            output_size,
            roi=roi,
            interpolation=interpolation,
        )
        if fused is not None:
            return fused

        u_dst = np.arange(target_width, dtype=np.float32)
        v_dst = np.arange(target_height, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_dst, v_dst)
        dst_pixels = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)

        world_vecs = self.unproject(dst_pixels)
        src_pixels = camera.project(world_vecs)

        map_x = src_pixels[:, 0].reshape(target_height, target_width)
        map_y = src_pixels[:, 1].reshape(target_height, target_width)

        if roi is not None:
            x1, y1, x2, y2 = roi
            img_use = img[y1:y2, x1:x2]
            map_x = map_x - x1
            map_y = map_y - y1
        else:
            img_use = img

        return cv2.remap(img_use,
                         map_x.astype(np.float32),
                         map_y.astype(np.float32),
                         interpolation=interpolation,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0 if len(img_use.shape) == 2 else (0, 0, 0))


@dataclasses.dataclass
class View:
    """Complete view description."""
    focal_length: float
    sensor_width_mm: float
    sensor_height_mm: float
    img_width: int
    img_height: int
    az_deg: Optional[float] = None
    alt_deg: Optional[float] = None
    world_roll_deg: Optional[float] = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    sky_roll_deg: Optional[float] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    jd: Optional[float] = None

    def altaz_to_radec(self):
        if self.az_deg is None or self.alt_deg is None or self.world_roll_deg is None:
            raise ValueError(
                "Azimuth, Altitude and Roll deg must be set to convert to RA/Dec."
            )
        if self.latitude_deg is None or self.longitude_deg is None or self.jd is None:
            raise ValueError(
                "Latitude, Longitude, and Julian Day must be set to convert to RA/Dec."
            )
        ra, dec = altaz_to_radec(azimuth_deg=self.az_deg,
                                 elevation_deg=self.alt_deg,
                                 latitude_deg=self.latitude_deg,
                                 longitude_deg=self.longitude_deg,
                                 jd=self.jd)
        parallactic_angle = compute_parallactic_angle(self.az_deg, self.alt_deg,
                                                     self.latitude_deg)
        self.ra_deg = ra
        self.dec_deg = dec
        self.sky_roll_deg = (self.world_roll_deg + parallactic_angle) % 360.0


class FlatCameraModel(BaseCameraModel):
    """Camera model using flat projection (no intrinsics or distortion).

    Uses image center as principal point and max(h, w) as focal length.
    Suitable for the 2D homography alignment path when no camera metadata
    is available.
    """

    def __init__(self, img_shape: tuple):
        h, w = img_shape[:2]
        self._f = float(max(h, w))
        self._cx, self._cy = w / 2.0, h / 2.0

    def unproject(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        x = (pts[:, 0] - self._cx) / self._f
        y = (pts[:, 1] - self._cy) / self._f
        z = np.ones(len(pts))
        vecs = np.stack([x, y, z], axis=1)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs
