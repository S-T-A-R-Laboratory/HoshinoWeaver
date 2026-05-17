"""Simple pinhole camera model (no distortion for M1)."""
import dataclasses
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from .types import View
from .geometry import CoordSystem

COORD_SYSTEM_ALTAZ = CoordSystem.ALTAZ.value
COORD_SYSTEM_RADEC = CoordSystem.RADEC.value
COORD_SYSTEM_RELATIVE = "relative"
from .geometry import build_rotation_matrix
from .projection import (make_intrinsic_matrix, project_vectors,
                         unproject_pixels, undistort_points)


@dataclasses.dataclass
class DistorationParam:
    """畸变参数。
    包含径向畸变参数 k1, k2, k3 和切向畸变参数 p1, p2。
    """
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def to_numpy_array(self) -> NDArray[np.float64]:
        """Convert to numpy array for OpenCV."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=np.float64)

    @classmethod
    def from_numpy_array(cls, arr: Union[NDArray[np.float64], list, tuple]):
        """Create DistorationParam from numpy array or list/tuple."""
        if isinstance(arr, (list, tuple)):
            arr = np.array(arr, dtype=np.float64)
        if len(arr) == 5:
            return cls(k1=arr[0], k2=arr[1], p1=arr[2], p2=arr[3], k3=arr[4])
        elif len(arr) == 2:
            return cls(k1=arr[0], k2=arr[1])
        elif len(arr) == 1:
            return cls(k1=arr[0])
        else:
            raise ValueError("Invalid array length for DistorationParam.")


@dataclasses.dataclass
class CameraModel(object):
    """（针孔）相机模型。
    经过相机模型，可以将世界坐标系下的坐标转换为像素坐标系下的坐标，
    或者将像素坐标系下的坐标转换为世界坐标系下的坐标。

    Args:
        focal_length (float): 
        sensor_width (float): 
        sensor_height (float): 
        image_width (int): 
        image_height (int): 
        az_deg (Optional[float]): None
        alt_deg (Optional[float]): None
        roll_deg (Optional[float]): None
        coord_system (str) COORD_SYSTEM_ALTAZ
        distorted_level (int): 需要畸变优化的等级。0代表不启用畸变优化；1代表仅优化k1, 2代表优化k1, k2; 
            3代表启用全部优化(k1, k2, k3, p1, p2). Defaults to 0.
        distorted_parameters (Optional): [IntrinsicParam] = None
        object (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    focal_length: float
    sensor_width: float
    sensor_height: float
    image_width: int
    image_height: int
    az_deg: Optional[float] = None
    alt_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    coord_system: str = COORD_SYSTEM_ALTAZ
    distorted_level: int = 0
    distorted_parameters: DistorationParam = dataclasses.field(
        default=DistorationParam())

    def __post_init__(self):
        self._theoretical_focal_length = self.focal_length
        self._extrinsic_param = None
        if self.az_deg is not None and self.alt_deg is not None and self.roll_deg is not None:
            self.update_rotaion_matrix()
        self.update_intrinsic_param()

    @property
    def theoretical_focal_length(self) -> float:
        """Theoretical focal length without distortion."""
        return self._theoretical_focal_length

    @property
    def extrinsic_param(self) -> NDArray[np.float64]:
        """ Extrinsic parameters (rotation matrix), a 3x3 numpy array."""
        return self._extrinsic_param

    @property
    def intrinsic_param(self) -> NDArray[np.float64]:
        """ Intrinsic parameters, a 3x3 numpy array."""
        return self._intrinsic_param

    @property
    def distortion_parameters(self):
        return self.distorted_parameters.to_numpy_array()

    def update_focal_length(self, focal_length: float):
        """更新焦距信息，同步更新内参。

        Args:
            focal_length (float): 焦距(mm)。
        """
        self.focal_length = focal_length
        self.update_intrinsic_param()

    def update_distortion_parameters(self,
                                     distorted_parameters: DistorationParam):
        """更新畸变参数。

        Args:
            distorted_parameters (DistorationParam): 畸变参数。
        """
        self.distorted_parameters = distorted_parameters

    def project_unit_vectors(self, v: NDArray[np.float64]):
        """Project a unit direction vector in world frame to pixel coordinates, applying distortion.
        v: shape (n, 3), n is the number of unit vectors.
        """
        dist = self.distorted_parameters.to_numpy_array() \
            if self.distorted_level != 0 and self.distorted_parameters is not None \
            else None
        return project_vectors(v, self.intrinsic_param, dist,
                               self.extrinsic_param)

    def undistort_points(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Undistort pixel coordinates.
        u: shape (n, 2), n is the number of pixel coordinates.
        """
        dist = self.distorted_parameters.to_numpy_array() \
            if self.distorted_level != 0 and self.distorted_parameters is not None \
            else None
        return undistort_points(u, self.intrinsic_param, dist)

    def get_unified_pts_coordinates(
            self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get normalized image coordinates from pixel coordinates on the image plane.
        pts: shape (n, 2), n is the number of pixel coordinates.
        """
        upts = np.array([[self.image_width / 2.0, self.image_height / 2.0]])
        return (pts - upts) / upts

    def unproject_pixel_to_unit_vector(
            self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unproject a pixel coordinate to a unit direction vector in world frame, removing distortion.
        u: shape (n, 2), n is the number of pixel coordinates.
        """
        dist = self.distorted_parameters.to_numpy_array() \
            if self.distorted_level != 0 and self.distorted_parameters is not None \
            else None
        return unproject_pixels(u, self.intrinsic_param, dist,
                                self.extrinsic_param)

    @classmethod
    def build_from_view(cls, view: View, mode="auto"):
        """
        Build camera model from sky view or world view.
        
        Args:
            view: View
            mode: "auto", "altaz", "radec", "relative"
        Returns:
            CameraModel
        Raises:
            ValueError: Invalid view type
        """
        assert mode in ["auto",COORD_SYSTEM_ALTAZ, COORD_SYSTEM_RADEC, COORD_SYSTEM_RELATIVE], \
            "Invalid mode for building camera model."
        if mode == "auto":
            if view.az_deg and view.alt_deg and view.world_roll_deg:
                mode = COORD_SYSTEM_ALTAZ
            elif view.ra_deg and view.dec_deg and view.sky_roll_deg:
                mode = COORD_SYSTEM_RADEC
            else:
                mode = COORD_SYSTEM_RELATIVE
        az_deg = None
        alt_deg = None
        roll_deg = None
        if mode == COORD_SYSTEM_ALTAZ:
            az_deg = view.az_deg
            alt_deg = view.alt_deg
            roll_deg = view.world_roll_deg
        elif mode == COORD_SYSTEM_RADEC:
            az_deg = view.ra_deg
            alt_deg = view.dec_deg
            roll_deg = view.sky_roll_deg

        return cls(focal_length=view.focal_length,
                   sensor_width=view.sensor_width_mm,
                   sensor_height=view.sensor_height_mm,
                   image_width=view.img_width,
                   image_height=view.img_height,
                   az_deg=az_deg,
                   alt_deg=alt_deg,
                   roll_deg=roll_deg,
                   coord_system=mode)

    def update_rotaion_matrix(self):
        """
        基于当前的转角，构建当前参考系下外参（的从0方向到当前方向的旋转矩阵）。

        返回:
            rotation_matrix: 3x3 旋转矩阵，世界坐标 -> 相机坐标
        """
        if self.az_deg is None or self.alt_deg is None or self.roll_deg is None:
            logger.warning(f"update_rotaion_matrix failed: az_deg, alt_deg, "
                           "roll_deg must be set to compute rotation matrix.")
            return
        self._extrinsic_param = build_rotation_matrix(
            self.az_deg, self.alt_deg, self.roll_deg)

    def update_intrinsic_param(self):
        self._intrinsic_param = make_intrinsic_matrix(
            self.focal_length, self.sensor_width, self.sensor_height,
            self.image_width, self.image_height)

    def generate_rectify_map(
        self, img_size: Tuple[int, int]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """生成矫正映射。
        NOTE: Unchecked.
        """
        if not self.distorted_level != 0 or self.distorted_parameters is None:
            raise ValueError("Camera model is not distorted.")
        rectify_map = cv2.initUndistortRectifyMap(
            self.intrinsic_param, self.distorted_parameters.to_numpy_array(),
            None, self.intrinsic_param, img_size, cv2.CV_32FC1)
        return rectify_map

    def _calculate_projection_matrix(self, camera: "CameraModel"):
        """计算从另一个相机模型投影到当前相机模型的投影矩阵。
        
        Args:
            camera (CameraModel): 另一个相机模型。
        
        Returns:
            NDArray[np.float64]: 3x3 投影矩阵。
        """
        K_src = camera.intrinsic_param
        K_dst = self.intrinsic_param
        R_src = camera.extrinsic_param
        R_dst = self.extrinsic_param
        project_mat = K_dst @ (R_dst @ R_src.T) @ np.linalg.inv(K_src)
        return project_mat

    def project_points_from_camera(self, camera: "CameraModel",
                                   points: NDArray[np.float64]):
        """从另一个相机模型投影平面点到当前相机模型的像素坐标系下。
        
        Args:
            camera (CameraModel): 另一个相机模型。
            points (NDArray[np.float64]): 形状为 (n, 2) 的像素坐标点数组。
        """
        if not camera.distorted_level != 0 and not self.distorted_level != 0:
            # 无畸变，从旋转矩阵和内参直接计算仿射变换矩阵
            return (self._calculate_projection_matrix(camera) @ points.T).T
        else:
            raise NotImplementedError(
                "Projecting points with distortion is not implemented.")

    def project_image_from_camera(self,
                                  camera: "CameraModel",
                                  img: NDArray[np.uint8],
                                  output_size: Tuple[int, int],
                                  roi: Optional[Tuple[int, int, int,
                                                      int]] = None,
                                  interpolation: int = cv2.INTER_LINEAR):
        """从另一个相机模型投影图像到当前相机模型的像素坐标系下。
        
        Args:
            camera (CameraModel): 另一个相机模型。
            img (NDArray[np.uint8]): 输入图像。
            output_size (Tuple[int, int]): 输出图像大小 (width, height)。
        """
        if not camera.distorted_level != 0 and not self.distorted_level != 0:
            target_width, target_height = output_size
            src_height, src_width = img.shape[:2]
            H = self._calculate_projection_matrix(camera)
            # 如果指定了 ROI，需要调整单应性矩阵
            if roi is not None:
                x1, y1, x2, y2 = roi
                # 将 ROI 局部坐标映射到目标图像坐标
                T_roi = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]],
                                 dtype=np.float64)
                H = H @ T_roi
                # 裁剪源图像
                image_to_use = img[y1:y2, x1:x2]
            else:
                image_to_use = img

            # 使用 cv2.warpPerspective 进行透视变换
            projected = cv2.warpPerspective(
                image_to_use,
                H.astype(np.float32), (target_width, target_height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0 if len(image_to_use.shape) == 2 else (0, 0, 0))
            return projected
        else:
            raise NotImplementedError(
                "Projecting image with distortion is not implemented.")

    def copy(self):
        return self.__class__(focal_length=self.focal_length,
                              sensor_width=self.sensor_width,
                              sensor_height=self.sensor_height,
                              image_width=self.image_width,
                              image_height=self.image_height,
                              az_deg=self.az_deg,
                              alt_deg=self.alt_deg,
                              roll_deg=self.roll_deg,
                              coord_system=self.coord_system,
                              distorted_level=self.distorted_level,
                              distorted_parameters=DistorationParam(
                                  k1=self.distorted_parameters.k1,
                                  k2=self.distorted_parameters.k2,
                                  p1=self.distorted_parameters.p1,
                                  p2=self.distorted_parameters.p2,
                                  k3=self.distorted_parameters.k3,
                              ))
