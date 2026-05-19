"""Two-tier caching for star detection and geometry computation.

StarDetectionCache: pixel-level, camera-independent (one per image).
GeometryView: depends on (StarDetectionCache, BaseCameraModel).
"""
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .detection import DetectedStars, detect_star_points
from .matching import extract_point_features
from .types import BaseCameraModel, FlatCameraModel


class StarDetectionCache:
    """Pixel-level cache: independent of camera model, computed once per image.

    Accepts a pre-converted float64 grayscale array in [0, 1] range.
    Callers are responsible for dtype normalisation (e.g. via _to_gray_f64).
    """

    def __init__(self,
                 image_gray: NDArray[np.float64],
                 mask: Optional[np.ndarray] = None):
        self._image_gray = image_gray
        self.mask = mask
        self.img_shape = image_gray.shape

    @cached_property
    def detected_stars(self) -> DetectedStars:
        return detect_star_points(self._image_gray, self.mask)

    @property
    def positions(self) -> NDArray[np.float64]:
        return self.detected_stars.positions

    @property
    def volumes(self) -> NDArray[np.float64]:
        return self.detected_stars.volumes


class GeometryView:
    """Geometry-level cache: depends on (StarDetectionCache, BaseCameraModel).

    Recomputed when camera model changes (e.g. after focal length refinement).
    Accepts any BaseCameraModel subclass, including CameraModel and FlatCameraModel.
    """

    def __init__(self, detection: StarDetectionCache, camera: BaseCameraModel):
        self._detection = detection
        self._camera = camera

    @property
    def camera(self) -> BaseCameraModel:
        return self._camera

    @property
    def positions(self) -> NDArray[np.float64]:
        return self._detection.positions

    @property
    def volumes(self) -> NDArray[np.float64]:
        return self._detection.volumes

    @cached_property
    def unit_vectors(self) -> NDArray[np.float64]:
        return self._camera.unproject(self._detection.positions)

    @cached_property
    def features(self) -> NDArray[np.float64]:
        return extract_point_features(self.unit_vectors,
                                      self._detection.volumes)

    def with_camera(self, camera: BaseCameraModel) -> "GeometryView":
        """Create a new GeometryView with a different camera (reuses detection)."""
        return GeometryView(self._detection, camera)

    @classmethod
    def from_flat_projection(cls,
                             detection: StarDetectionCache) -> "GeometryView":
        """Create a GeometryView using flat projection (2D homography path).

        Uses image center as principal point and max(h, w) as focal length.
        No CameraModel required.
        """
        return cls(detection, FlatCameraModel(detection.img_shape))
