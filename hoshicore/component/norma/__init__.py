"""Norma: star-point alignment for astrophotography."""
from .alignment import (AlignmentResult, match_star_pairs, optimize_alignment,
                        warp_image, warp_image_by_remap)
from .cache import GeometryView, StarDetectionCache
from .frame_align import (AlignmentError, align_frame_camera_model,
                          align_frame_homography, make_geometry,
                          try_build_camera)
from .geometry import CoordSystem
from .intrinsics_from_exif import intrinsics_from_exif
from .sky_model import (altaz_to_radec, compute_julian_day,
                        compute_parallactic_angle)
from .types import BaseCameraModel, CameraModel, Distortion, Intrinsics, Pointing, View, FlatCameraModel

COORD_SYSTEM_RADEC = CoordSystem.RADEC.value
COORD_SYSTEM_ALTAZ = CoordSystem.ALTAZ.value
