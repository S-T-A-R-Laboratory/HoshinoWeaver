"""Application services. Importing this module starts no external resources."""
from .onnx import ONNX, OnnxService, onnx_settings
from .segmentation import SEGMENTATION, SkySegmentationService, segmentation_settings
from .photoshop import PHOTOSHOP, PhotoshopService, photoshop_settings
from .registry import SerialService, ServiceKey, ServiceRegistry, ServiceScope


def create_service_registry() -> ServiceRegistry:
    registry = ServiceRegistry()
    registry.register(PHOTOSHOP, PhotoshopService, settings=photoshop_settings)
    registry.register(ONNX, OnnxService, settings=onnx_settings)
    registry.register(SEGMENTATION, SkySegmentationService, settings=segmentation_settings)
    return registry


# Only factory registrations are process-wide; each root graph owns its services.
DEFAULT_SERVICES = create_service_registry()
