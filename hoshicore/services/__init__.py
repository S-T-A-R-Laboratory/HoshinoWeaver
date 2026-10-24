"""Application services. Importing this module starts no external resources."""
from .onnx import SEGMENTATION, OnnxService, segmentation_settings
from .photoshop import PHOTOSHOP, PhotoshopService, photoshop_settings
from .registry import SerialService, ServiceKey, ServiceRegistry, ServiceScope


def create_service_registry() -> ServiceRegistry:
    registry = ServiceRegistry()
    registry.register(PHOTOSHOP, PhotoshopService, settings=photoshop_settings)
    registry.register(SEGMENTATION, OnnxService, settings=segmentation_settings)
    return registry


# Only factory registrations are process-wide; each root graph owns its services.
DEFAULT_SERVICES = create_service_registry()
