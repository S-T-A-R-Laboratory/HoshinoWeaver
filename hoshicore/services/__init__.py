"""Application services. Importing this module starts no external resources."""
from .onnx import SEGMENTATION, OnnxService, segmentation_settings
from .photoshop import PHOTOSHOP, PhotoshopService, photoshop_settings
from .registry import SerialService, ServiceKey, ServiceRegistry, ServiceScope


def create_service_registry() -> ServiceRegistry:
    registry = ServiceRegistry()
    registry.register(PHOTOSHOP, PhotoshopService, settings=photoshop_settings,
                      resource_key=lambda options: options['path'])
    registry.register(SEGMENTATION, OnnxService, settings=segmentation_settings)
    return registry


# Factories and active leases are process-local; native resources live only
# while at least one root graph holds a lease.
DEFAULT_SERVICES = create_service_registry()
