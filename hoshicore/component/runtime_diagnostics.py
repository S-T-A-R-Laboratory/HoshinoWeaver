"""Lightweight startup probes for native backends and I/O integrations."""
from __future__ import annotations

import importlib
import sys
from collections import Counter
from typing import Any

from .._custom_op._dispatch import (cuda_memory_info, load_compiled_module,
                                    metal_device_info)
from .._custom_op.backend_registry import (registered_backend_candidates,
                                           select_backend)
from .photoshop import resolve_photoshop
from .video_io import probe_ffmpeg, resolve_ffmpeg

_logged = False
_last_report: dict[str, Any] | None = None


def _module_probe(module_name: str, *, initialize=None) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        if initialize is not None:
            initialize(module)
        version = getattr(module, "__version__", None)
        return {"status": "available", "version": version}
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _turbojpeg_initialize(module: Any) -> None:
    # Importing the binding alone does not prove that the native library can be
    # located. Construction resolves and loads libturbojpeg without decoding.
    module.TurboJPEG()


def _runtime_probe(probe) -> dict[str, Any]:
    try:
        payload = dict(probe())
        payload.setdefault(
            "status", "available" if payload.get("available") else "unavailable")
        return payload
    except Exception as exc:
        return {
            "available": False,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def probe_external_components(configs) -> dict[str, Any]:
    """Check configured tools on request.

    FFmpeg executes bounded, hidden encoder/muxer help commands. These checks
    are intentionally separate from startup probes and their cached report.
    """
    ffmpeg = {"status": "unavailable"}
    try:
        ffmpeg['path'] = resolve_ffmpeg(configs.get('ffmpeg_path'))
        formats = {}
        for label, container, codec in (('H.264 / MP4', 'mp4', 'h264'),
                                         ('H.264 / AVI', 'avi', 'h264'),
                                         ('RGB / AVI', 'avi', 'raw_rgb')):
            try:
                probe_ffmpeg(ffmpeg['path'], container=container, codec=codec)
                formats[label] = {"status": "available"}
            except Exception as error:
                formats[label] = {"status": "unavailable", "reason": str(error)}
        ffmpeg['formats'] = formats
        if any(value['status'] == 'available' for value in formats.values()):
            ffmpeg['status'] = 'available'
        else:
            ffmpeg['reason'] = 'No supported video encoder/muxer combination is available'
    except Exception as error:
        ffmpeg['reason'] = str(error)

    photoshop = {"status": "unavailable"}
    try:
        

        path = resolve_photoshop(configs.get('photoshop_path'))
        if sys.platform == 'win32' and (not path.is_file() or path.suffix.lower() != '.exe'):
            raise ValueError('photoshop_path must point to the Photoshop .exe file')
        if sys.platform == 'darwin' and not path.is_dir():
            raise ValueError('photoshop_path must point to the Photoshop .app bundle')
        if sys.platform == 'darwin':
            import plistlib

            with (path / 'Contents/Info.plist').open('rb') as stream:
                executable = plistlib.load(stream)['CFBundleExecutable']
            if not (path / 'Contents/MacOS' / executable).is_file():
                raise ValueError('Photoshop bundle executable is missing')
        photoshop.update(status='configured', path=str(path))
    except Exception as error:
        photoshop['reason'] = str(error)
    return {'ffmpeg': ffmpeg, 'photoshop': photoshop}


def probe_runtime_components() -> dict[str, Any]:
    """Inspect component availability without executing image-processing kernels."""
    compiled_module, compiled_error = load_compiled_module()
    if compiled_module is None:
        compiled = {
            "status": "unavailable",
            "available": False,
            "reason": compiled_error or "compiled module is unavailable",
        }
        build_info: dict[str, Any] = {}
    else:
        try:
            build_info = dict(compiled_module.build_info())
            compiled = {"status": "available", **build_info}
        except Exception as exc:
            build_info = {}
            compiled = {
                "status": "error",
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    if build_info.get("cuda"):
        cuda = _runtime_probe(cuda_memory_info)
    else:
        cuda = {
            "available": False,
            "status": "not_built",
            "reason": "compiled custom ops do not include CUDA",
        }
    metal = _runtime_probe(metal_device_info)

    excluded = set()
    if cuda.get("status") != "available":
        excluded.add("cuda_host_io")
    if metal.get("status") != "available":
        excluded.add("metal_host_io")
    logical_ops = sorted({
        candidate.logical_op for candidate in registered_backend_candidates()
    })
    selections = {}
    for logical_op in logical_ops:
        selection = select_backend(
            logical_op,
            load_module=lambda: (compiled_module, compiled_error),
            build_info=build_info,
            exclude_backends=excluded,
        )
        selections[logical_op] = {
            "backend": selection.backend,
            "native": selection.native,
            "reason": selection.reason,
        }

    modules = {
        "turbojpeg": _module_probe(
            "turbojpeg", initialize=_turbojpeg_initialize),
        "pyexiv2": _module_probe("pyexiv2")
    }
    return {
        "compiled_custom_ops": compiled,
        "cuda_runtime": cuda,
        "metal_runtime": metal,
        "logical_ops": selections,
        "modules": modules,
    }


def get_runtime_components_report(*, refresh: bool = False) -> dict[str, Any]:
    """Return the cached startup report, optionally probing again."""
    global _last_report
    if refresh or _last_report is None:
        _last_report = probe_runtime_components()
    return _last_report


def _status_text(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "unknown"))
    reason = payload.get("reason")
    return f"{status} ({reason})" if reason else status


def log_runtime_components(logger: Any) -> dict[str, Any] | None:
    """Probe and log capabilities once per process; never fail application startup."""
    global _logged
    if _logged:
        return None
    _logged = True
    try:
        report = get_runtime_components_report()
    except Exception as exc:  # Keep diagnostics strictly non-blocking.
        logger.warning("[Capabilities] probe failed: {}: {}",
                       type(exc).__name__, exc)
        return None

    compiled = report["compiled_custom_ops"]
    if compiled.get("status") == "available":
        logger.info(
            "[Capabilities] custom_ops=available compiler={} openmp={} cuda={} "
            "metal_module={}",
            compiled.get("compiler", "unknown"),
            bool(compiled.get("openmp")),
            bool(compiled.get("cuda")),
            report["metal_runtime"].get("status") == "available",
        )
    else:
        logger.info("[Capabilities] custom_ops={}", _status_text(compiled))

    cuda = report["cuda_runtime"]
    cuda_detail = _status_text(cuda)
    if cuda.get("status") == "available":
        cuda_detail += (
            f" device={cuda.get('device', '?')}"
            f" compute={cuda.get('compute_capability_major', '?')}."
            f"{cuda.get('compute_capability_minor', '?')}"
        )
    logger.info("[Capabilities] cuda={} metal={}",
                cuda_detail, _status_text(report["metal_runtime"]))

    module_parts = [
        f"{name}={_status_text(payload)}"
        for name, payload in report["modules"].items()
    ]
    logger.info("[Capabilities] io: {}", ", ".join(module_parts))

    backend_counts = Counter(
        payload["backend"] for payload in report["logical_ops"].values())
    logger.info(
        "[Capabilities] logical backends: {}",
        ", ".join(f"{name}={count}" for name, count in sorted(
            backend_counts.items())),
    )
    logger.debug("[Capabilities] logical backend map: {}",
                 report["logical_ops"])
    return report


def main() -> None:
    """Standalone entry point: ``python -m hoshicore.component.runtime_diagnostics``."""
    from loguru import logger

    log_runtime_components(logger)


if __name__ == "__main__":
    main()
