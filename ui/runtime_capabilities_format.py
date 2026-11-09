"""Qt-free presentation helpers for runtime capability diagnostics."""

from __future__ import annotations

import html
from collections import Counter
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _reason(payload: dict[str, Any]) -> str:
    return _escape(payload.get("reason") or "未提供具体原因")


def format_runtime_capabilities_html(report: dict[str, Any]) -> str:
    """Format a capability report for non-technical users."""
    compiled = report["compiled_custom_ops"]
    cuda = report["cuda_runtime"]
    metal = report["metal_runtime"]
    modules = report["modules"]
    logical_ops = report["logical_ops"]

    if compiled.get("status") == "available":
        compiler = _escape(compiled.get("compiler", "未知"))
        native_text = f"原生算子可用（{compiler} 编译）"
        native_class = "ok"
    else:
        native_text = (
            "原生算子不可用；程序将使用 NumPy/兼容实现。"
            f"原因：{_reason(compiled)}"
        )
        native_class = "warn"

    openmp_text = (
        "可用，可使用多核 CPU 加速"
        if compiled.get("openmp")
        else "不可用，将使用单线程或 NumPy 路径"
    )
    if not compiled.get("cuda"):
        cuda_text = "当前版本未编译 CUDA"
        cuda_class = "neutral"
    elif cuda.get("status") == "available":
        cuda_text = "可用，设备 {device}，计算能力 {major}.{minor}".format(
            device=_escape(cuda.get("device", "?")),
            major=_escape(cuda.get("compute_capability_major", "?")),
            minor=_escape(cuda.get("compute_capability_minor", "?")),
        )
        cuda_class = "ok"
    else:
        cuda_text = (
            "当前版本包含 CUDA，但目标设备/驱动不可用；将自动回退 CPU。原因："
            f"{_reason(cuda)}"
        )
        cuda_class = "warn"

    if metal.get("status") == "available":
        metal_text = "可使用 Apple Metal 加速"
        metal_class = "ok"
    else:
        metal_text = "不可用或当前平台未包含（非 macOS 平台通常无需此后端）"
        metal_class = "neutral"

    counts = Counter(item["backend"] for item in logical_ops.values())
    backend_names = {
        "cuda_host_io": "CUDA",
        "metal_host_io": "Metal",
        "openmp_cpu": "OpenMP CPU",
        "numpy": "NumPy/兼容路径",
    }
    backend_text = "；".join(
        f"{backend_names.get(name, name)} {count} 项"
        for name, count in sorted(counts.items())
    ) or "未注册逻辑算子"

    io_labels = {
        "turbojpeg": ("TurboJPEG", "JPEG 快速解码", "回退 OpenCV 解码"),
        "pyexiv2": ("pyexiv2", "EXIF/ICC 读写", "EXIF/ICC 功能可能不可用"),
    }
    io_rows = []
    for key, (label, available_hint, missing_hint) in io_labels.items():
        payload = modules.get(key, {"status": "unavailable"})
        available = payload.get("status") == "available"
        state = "可用" if available else "不可用"
        css = "ok" if available else "warn"
        hint = available_hint if available else missing_hint
        io_rows.append(
            f'<tr><td>{label}</td><td class="{css}">{state}</td>'
            f"<td>{hint}</td></tr>"
        )

    external_rows = []
    for key, label in (('ffmpeg', 'FFmpeg'), ('photoshop', 'Photoshop')):
        payload = report.get('external_components', {}).get(key, {'status': 'pending'})
        status = payload.get('status')
        state = {'available': '可用', 'configured': '路径有效',
                 'pending': '检测中……'}.get(status, '不可用')
        css = 'ok' if status == 'available' else 'neutral' if status in ('configured', 'pending') else 'warn'
        detail = _escape(payload.get('path', ''))
        if status == 'unavailable':
            detail += '<br>' + _reason(payload)
        if key == 'ffmpeg':
            for name, result in payload.get('formats', {}).items():
                text = '可用' if result['status'] == 'available' else '不可用：' + _reason(result)
                detail += f'<br>{_escape(name)}：{text}'
            if state.startswith("不可用"):
                detail += '<br>需在全局设置配置 ffmpeg 路径。'
        else:
            if state.startswith("不可用"):
                detail += '<br>需在全局设置配置 photoshop 路径。'
        external_rows.append(f'<tr><td>{label}</td><td class="{css}">{state}</td><td>{detail}</td></tr>')

    return f"""
    <style>
      body {{ color: #eeeeee; font-size: 13px; }}
      h3 {{ margin: 10px 0 5px 0; color: #000000; }}
      .ok {{ color: #75d69c; }}
      .warn {{ color: #ffbf69; }}
      .neutral {{ color: #c8c8c8; }}
      table {{ border-collapse: collapse; width: 100%; }}
      td {{ padding: 4px 8px 4px 0; border-bottom: 1px solid #555555; }}
    </style>
    <h3>计算后端</h3>
    <p class="{native_class}">{native_text}</p>
    <table>
      <tr><td>OpenMP CPU</td><td>{openmp_text}</td></tr>
      <tr><td>CUDA</td><td class="{cuda_class}">{cuda_text}</td></tr>
      <tr><td>Apple Metal</td><td class="{metal_class}">{metal_text}</td></tr>
    </table>
    <p><b>当前逻辑算子路径：</b>{backend_text}</p>
    <h3>图像与视频组件</h3>
    <table>{''.join(io_rows)}</table>
    <h3>外部程序</h3>
    <table>{''.join(external_rows)}</table>
    """
