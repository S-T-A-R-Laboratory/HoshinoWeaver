"""Readable GUI presentation of startup backend diagnostics."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (QDialogButtonBox, QLabel, QPushButton,
                               QTextBrowser, QVBoxLayout)

from hoshicore.component.runtime_diagnostics import (get_runtime_components_report,
                                                     probe_external_components)
from hoshicore.engine.wiring import _load_default_settings
from ui.UILibs import uQDialog
from ui.runtime_capabilities_format import format_runtime_capabilities_html


class _ProbeSignals(QObject):
    ready = Signal(dict)


class _ExternalProbe(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _ProbeSignals()

    def run(self):
        try:
            report = probe_external_components(_load_default_settings())
        except Exception as error:
            report = {name: {'status': 'unavailable', 'reason': str(error)}
                      for name in ('ffmpeg', 'photoshop')}
        self.signals.ready.emit(report)


class RuntimeCapabilitiesDialog(uQDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("运行环境与组件检测")
        self.setMinimumSize(650, 520)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "检查加速后端、图像组件及外部程序可用性。\n"
            "缺失可选加速通常不会影响功能，但可能降低处理速度。", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.details = QTextBrowser(self)
        self.details.setOpenExternalLinks(False)
        layout.addWidget(self.details, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        refresh = self.refresh_button = QPushButton("重新检测", self)
        buttons.addButton(refresh, QDialogButtonBox.ActionRole)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        refresh.clicked.connect(
            lambda _checked=False: self.refresh(refresh=True))
        layout.addWidget(buttons)

        self.refresh()

    def refresh(self, *, refresh: bool = False) -> None:
        if not self.refresh_button.isEnabled():
            return
        self._report = dict(get_runtime_components_report(refresh=refresh))
        self.details.setHtml(format_runtime_capabilities_html(self._report))
        self.refresh_button.setEnabled(False)
        self._probe = _ExternalProbe()
        self._probe.signals.ready.connect(self._external_ready)
        QThreadPool.globalInstance().start(self._probe)

    @Slot(dict)
    def _external_ready(self, report):
        self._report['external_components'] = report
        self.details.setHtml(format_runtime_capabilities_html(self._report))
        self.refresh_button.setEnabled(True)
