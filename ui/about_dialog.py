"""AboutDialog: project information dialog opened by menu_about button."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)

from hoshicore.component.utils import ORG_NAME, SOFTWARE_NAME, VERSION
from ui import resource  # noqa: F401  — registers Qt icon resources
from ui.UILibs import uQDialog

_LICENSE_PATH = Path(__file__).parent.parent / "LICENSE"
_GITHUB_URL = "https://github.com/PLACEHOLDER/HoshinoWeaver"


def _read_license_first_line() -> str:
    try:
        with open(_LICENSE_PATH, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except OSError:
        return ""


class AboutDialog(uQDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于")
        self.setMinimumWidth(420)
        self._build()
        self._center()

    def _center(self):
        parent = self.parent()
        if isinstance(parent, QWidget):
            geo = parent.geometry()
            self.adjustSize()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # Top row: icon (left, top-aligned) + title block (right)
        top = QHBoxLayout()
        top.setSpacing(14)

        icon_label = QLabel(self)
        px = QPixmap(":/icons/resource/icon/about.png").scaled(
            64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(px)
        icon_label.setFixedSize(64, 64)
        icon_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        top.addWidget(icon_label, 0, Qt.AlignTop)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)

        name_lbl = QLabel(SOFTWARE_NAME, self)
        name_lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_block.addWidget(name_lbl)

        subtitle_lbl = QLabel("织此星辰", self)
        subtitle_lbl.setStyleSheet("font-size: 12px; color: rgba(80,80,80,200);")
        title_block.addWidget(subtitle_lbl)

        meta_lbl = QLabel(f"v{VERSION}  ·  {ORG_NAME}", self)
        meta_lbl.setStyleSheet("font-size: 11px; color: rgba(100,100,100,180);")
        title_block.addWidget(meta_lbl)

        top.addLayout(title_block, 1)
        root.addLayout(top)

        # Separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # Description
        desc_lbl = QLabel(
            "天文图像堆叠预处理工具，支持星轨、天地分离、降噪等工作流。", self)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 12px;")
        root.addWidget(desc_lbl)

        # GitHub link
        gh_lbl = QLabel(
            f'GitHub: <a href="{_GITHUB_URL}">{_GITHUB_URL}</a>', self)
        gh_lbl.setOpenExternalLinks(True)
        gh_lbl.setStyleSheet("font-size: 11px;")
        root.addWidget(gh_lbl)

        # License (first line of LICENSE file)
        license_text = _read_license_first_line()
        if license_text:
            lic_lbl = QLabel(f"许可证：{license_text}", self)
            lic_lbl.setStyleSheet("font-size: 11px; color: rgba(80,80,80,200);")
            root.addWidget(lic_lbl)

        # OK button
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok, self)
        btn_box.button(QDialogButtonBox.Ok).setText("确定")
        btn_box.accepted.connect(self.accept)
        root.addWidget(btn_box)
