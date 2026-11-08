"""Preview tool strip with explicit mask availability and visibility states."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton


class PreviewToolbar(QFrame):
    def __init__(self, icons_dir, parent=None):
        super().__init__(parent)
        self.setObjectName('preview_toolbar')
        self.setFixedHeight(36)
        self.setStyleSheet('''
            QFrame#preview_toolbar {
                background: #f7f9fc; border: none; border-bottom: 1px solid #dce5ed;
            }
            QToolButton {
                background: transparent; border: 1px solid transparent;
                border-radius: 7px; padding: 2px;
            }
            QToolButton:hover:enabled { background: #f0f6fc; border-color: #b7ccdf; }
            QToolButton:checked:enabled { background: #e4effa; border-color: #9bbad5; }
            QToolButton:pressed:enabled { background: #dceaf7; }
            QToolButton:focus:enabled { border-color: #719abb; }
        ''')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 4, 3)
        layout.setSpacing(4)
        layout.addStretch()

        def button(name, label):
            control = QToolButton(self)
            control.setObjectName(name)
            control.setText(label)
            control.setAccessibleName(label)
            control.setToolTip(label)
            control.setToolButtonStyle(Qt.ToolButtonIconOnly)
            control.setAutoRaise(True)
            control.setIconSize(QSize(24, 24))
            control.setFixedSize(30, 30)
            layout.addWidget(control)
            return control

        self._visible_icon = QIcon(str(icons_dir / 'mask_visible.svg'))
        self._hidden_icon = QIcon(str(icons_dir / 'mask_hidden.svg'))
        self._mask_reason = '尚未应用蒙版，请先在工作流蒙版参数中保存并应用。'
        self.mask_toggle = button('mask_preview_toggle', '显示蒙版')
        self.mask_toggle.setCheckable(True)
        self.mask_toggle.toggled.connect(self._update_mask_icon)
        self.edit_button = button('mask_editor_button', '蒙版编辑器')
        self.edit_button.setIcon(QIcon(str(icons_dir / 'mask_editor.svg')))
        self.set_mask_state(False, self._mask_reason)

    def set_mask_state(self, available, reason):
        self._mask_reason = reason
        self.mask_toggle.setEnabled(available)
        self._update_mask_icon()

    def _update_mask_icon(self):
        shown = self.mask_toggle.isChecked()
        self.mask_toggle.setIcon(self._visible_icon if shown else self._hidden_icon)
        label = '隐藏蒙版（当前已显示）' if shown else '显示蒙版（当前已隐藏）'
        if not self.mask_toggle.isEnabled():
            label = '蒙版显示不可用'
        self.mask_toggle.setAccessibleName(label)
        self.mask_toggle.setToolTip(f'{label}\n{self._mask_reason}')
