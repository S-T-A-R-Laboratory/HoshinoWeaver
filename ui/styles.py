"""Shared UI style constants for dynamic panels."""

LINEEDIT_STYLE = (
    "QLineEdit {"
    "  border: 1px solid rgba(220,220,220,200);"
    "  border-radius: 3px;"
    "  background-color: rgba(250,250,250,200);"
    "  padding: 2px 4px;"
    "  font-size: 11px;"
    "}"
)

COMBOBOX_STYLE = (
    "QComboBox {"
    "  border: 0px solid rgba(199,199,199,100);"
    "  padding: 3px;"
    "  border-radius: 3px;"
    "  font-size: 11px;"
    "  color: rgba(30,30,30,200);"
    "}"
    "QComboBox::hover {"
    "  background-color: rgba(0, 212, 254, 10);"
    "}"
    "QComboBox QAbstractItemView {"
    "  background-color: rgba(255,255,255,1);"
    "  border: 0px solid rgba(199,199,199,100);"
    "}"
)

LABEL_STYLE = "font-size: 11px; color: rgba(30,30,30,200);"

GROUP_HEADER_STYLE = (
    "font-weight: bold; font-size: 11px;"
    "color: rgba(255,255,255,220);"
    "background-color: rgba(90,90,90,200);"
    "border: none;"
    "border-top-left-radius: 3px;"
    "border-top-right-radius: 3px;"
    "padding: 3px 6px;"
)

GROUP_BOX_STYLE = (
    "QFrame#groupBox {"
    "  border: 1px solid rgba(160,160,160,160);"
    "  border-radius: 4px;"
    "  background-color: rgba(245,245,245,80);"
    "  margin-top: 4px;"
    "}"
    "QFrame#groupBox > QLabel#groupHeader {"
    "  border: none;"
    "  border-radius: 0px;"
    "}"
)

ROW_HEIGHT = 36
LABEL_MIN_WIDTH = 60
LABEL_MAX_WIDTH = 80
