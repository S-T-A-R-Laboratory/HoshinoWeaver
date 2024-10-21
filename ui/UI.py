# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QSplitter,
    QTabWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSpacerItem, QStackedWidget, QVBoxLayout, QWidget)


from ui.UILibs import ClickableLabel, DoubleSlider, ImgTreeWidget, hoverDisplayButton,imgDisplayQFrame,ClickableLabel
from ui import resource


class Ui_HNW(object):
    def setupUi(self, HNW):
        if not HNW.objectName():
            HNW.setObjectName(u"HNW")
        HNW.resize(1086, 769)
        self.centralwidget = QWidget(HNW)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QSize(1000, 700))
        self.centralwidget.setStyleSheet(u"#centralwidget {\n"
"}\n"
"QFrame{\n"
"border: none;\n"
"}\n"
"QWidget{\n"
"border : none\n"
"}\n"
"* QToolTip {\n"
"	/* \u5168\u5c40\u5e94\u7528 */\n"
"    background-color: rgba(250, 250, 250, 240) !important;  /* \u8bbe\u7f6e\u5de5\u5177\u63d0\u793a\u7684\u80cc\u666f\u4e3a\u534a\u900f\u660e\u767d\u8272 */\n"
"    color:rgba(35,35,35,210) !important;  /* \u8bbe\u7f6e\u6587\u5b57\u989c\u8272\u4e3a\u9ed1\u8272 */\n"
"    border: 1px solid rgba(220, 220, 220, 250) !important;  /* \u53ef\u9009\uff1a\u4e3a\u5de5\u5177\u63d0\u793a\u6dfb\u52a0\u8fb9\u6846 */\n"
"}\n"
"")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.main_frame = QFrame(self.centralwidget)
        self.main_frame.setObjectName(u"main_frame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.main_frame.sizePolicy().hasHeightForWidth())
        self.main_frame.setSizePolicy(sizePolicy1)
        self.main_frame.setStyleSheet(u"#main_frame {\n"
"	border:none;\n"
"	background-image: url(:/img/resource/img/\u76bf\u4ed3\u5c71\u661f\u8f68-s.jpg);\n"
"	background-repeat: no-repeat;                     /* \u4e0d\u91cd\u590d\u80cc\u666f\u56fe */\n"
"    background-position: center;                      /* \u5c45\u4e2d\u663e\u793a\u80cc\u666f\u56fe */\n"
"\n"
"}\n"
"")
        self.main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.main_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.main_frame)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.navigation_bar = QFrame(self.main_frame)
        self.navigation_bar.setObjectName(u"navigation_bar")
        sizePolicy1.setHeightForWidth(self.navigation_bar.sizePolicy().hasHeightForWidth())
        self.navigation_bar.setSizePolicy(sizePolicy1)
        self.navigation_bar.setMinimumSize(QSize(0, 40))
        self.navigation_bar.setMaximumSize(QSize(16777215, 40))
        self.navigation_bar.setStyleSheet(u"#navigation_bar {\n"
"\n"
"}")
        self.navigation_bar.setFrameShape(QFrame.Shape.StyledPanel)
        self.navigation_bar.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.navigation_bar)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_top_right = QFrame(self.navigation_bar)
        self.frame_top_right.setObjectName(u"frame_top_right")
        sizePolicy1.setHeightForWidth(self.frame_top_right.sizePolicy().hasHeightForWidth())
        self.frame_top_right.setSizePolicy(sizePolicy1)
        self.frame_top_right.setMinimumSize(QSize(250, 0))
        self.frame_top_right.setMaximumSize(QSize(250, 16777215))
        self.frame_top_right.setStyleSheet(u"QFrame{\n"
"		border:None;\n"
"		background-color: rgba(255, 255, 255, 150);\n"
"    }")
        self.frame_top_right.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_top_right.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_top_right)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(self.frame_top_right)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"QLabel {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	font: 16pt \"\u6c49\u4eea\u4e2d\u9ed1 197\";\n"
"}")

        self.horizontalLayout_3.addWidget(self.label)


        self.horizontalLayout_2.addWidget(self.frame_top_right)

        self.frame_top_center = QFrame(self.navigation_bar)
        self.frame_top_center.setObjectName(u"frame_top_center")
        sizePolicy1.setHeightForWidth(self.frame_top_center.sizePolicy().hasHeightForWidth())
        self.frame_top_center.setSizePolicy(sizePolicy1)
        self.frame_top_center.setStyleSheet(u"QFrame{\n"
"\n"
"		border:None;\n"
"		background-color: rgba(255, 255, 255, 150);\n"
"    }")
        self.frame_top_center.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_top_center.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_top_center)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_current_mode = ClickableLabel(self.frame_top_center)
        self.label_current_mode.setObjectName(u"label_current_mode")
        self.label_current_mode.setMaximumSize(QSize(159, 16777215))
        self.label_current_mode.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border:None;\n"
"	font: 14pt \"\u5b8b\u4f53\";\n"
"}")

        self.horizontalLayout_4.addWidget(self.label_current_mode)


        self.horizontalLayout_2.addWidget(self.frame_top_center)

        self.frame_top_left = QFrame(self.navigation_bar)
        self.frame_top_left.setObjectName(u"frame_top_left")
        sizePolicy1.setHeightForWidth(self.frame_top_left.sizePolicy().hasHeightForWidth())
        self.frame_top_left.setSizePolicy(sizePolicy1)
        self.frame_top_left.setMinimumSize(QSize(220, 0))
        self.frame_top_left.setMaximumSize(QSize(220, 16777215))
        self.frame_top_left.setStyleSheet(u"QPushButton {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	margin:5px;\n"
"}\n"
"QPushButton:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"QPushButton:pressed {\n"
"	padding-bottom: 0px;\n"
"}\n"
"QFrame{\n"
"		border:None;\n"
"    }\n"
"\n"
"#frame_top_left {\n"
"	background-color: rgba(255, 255, 255, 150);\n"
"}\n"
"")
        self.frame_top_left.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_top_left.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_top_left)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.frame_menu = QFrame(self.frame_top_left)
        self.frame_menu.setObjectName(u"frame_menu")
        sizePolicy1.setHeightForWidth(self.frame_menu.sizePolicy().hasHeightForWidth())
        self.frame_menu.setSizePolicy(sizePolicy1)
        self.frame_menu.setMinimumSize(QSize(90, 0))
        self.frame_menu.setMaximumSize(QSize(90, 16777215))
        self.frame_menu.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_menu.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_menu)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.menu_setting = QPushButton(self.frame_menu)
        self.menu_setting.setObjectName(u"menu_setting")
        self.menu_setting.setMinimumSize(QSize(25, 25))
        self.menu_setting.setMaximumSize(QSize(25, 25))
        self.menu_setting.setStyleSheet(u"")
        icon = QIcon()
        icon.addFile(u":/icons/resource/icon/setting-2.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.menu_setting.setIcon(icon)
        self.menu_setting.setIconSize(QSize(20, 20))

        self.horizontalLayout_7.addWidget(self.menu_setting)

        self.menu_about = QPushButton(self.frame_menu)
        self.menu_about.setObjectName(u"menu_about")
        self.menu_about.setMinimumSize(QSize(25, 25))
        self.menu_about.setMaximumSize(QSize(25, 25))
        self.menu_about.setStyleSheet(u"")
        icon1 = QIcon()
        icon1.addFile(u":/icons/resource/icon/about.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.menu_about.setIcon(icon1)
        self.menu_about.setIconSize(QSize(20, 20))

        self.horizontalLayout_7.addWidget(self.menu_about)


        self.horizontalLayout_5.addWidget(self.frame_menu)

        self.label_3 = QLabel(self.frame_top_left)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(1, 25))
        self.label_3.setMaximumSize(QSize(1, 25))
        self.label_3.setStyleSheet(u"#label_3 {\n"
"border-width:1px;\n"
"border-color:rgba(199,199,199,255);\n"
"border-style:inset\n"
"\n"
"}")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.frame_ui_win_handle = QFrame(self.frame_top_left)
        self.frame_ui_win_handle.setObjectName(u"frame_ui_win_handle")
        sizePolicy1.setHeightForWidth(self.frame_ui_win_handle.sizePolicy().hasHeightForWidth())
        self.frame_ui_win_handle.setSizePolicy(sizePolicy1)
        self.frame_ui_win_handle.setMinimumSize(QSize(130, 0))
        self.frame_ui_win_handle.setMaximumSize(QSize(130, 16777215))
        self.frame_ui_win_handle.setStyleSheet(u"")
        self.frame_ui_win_handle.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_ui_win_handle.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_ui_win_handle)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.ui_min = QPushButton(self.frame_ui_win_handle)
        self.ui_min.setObjectName(u"ui_min")
        self.ui_min.setMinimumSize(QSize(25, 25))
        self.ui_min.setMaximumSize(QSize(25, 25))
        self.ui_min.setStyleSheet(u"")
        icon2 = QIcon()
        icon2.addFile(u":/icons/resource/icon/min.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.ui_min.setIcon(icon2)
        self.ui_min.setIconSize(QSize(20, 20))

        self.horizontalLayout_6.addWidget(self.ui_min)

        self.ui_max = QPushButton(self.frame_ui_win_handle)
        self.ui_max.setObjectName(u"ui_max")
        self.ui_max.setMinimumSize(QSize(25, 25))
        self.ui_max.setMaximumSize(QSize(25, 25))
        icon3 = QIcon()
        icon3.addFile(u":/icons/resource/icon/max.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.ui_max.setIcon(icon3)
        self.ui_max.setIconSize(QSize(20, 20))

        self.horizontalLayout_6.addWidget(self.ui_max)

        self.ui_close = QPushButton(self.frame_ui_win_handle)
        self.ui_close.setObjectName(u"ui_close")
        self.ui_close.setMinimumSize(QSize(25, 25))
        self.ui_close.setMaximumSize(QSize(25, 25))
        icon4 = QIcon()
        icon4.addFile(u":/icons/resource/icon/close.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.ui_close.setIcon(icon4)
        self.ui_close.setIconSize(QSize(20, 20))

        self.horizontalLayout_6.addWidget(self.ui_close)


        self.horizontalLayout_5.addWidget(self.frame_ui_win_handle)


        self.horizontalLayout_2.addWidget(self.frame_top_left)


        self.verticalLayout_2.addWidget(self.navigation_bar)

        self.workspace = QFrame(self.main_frame)
        self.workspace.setObjectName(u"workspace")
        sizePolicy.setHeightForWidth(self.workspace.sizePolicy().hasHeightForWidth())
        self.workspace.setSizePolicy(sizePolicy)
        self.workspace.setStyleSheet(u"#args_area {\n"
"background-color: rgba(0,0, 255,0);\n"
"}\n"
"#preview_area {\n"
"background-color: rgba(255,255, 255,255);\n"
"}\n"
"#workspace {\n"
"	padding-bottom: 0px;\n"
"}")
        self.workspace.setFrameShape(QFrame.Shape.StyledPanel)
        self.workspace.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.workspace)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(self.workspace)
        self.splitter.setObjectName(u"splitter")
        sizePolicy1.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy1)
        self.splitter.setStyleSheet(u"QSplitter::handle {\n"
"    width: 0px; /* \u4fee\u6539\u4e3a\u4f60\u60f3\u8981\u7684\u5bbd\u5ea6 */\n"
"	background-color: rgba(190,190,190,255);\n"
"}\n"
"")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.args_area = QFrame(self.splitter)
        self.args_area.setObjectName(u"args_area")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.args_area.sizePolicy().hasHeightForWidth())
        self.args_area.setSizePolicy(sizePolicy2)
        self.args_area.setMinimumSize(QSize(250, 600))
        self.args_area.setMaximumSize(QSize(250, 16777215))
        self.args_area.setStyleSheet(u"#args_area {\n"
"		border:None;\n"
"		background-color: rgba(255, 255, 255, 240);\n"
"\n"
"}\n"
"#args_area * {\n"
"		border:None;\n"
"		background-color: rgba(255, 255, 255, 0);\n"
"\n"
"}\n"
"\n"
"")
        self.args_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.args_area.setFrameShadow(QFrame.Shadow.Raised)
        self.args_area.setLineWidth(0)
        self.horizontalLayout_8 = QHBoxLayout(self.args_area)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.star_trail_option_box = QTabWidget(self.args_area)
        self.star_trail_option_box.setObjectName(u"star_trail_option_box")
        sizePolicy.setHeightForWidth(self.star_trail_option_box.sizePolicy().hasHeightForWidth())
        self.star_trail_option_box.setSizePolicy(sizePolicy)
        self.star_trail_option_box.setMinimumSize(QSize(0, 0))
        self.star_trail_option_box.setStyleSheet(u"QTabWidget::pane {\n"
"	border-left: 1px solid rgba(240, 240, 240, 200);\n"
"}\n"
"QTabWidget::tab-bar {\n"
"	background-color: rgba(0, 255, 255, 255);\n"
"}\n"
"QTabWidget::left-corner{\n"
"	background-color: rgba(0, 255, 255, 255);\n"
"}\n"
"QTabBar::tab {\n"
"	background-color: rgba(220,220,220,150);\n"
"	color:rgba(35,35,35,210);\n"
"	padding: 0px;\n"
"	border: 0px solid #ccc;\n"
"	border-radius: 5px;\n"
"	width:25px;\n"
"	height:80px;\n"
"	padding-left: 3px;\n"
"	padding-top: 3px;\n"
"	margin-bottom: 1px;\n"
"	margin-left: 2px\n"
"}\n"
"QTabBar::tab:selected {\n"
"	background-color: transparent ; \n"
"	font: 550 10pt \"Microsoft YaHei UI\";\n"
"	color:rgba(40,40,40,245);\n"
"	border: 0px solid #dcdcdc;\n"
"	border-bottom: none;\n"
"	padding-left: 0px;\n"
"}\n"
"QTabBar::tab::icon {\n"
"    padding-bottom: 5px;  /* \u56fe\u6807\u4e0e\u6807\u7b7e\u8fb9\u6846\u7684\u8ddd\u79bb */\n"
"}\n"
"* QToolTip {\n"
"	/* \u5168\u5c40\u5e94\u7528 */\n"
"    background-color: rgba(250, 250, 250, 240) !important;  /* \u8bbe"
                        "\u7f6e\u5de5\u5177\u63d0\u793a\u7684\u80cc\u666f\u4e3a\u534a\u900f\u660e\u767d\u8272 */\n"
"    color:rgba(35,35,35,210) !important;  /* \u8bbe\u7f6e\u6587\u5b57\u989c\u8272\u4e3a\u9ed1\u8272 */\n"
"    border: 1px solid rgba(220, 220, 220, 250) !important;  /* \u53ef\u9009\uff1a\u4e3a\u5de5\u5177\u63d0\u793a\u6dfb\u52a0\u8fb9\u6846 */\n"
"}\n"
"\n"
"\n"
"\n"
"/* \u6ed1\u5757\u8bbe\u7f6e */\n"
"QSlider {\n"
"	margin-left:10px\n"
"}\n"
"QSlider::groove {\n"
"    /* \u6ed1\u69fd\u90e8\u5206 \u4f1a\u88ab sub-page\u548cadd-page\u8986\u76d6*/\n"
"    border: 0px solid #999999;\n"
"    height: 2px;\n"
"    background: rgba(190,190,190,190);\n"
"    border-radius: 2px;\n"
"}\n"
"QSlider::handle {\n"
"	/* \u6ed1\u5757\u90e8\u5206 */\n"
"    background: #5c5c5c;\n"
"    border: 0px solid #5c5c5c;\n"
"    width: 6px;\n"
"    margin: -6px 0;\n"
"    border-radius: 10px;\n"
"}\n"
"QSlider::sub-page {\n"
"	/* \u5df2\u7ecf\u8fc7\u7684\u6ed1\u69fd\u90e8\u5206 */\n"
"    background: #66cc66;\n"
"}\n"
"QSlider::add-page {\n"
""
                        "	/* \u672a\u7ecf\u8fc7\u7684\u6ed1\u69fd\u90e8\u5206 */\n"
"}\n"
"QSlider::handle:horizontal:hover,QSlider::handle:horizontal:pressed {\n"
"    background: #66cc66;\n"
"}\n"
"")
        self.star_trail_option_box.setTabPosition(QTabWidget.TabPosition.West)
        self.star_trail_option_box.setIconSize(QSize(20, 20))
        self.star_trail_option_box.setElideMode(Qt.TextElideMode.ElideLeft)
        self.star_trail_option_box.setUsesScrollButtons(True)
        self.star_trail_option_box.setTabBarAutoHide(False)
        self.star_trail_input_files = QWidget()
        self.star_trail_input_files.setObjectName(u"star_trail_input_files")
        self.star_trail_input_files.setStyleSheet(u"")
        self.verticalLayout_3 = QVBoxLayout(self.star_trail_input_files)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_2 = QFrame(self.star_trail_input_files)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(0, 30))
        self.frame_2.setMaximumSize(QSize(16777215, 30))
        self.frame_2.setStyleSheet(u"QPushButton {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	margin:5px;\n"
"}\n"
"QPushButton:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"QPushButton:pressed {\n"
"	padding-bottom: 0px;\n"
"}\n"
"QFrame{\n"
"		border:None;\n"
"		 border-bottom: 1px solid rgba(240, 240, 240, 200);\n"
"		\n"
"    }\n"
"\n"
"#frame_top_left {\n"
"	background-color: rgba(255, 255, 255, 150);\n"
"}\n"
"")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_2)

        self.add_files = QPushButton(self.frame_2)
        self.add_files.setObjectName(u"add_files")
        self.add_files.setMinimumSize(QSize(30, 30))
        self.add_files.setMaximumSize(QSize(30, 30))
        icon5 = QIcon()
        icon5.addFile(u":/icons/resource/icon/add image.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.add_files.setIcon(icon5)
        self.add_files.setIconSize(QSize(25, 25))

        self.horizontalLayout_10.addWidget(self.add_files)

        self.add_folder = QPushButton(self.frame_2)
        self.add_folder.setObjectName(u"add_folder")
        self.add_folder.setMinimumSize(QSize(30, 30))
        self.add_folder.setMaximumSize(QSize(30, 30))
        icon6 = QIcon()
        icon6.addFile(u":/icons/resource/icon/add folder.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.add_folder.setIcon(icon6)
        self.add_folder.setIconSize(QSize(23, 23))

        self.horizontalLayout_10.addWidget(self.add_folder)

        self.clear_files = QPushButton(self.frame_2)
        self.clear_files.setObjectName(u"clear_files")
        self.clear_files.setMinimumSize(QSize(30, 30))
        self.clear_files.setMaximumSize(QSize(30, 30))
        icon7 = QIcon()
        icon7.addFile(u":/icons/resource/icon/delete 2.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.clear_files.setIcon(icon7)
        self.clear_files.setIconSize(QSize(25, 25))

        self.horizontalLayout_10.addWidget(self.clear_files)


        self.verticalLayout_3.addWidget(self.frame_2)

        self.frame_3 = QFrame(self.star_trail_input_files)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy1.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy1)
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_11 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.star_trail_file_tree = ImgTreeWidget(self.frame_3)
        self.star_trail_file_tree.headerItem().setText(0, "")
        self.star_trail_file_tree.setObjectName(u"star_trail_file_tree")
        self.star_trail_file_tree.setStyleSheet(u"/* \u8bbe\u7f6e\u5782\u76f4\u6eda\u52a8\u6761\u7684\u6574\u4f53\u6837\u5f0f */\n"
"QScrollBar:vertical {\n"
"    border: 0px solid grey;   /* \u6eda\u52a8\u6761\u7684\u8fb9\u6846*/\n"
"    background: rgba(190,190,190,0);      /* \u6eda\u52a8\u6761\u7684\u80cc\u666f\u8272 */\n"
"    width: 10px;              /* \u6eda\u52a8\u6761\u7684\u5bbd\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    margin: 10px 3px 10px 3px;    /* \u6eda\u52a8\u6761\u4e0a\u4e0b\u7684\u5916\u8fb9\u8ddd\u4e3a22\u50cf\u7d20\uff0c\u5de6\u53f3\u7684\u5916\u8fb9\u8ddd\u4e3a0\u50cf\u7d20 */\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e\u5782\u76f4\u6eda\u52a8\u6761\u6ed1\u5757\u7684\u6837\u5f0f */\n"
"QScrollBar::handle:vertical {\n"
"    background: rgba(190,190,190,100);      /* \u6ed1\u5757\uff08handle\uff09\u7684\u80cc\u666f\u989c\u8272 */\n"
"    min-height: 20px;         /* \u6ed1\u5757\u7684\u6700\u5c0f\u9ad8\u5ea6\u4e3a20\u50cf\u7d20\uff0c\u907f\u514d\u6ed1\u5757\u8fc7\u5c0f */\n"
"	border-Radius : 2px;\n"
"	margin: 0px 0px 0px 0px\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e"
                        "\u5782\u76f4\u6eda\u52a8\u6761\u5411\u4e0b\u7bad\u5934\u7684\u6837\u5f0f */\n"
"QScrollBar::add-line:vertical {\n"
"    background:rgba(190,190,190,190);      /* \u5411\u4e0b\u7bad\u5934\u7684\u80cc\u666f\u989c\u8272\u4e3a\u6a59\u8272 */\n"
"    height: 0px;             /* \u5411\u4e0b\u7bad\u5934\u7684\u9ad8\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    subcontrol-position: bottom;  /* \u5411\u4e0b\u7bad\u5934\u7684\u4f4d\u7f6e\u5728\u6eda\u52a8\u6761\u7684\u5e95\u90e8 */\n"
"    subcontrol-origin: margin;    /* \u5b50\u63a7\u4ef6\u7684\u4f4d\u7f6e\u4ece\u5916\u8fb9\u8ddd\u5f00\u59cb\u8ba1\u7b97 */\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e\u5782\u76f4\u6eda\u52a8\u6761\u5411\u4e0a\u7bad\u5934\u7684\u6837\u5f0f */\n"
"QScrollBar::sub-line:vertical {\n"
"    background: rgba(190,190,190,190);      /* \u5411\u4e0a\u7bad\u5934\u7684\u80cc\u666f\u989c\u8272\u4e3a\u6a59\u8272 */\n"
"    height: 0px;             /* \u5411\u4e0a\u7bad\u5934\u7684\u9ad8\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    subcontrol-position: top;     /* \u5411\u4e0a\u7bad"
                        "\u5934\u7684\u4f4d\u7f6e\u5728\u6eda\u52a8\u6761\u7684\u9876\u90e8 */\n"
"    subcontrol-origin: margin;    /* \u5b50\u63a7\u4ef6\u7684\u4f4d\u7f6e\u4ece\u5916\u8fb9\u8ddd\u5f00\u59cb\u8ba1\u7b97 */\n"
"}\n"
"\n"
"\n"
"/* \u8bbe\u7f6e\u6c34\u5e73\u6eda\u52a8\u6761\u7684\u6574\u4f53\u6837\u5f0f */\n"
"QScrollBar:horizontal {\n"
"    border: 2px solid grey;   /* \u6eda\u52a8\u6761\u7684\u8fb9\u6846\u989c\u8272\u4e3a\u7070\u8272\uff0c\u8fb9\u6846\u5bbd\u5ea6\u4e3a2\u50cf\u7d20 */\n"
"    background: #32CC99;      /* \u6eda\u52a8\u6761\u7684\u80cc\u666f\u8272\u4e3a\u6d45\u7eff\u8272 */\n"
"    height: 20px;             /* \u6eda\u52a8\u6761\u7684\u9ad8\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    margin: 0px 22px 0px 22px;  /* \u6eda\u52a8\u6761\u5de6\u53f3\u7684\u5916\u8fb9\u8ddd\u4e3a22\u50cf\u7d20\uff0c\u4e0a\u4e0b\u7684\u5916\u8fb9\u8ddd\u4e3a0\u50cf\u7d20 */\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e\u6c34\u5e73\u6eda\u52a8\u6761\u6ed1\u5757\u7684\u6837\u5f0f */\n"
"QScrollBar::handle:horizontal {\n"
"    background: #FF9933;   "
                        "   /* \u6ed1\u5757\uff08handle\uff09\u7684\u80cc\u666f\u989c\u8272\u4e3a\u6a59\u8272 */\n"
"    min-width: 20px;          /* \u6ed1\u5757\u7684\u6700\u5c0f\u5bbd\u5ea6\u4e3a20\u50cf\u7d20\uff0c\u907f\u514d\u6ed1\u5757\u8fc7\u5c0f */\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e\u6c34\u5e73\u6eda\u52a8\u6761\u5411\u53f3\u7bad\u5934\u7684\u6837\u5f0f */\n"
"QScrollBar::add-line:horizontal {\n"
"    background: #FF9933;      /* \u5411\u53f3\u7bad\u5934\u7684\u80cc\u666f\u989c\u8272\u4e3a\u6a59\u8272 */\n"
"    width: 20px;              /* \u5411\u53f3\u7bad\u5934\u7684\u5bbd\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    subcontrol-position: right;  /* \u5411\u53f3\u7bad\u5934\u7684\u4f4d\u7f6e\u5728\u6eda\u52a8\u6761\u7684\u53f3\u4fa7 */\n"
"    subcontrol-origin: margin;   /* \u5b50\u63a7\u4ef6\u7684\u4f4d\u7f6e\u4ece\u5916\u8fb9\u8ddd\u5f00\u59cb\u8ba1\u7b97 */\n"
"}\n"
"\n"
"/* \u8bbe\u7f6e\u6c34\u5e73\u6eda\u52a8\u6761\u5411\u5de6\u7bad\u5934\u7684\u6837\u5f0f */\n"
"QScrollBar::sub-line:horizontal {\n"
"    background: #FF9933;    "
                        "  /* \u5411\u5de6\u7bad\u5934\u7684\u80cc\u666f\u989c\u8272\u4e3a\u6a59\u8272 */\n"
"    width: 20px;              /* \u5411\u5de6\u7bad\u5934\u7684\u5bbd\u5ea6\u4e3a20\u50cf\u7d20 */\n"
"    subcontrol-position: left;   /* \u5411\u5de6\u7bad\u5934\u7684\u4f4d\u7f6e\u5728\u6eda\u52a8\u6761\u7684\u5de6\u4fa7 */\n"
"    subcontrol-origin: margin;   /* \u5b50\u63a7\u4ef6\u7684\u4f4d\u7f6e\u4ece\u5916\u8fb9\u8ddd\u5f00\u59cb\u8ba1\u7b97 */\n"
"}\n"
"\n"
"\n"
"\n"
"")

        self.horizontalLayout_11.addWidget(self.star_trail_file_tree)


        self.verticalLayout_3.addWidget(self.frame_3)

        icon8 = QIcon()
        icon8.addFile(u":/icons/resource/icon/image.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.star_trail_option_box.addTab(self.star_trail_input_files, icon8, "")
        self.star_trail_input_setting = QWidget()
        self.star_trail_input_setting.setObjectName(u"star_trail_input_setting")
        self.star_trail_input_setting.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.star_trail_input_setting.setStyleSheet(u"#star_trail_input_setting *{\n"
"	font-size: 11px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(30,30,30,200);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 500;\n"
"}\n"
"#star_trail_input_setting QLabel{\n"
"	font-size: 12px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(20,20,20,220);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 550;\n"
"}\n"
"#star_trail_input_setting QComboBox{\n"
"	font-size: 11px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(30,30,30,200);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 500;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"#star_trail_input_setting QComboBox {\n"
"    border: 0px solid rgba(199,199,199,100);    /* \u8bbe\u7f6e\u8fb9\u6846 */\n"
"    padding: 3px;              /* \u8bbe\u7f6e\u5185\u8fb9\u8ddd */\n"
"    border-radius: 3px;        /* \u8bbe\u7f6e\u5706\u89d2 */\n"
"	min-width:80px;\n"
"	max-width:80px\n"
"}"
                        "\n"
"#star_trail_input_setting QComboBox::down-arrow {\n"
"    image: url('E:/Code/python/HoshinoWeaver-main/HoshinoWeaver-main/ui/resource/icon/combobox-down-arrow.png');  /* \u66ff\u6362\u4e0b\u62c9\u7bad\u5934\u7684\u56fe\u6807 */\n"
"	margin-left:-10px;\n"
"	width: 16px; \n"
"    height: 16px;\n"
"}\n"
"\n"
"\n"
"#star_trail_input_setting QComboBox::hover {\n"
"	background-color: rgba(0, 212, 254, 10);\n"
"}\n"
"\n"
"#star_trail_input_setting QComboBox QAbstractItemView {\n"
"    background-color:  rgba(255,255,255,1);  /* \u4fee\u6539\u4e0b\u62c9\u5217\u8868\u7684\u80cc\u666f\u8272 */\n"
"    border: 0px solid rgba(199,199,199,100);       /* \u4fee\u6539\u4e0b\u62c9\u5217\u8868\u7684\u8fb9\u6846 */\n"
"\n"
"}")
        self.verticalLayout_5 = QVBoxLayout(self.star_trail_input_setting)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.star_trail_input_setting)
        self.frame.setObjectName(u"frame")
        sizePolicy1.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy1)
        self.frame.setStyleSheet(u"\n"
"#frame > QFrame{\n"
"    margin: 0px 5px 0px 5px;\n"
"	padding:10px 10px 10px 10px;\n"
"	border-bottom:1px solid rgba(199,199,199,100);\n"
"	border-radius:5px\n"
"}")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.frame_algorithm_startrail = QFrame(self.frame)
        self.frame_algorithm_startrail.setObjectName(u"frame_algorithm_startrail")
        sizePolicy1.setHeightForWidth(self.frame_algorithm_startrail.sizePolicy().hasHeightForWidth())
        self.frame_algorithm_startrail.setSizePolicy(sizePolicy1)
        self.frame_algorithm_startrail.setMinimumSize(QSize(0, 40))
        self.frame_algorithm_startrail.setMaximumSize(QSize(16777215, 40))
        self.frame_algorithm_startrail.setStyleSheet(u"")
        self.frame_algorithm_startrail.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_algorithm_startrail.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_18 = QHBoxLayout(self.frame_algorithm_startrail)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.label_19 = QLabel(self.frame_algorithm_startrail)
        self.label_19.setObjectName(u"label_19")
        sizePolicy1.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy1)
        self.label_19.setMinimumSize(QSize(60, 20))
        self.label_19.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_18.addWidget(self.label_19)

        self.alter_algorithm_startrail = QComboBox(self.frame_algorithm_startrail)
        self.alter_algorithm_startrail.addItem("")
        self.alter_algorithm_startrail.addItem("")
        self.alter_algorithm_startrail.setObjectName(u"alter_algorithm_startrail")
        sizePolicy1.setHeightForWidth(self.alter_algorithm_startrail.sizePolicy().hasHeightForWidth())
        self.alter_algorithm_startrail.setSizePolicy(sizePolicy1)
        self.alter_algorithm_startrail.setMinimumSize(QSize(86, 20))
        self.alter_algorithm_startrail.setMaximumSize(QSize(86, 20))
        self.alter_algorithm_startrail.setStyleSheet(u"")
        self.alter_algorithm_startrail.setIconSize(QSize(40, 40))

        self.horizontalLayout_18.addWidget(self.alter_algorithm_startrail)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_14)


        self.verticalLayout_7.addWidget(self.frame_algorithm_startrail)

        self.frame_algorithm_mean = QFrame(self.frame)
        self.frame_algorithm_mean.setObjectName(u"frame_algorithm_mean")
        sizePolicy1.setHeightForWidth(self.frame_algorithm_mean.sizePolicy().hasHeightForWidth())
        self.frame_algorithm_mean.setSizePolicy(sizePolicy1)
        self.frame_algorithm_mean.setMinimumSize(QSize(0, 40))
        self.frame_algorithm_mean.setMaximumSize(QSize(16777215, 40))
        self.frame_algorithm_mean.setStyleSheet(u"")
        self.frame_algorithm_mean.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_algorithm_mean.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_17 = QHBoxLayout(self.frame_algorithm_mean)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_16 = QLabel(self.frame_algorithm_mean)
        self.label_16.setObjectName(u"label_16")
        sizePolicy1.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy1)
        self.label_16.setMinimumSize(QSize(60, 20))
        self.label_16.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_17.addWidget(self.label_16)

        self.alter_algorithm_mean = QComboBox(self.frame_algorithm_mean)
        self.alter_algorithm_mean.addItem("")
        self.alter_algorithm_mean.addItem("")
        self.alter_algorithm_mean.setObjectName(u"alter_algorithm_mean")
        sizePolicy1.setHeightForWidth(self.alter_algorithm_mean.sizePolicy().hasHeightForWidth())
        self.alter_algorithm_mean.setSizePolicy(sizePolicy1)
        self.alter_algorithm_mean.setMinimumSize(QSize(86, 20))
        self.alter_algorithm_mean.setMaximumSize(QSize(86, 20))
        self.alter_algorithm_mean.setStyleSheet(u"")
        self.alter_algorithm_mean.setIconSize(QSize(40, 40))

        self.horizontalLayout_17.addWidget(self.alter_algorithm_mean)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_16)


        self.verticalLayout_7.addWidget(self.frame_algorithm_mean)

        self.frame_algorithm_min = QFrame(self.frame)
        self.frame_algorithm_min.setObjectName(u"frame_algorithm_min")
        sizePolicy1.setHeightForWidth(self.frame_algorithm_min.sizePolicy().hasHeightForWidth())
        self.frame_algorithm_min.setSizePolicy(sizePolicy1)
        self.frame_algorithm_min.setMinimumSize(QSize(0, 40))
        self.frame_algorithm_min.setMaximumSize(QSize(16777215, 40))
        self.frame_algorithm_min.setStyleSheet(u"")
        self.frame_algorithm_min.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_algorithm_min.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_19 = QHBoxLayout(self.frame_algorithm_min)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.label_20 = QLabel(self.frame_algorithm_min)
        self.label_20.setObjectName(u"label_20")
        sizePolicy1.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy1)
        self.label_20.setMinimumSize(QSize(60, 20))
        self.label_20.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_19.addWidget(self.label_20)

        self.alter_algorithm_min = QComboBox(self.frame_algorithm_min)
        self.alter_algorithm_min.addItem("")
        self.alter_algorithm_min.setObjectName(u"alter_algorithm_min")
        sizePolicy1.setHeightForWidth(self.alter_algorithm_min.sizePolicy().hasHeightForWidth())
        self.alter_algorithm_min.setSizePolicy(sizePolicy1)
        self.alter_algorithm_min.setMinimumSize(QSize(56, 20))
        self.alter_algorithm_min.setMaximumSize(QSize(56, 20))
        self.alter_algorithm_min.setStyleSheet(u"QComboBox {\n"
"	min-width:50px;\n"
"	max-width:50px\n"
"}")
        self.alter_algorithm_min.setIconSize(QSize(40, 40))

        self.horizontalLayout_19.addWidget(self.alter_algorithm_min)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_17)


        self.verticalLayout_7.addWidget(self.frame_algorithm_min)

        self.frame_mask = QFrame(self.frame)
        self.frame_mask.setObjectName(u"frame_mask")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.frame_mask.sizePolicy().hasHeightForWidth())
        self.frame_mask.setSizePolicy(sizePolicy3)
        self.frame_mask.setMinimumSize(QSize(0, 60))
        self.frame_mask.setMaximumSize(QSize(16777215, 60))
        self.frame_mask.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_mask.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_mask)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.frame_mask_sub1 = QFrame(self.frame_mask)
        self.frame_mask_sub1.setObjectName(u"frame_mask_sub1")
        self.frame_mask_sub1.setMinimumSize(QSize(0, 20))
        self.frame_mask_sub1.setMaximumSize(QSize(16777215, 20))
        self.frame_mask_sub1.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_mask_sub1.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_24 = QHBoxLayout(self.frame_mask_sub1)
        self.horizontalLayout_24.setSpacing(0)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.horizontalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.label_mask = QLabel(self.frame_mask_sub1)
        self.label_mask.setObjectName(u"label_mask")
        sizePolicy1.setHeightForWidth(self.label_mask.sizePolicy().hasHeightForWidth())
        self.label_mask.setSizePolicy(sizePolicy1)
        self.label_mask.setMinimumSize(QSize(60, 20))
        self.label_mask.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_24.addWidget(self.label_mask)

        self.mask_able = QCheckBox(self.frame_mask_sub1)
        self.mask_able.setObjectName(u"mask_able")

        self.horizontalLayout_24.addWidget(self.mask_able)

        self.horizontalSpacer_mask_1 = QSpacerItem(85, 15, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_24.addItem(self.horizontalSpacer_mask_1)


        self.verticalLayout_8.addWidget(self.frame_mask_sub1)

        self.frame_mask_sub2 = QFrame(self.frame_mask)
        self.frame_mask_sub2.setObjectName(u"frame_mask_sub2")
        self.frame_mask_sub2.setMinimumSize(QSize(0, 20))
        self.frame_mask_sub2.setMaximumSize(QSize(16777215, 20))
        self.frame_mask_sub2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_mask_sub2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_23 = QHBoxLayout(self.frame_mask_sub2)
        self.horizontalLayout_23.setSpacing(0)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.horizontalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.mask_file_path = QLineEdit(self.frame_mask_sub2)
        self.mask_file_path.setObjectName(u"mask_file_path")
        self.mask_file_path.setMinimumSize(QSize(160, 20))
        self.mask_file_path.setMaximumSize(QSize(160, 20))
        self.mask_file_path.setStyleSheet(u"QLineEdit {\n"
"	border:1px solid  rgba(220,220,220,200);\n"
"	border-radius:3px;\n"
"	background-color: rgba(250,250,250,200);\n"
"	margin-left:10px\n"
"}")

        self.horizontalLayout_23.addWidget(self.mask_file_path)

        self.alter_mask_file = QPushButton(self.frame_mask_sub2)
        self.alter_mask_file.setObjectName(u"alter_mask_file")
        sizePolicy1.setHeightForWidth(self.alter_mask_file.sizePolicy().hasHeightForWidth())
        self.alter_mask_file.setSizePolicy(sizePolicy1)
        self.alter_mask_file.setMinimumSize(QSize(25, 20))
        self.alter_mask_file.setMaximumSize(QSize(25, 20))
        self.alter_mask_file.setStyleSheet(u"QPushButton {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	padding-left:5px;\n"
"}\n"
"QPushButton:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"QPushButton:pressed {\n"
"	padding-bottom: 0px;\n"
"}")
        icon9 = QIcon()
        icon9.addFile(u":/icons/resource/icon/choose_file.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.alter_mask_file.setIcon(icon9)
        self.alter_mask_file.setIconSize(QSize(20, 20))

        self.horizontalLayout_23.addWidget(self.alter_mask_file)

        self.horizontalSpacer_mask_2 = QSpacerItem(1, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_23.addItem(self.horizontalSpacer_mask_2)


        self.verticalLayout_8.addWidget(self.frame_mask_sub2)


        self.verticalLayout_7.addWidget(self.frame_mask)

        self.frame_max_iter = QFrame(self.frame)
        self.frame_max_iter.setObjectName(u"frame_max_iter")
        sizePolicy1.setHeightForWidth(self.frame_max_iter.sizePolicy().hasHeightForWidth())
        self.frame_max_iter.setSizePolicy(sizePolicy1)
        self.frame_max_iter.setMinimumSize(QSize(0, 60))
        self.frame_max_iter.setMaximumSize(QSize(16777215, 60))
        self.frame_max_iter.setStyleSheet(u"QFrame {\n"
"	width : 0px\n"
"}")
        self.frame_max_iter.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_max_iter.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_10 = QGridLayout(self.frame_max_iter)
        self.gridLayout_10.setSpacing(0)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_max_iter = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_10.addItem(self.horizontalSpacer_max_iter, 0, 2, 1, 1)

        self.label_max_iter = QLabel(self.frame_max_iter)
        self.label_max_iter.setObjectName(u"label_max_iter")
        self.label_max_iter.setMinimumSize(QSize(80, 20))
        self.label_max_iter.setMaximumSize(QSize(80, 20))

        self.gridLayout_10.addWidget(self.label_max_iter, 0, 0, 1, 1)

        self.alter_max_iter = QSlider(self.frame_max_iter)
        self.alter_max_iter.setObjectName(u"alter_max_iter")
        self.alter_max_iter.setMinimumSize(QSize(180, 20))
        self.alter_max_iter.setMaximumSize(QSize(180, 20))
        self.alter_max_iter.setStyleSheet(u"")
        self.alter_max_iter.setMinimum(1)
        self.alter_max_iter.setMaximum(10)
        self.alter_max_iter.setPageStep(1)
        self.alter_max_iter.setValue(5)
        self.alter_max_iter.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_10.addWidget(self.alter_max_iter, 1, 0, 1, 3)

        self.max_iter = QLabel(self.frame_max_iter)
        self.max_iter.setObjectName(u"max_iter")
        self.max_iter.setMinimumSize(QSize(30, 20))
        self.max_iter.setMaximumSize(QSize(30, 20))
        self.max_iter.setStyleSheet(u"QLabel {\n"
"	padding: 3px\n"
"}\n"
"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.gridLayout_10.addWidget(self.max_iter, 0, 1, 1, 1)


        self.verticalLayout_7.addWidget(self.frame_max_iter)

        self.frame_rejection = QFrame(self.frame)
        self.frame_rejection.setObjectName(u"frame_rejection")
        sizePolicy1.setHeightForWidth(self.frame_rejection.sizePolicy().hasHeightForWidth())
        self.frame_rejection.setSizePolicy(sizePolicy1)
        self.frame_rejection.setMinimumSize(QSize(0, 80))
        self.frame_rejection.setMaximumSize(QSize(16777215, 80))
        self.frame_rejection.setStyleSheet(u"#frame_rejection * {\n"
"	margin: 0px 0px 0px 0px ;\n"
"	padding:0px 0px 0px 0px ;\n"
"}")
        self.frame_rejection.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_rejection.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_rejection)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.frame_rejection_sub_1 = QFrame(self.frame_rejection)
        self.frame_rejection_sub_1.setObjectName(u"frame_rejection_sub_1")
        self.frame_rejection_sub_1.setMinimumSize(QSize(0, 20))
        self.frame_rejection_sub_1.setMaximumSize(QSize(16777215, 20))
        self.frame_rejection_sub_1.setStyleSheet(u"")
        self.frame_rejection_sub_1.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_rejection_sub_1.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_25 = QHBoxLayout(self.frame_rejection_sub_1)
        self.horizontalLayout_25.setSpacing(0)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalLayout_25.setContentsMargins(0, 0, 0, 0)
        self.label_28 = QLabel(self.frame_rejection_sub_1)
        self.label_28.setObjectName(u"label_28")
        sizePolicy1.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy1)
        self.label_28.setMinimumSize(QSize(60, 20))
        self.label_28.setMaximumSize(QSize(100000, 20))

        self.horizontalLayout_25.addWidget(self.label_28)


        self.verticalLayout_9.addWidget(self.frame_rejection_sub_1)

        self.frame_rejection_sub_2 = QFrame(self.frame_rejection)
        self.frame_rejection_sub_2.setObjectName(u"frame_rejection_sub_2")
        self.frame_rejection_sub_2.setMinimumSize(QSize(0, 20))
        self.frame_rejection_sub_2.setMaximumSize(QSize(16777215, 20))
        self.frame_rejection_sub_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_rejection_sub_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_26 = QHBoxLayout(self.frame_rejection_sub_2)
        self.horizontalLayout_26.setSpacing(0)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.horizontalLayout_26.setContentsMargins(0, 0, 0, 0)
        self.label_rejection_low = QLabel(self.frame_rejection_sub_2)
        self.label_rejection_low.setObjectName(u"label_rejection_low")
        self.label_rejection_low.setMinimumSize(QSize(30, 0))
        self.label_rejection_low.setMaximumSize(QSize(30, 20))
        self.label_rejection_low.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_26.addWidget(self.label_rejection_low)

        self.rejection_low = QLabel(self.frame_rejection_sub_2)
        self.rejection_low.setObjectName(u"rejection_low")
        self.rejection_low.setMinimumSize(QSize(35, 20))
        self.rejection_low.setMaximumSize(QSize(35, 20))
        self.rejection_low.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_26.addWidget(self.rejection_low)

        self.horizontalSpacer_rejection = QSpacerItem(10000, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_26.addItem(self.horizontalSpacer_rejection)

        self.rejection_high = QLabel(self.frame_rejection_sub_2)
        self.rejection_high.setObjectName(u"rejection_high")
        self.rejection_high.setMinimumSize(QSize(35, 20))
        self.rejection_high.setMaximumSize(QSize(35, 20))
        self.rejection_high.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_26.addWidget(self.rejection_high)

        self.label_rejection_high = QLabel(self.frame_rejection_sub_2)
        self.label_rejection_high.setObjectName(u"label_rejection_high")
        self.label_rejection_high.setMinimumSize(QSize(30, 20))
        self.label_rejection_high.setMaximumSize(QSize(30, 20))
        self.label_rejection_high.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_26.addWidget(self.label_rejection_high)


        self.verticalLayout_9.addWidget(self.frame_rejection_sub_2)

        self.frame_rejection_sub_3 = QFrame(self.frame_rejection)
        self.frame_rejection_sub_3.setObjectName(u"frame_rejection_sub_3")
        self.frame_rejection_sub_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_rejection_sub_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_30 = QHBoxLayout(self.frame_rejection_sub_3)
        self.horizontalLayout_30.setSpacing(0)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.horizontalLayout_30.setContentsMargins(0, 0, 0, 0)
        self.alter_rejection = DoubleSlider(self.frame_rejection_sub_3)
        self.alter_rejection.setObjectName(u"alter_rejection")
        sizePolicy.setHeightForWidth(self.alter_rejection.sizePolicy().hasHeightForWidth())
        self.alter_rejection.setSizePolicy(sizePolicy)
        self.alter_rejection.setMinimumSize(QSize(160, 20))
        self.alter_rejection.setMaximumSize(QSize(160, 20))
        self.alter_rejection.setFrameShape(QFrame.Shape.StyledPanel)
        self.alter_rejection.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_30.addWidget(self.alter_rejection)


        self.verticalLayout_9.addWidget(self.frame_rejection_sub_3)


        self.verticalLayout_7.addWidget(self.frame_rejection)

        self.frame_fade_in_out = QFrame(self.frame)
        self.frame_fade_in_out.setObjectName(u"frame_fade_in_out")
        sizePolicy1.setHeightForWidth(self.frame_fade_in_out.sizePolicy().hasHeightForWidth())
        self.frame_fade_in_out.setSizePolicy(sizePolicy1)
        self.frame_fade_in_out.setMinimumSize(QSize(0, 80))
        self.frame_fade_in_out.setMaximumSize(QSize(16777215, 80))
        self.frame_fade_in_out.setStyleSheet(u"")
        self.frame_fade_in_out.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_fade_in_out.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.frame_fade_in_out)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.frame_fade_in_out_sub_1 = QFrame(self.frame_fade_in_out)
        self.frame_fade_in_out_sub_1.setObjectName(u"frame_fade_in_out_sub_1")
        self.frame_fade_in_out_sub_1.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_fade_in_out_sub_1.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_28 = QHBoxLayout(self.frame_fade_in_out_sub_1)
        self.horizontalLayout_28.setSpacing(0)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.horizontalLayout_28.setContentsMargins(0, 0, 0, 0)
        self.lable_fade = QLabel(self.frame_fade_in_out_sub_1)
        self.lable_fade.setObjectName(u"lable_fade")
        self.lable_fade.setMinimumSize(QSize(60, 20))
        self.lable_fade.setMaximumSize(QSize(1800, 20))

        self.horizontalLayout_28.addWidget(self.lable_fade)


        self.verticalLayout_10.addWidget(self.frame_fade_in_out_sub_1)

        self.frame_fade_in_out_sub_2 = QFrame(self.frame_fade_in_out)
        self.frame_fade_in_out_sub_2.setObjectName(u"frame_fade_in_out_sub_2")
        self.frame_fade_in_out_sub_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_fade_in_out_sub_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_29 = QHBoxLayout(self.frame_fade_in_out_sub_2)
        self.horizontalLayout_29.setSpacing(0)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.lable_fade_in = QLabel(self.frame_fade_in_out_sub_2)
        self.lable_fade_in.setObjectName(u"lable_fade_in")
        self.lable_fade_in.setMinimumSize(QSize(30, 0))
        self.lable_fade_in.setMaximumSize(QSize(30, 20))
        self.lable_fade_in.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_29.addWidget(self.lable_fade_in)

        self.fade_in = QLabel(self.frame_fade_in_out_sub_2)
        self.fade_in.setObjectName(u"fade_in")
        self.fade_in.setMinimumSize(QSize(35, 20))
        self.fade_in.setMaximumSize(QSize(30, 20))
        self.fade_in.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_29.addWidget(self.fade_in)

        self.horizontalSpacer_fade = QSpacerItem(56, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_fade)

        self.fade_out = QLabel(self.frame_fade_in_out_sub_2)
        self.fade_out.setObjectName(u"fade_out")
        self.fade_out.setMinimumSize(QSize(35, 20))
        self.fade_out.setMaximumSize(QSize(35, 20))
        self.fade_out.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_29.addWidget(self.fade_out)

        self.lable_fade_out = QLabel(self.frame_fade_in_out_sub_2)
        self.lable_fade_out.setObjectName(u"lable_fade_out")
        self.lable_fade_out.setMinimumSize(QSize(30, 20))
        self.lable_fade_out.setMaximumSize(QSize(30, 20))
        self.lable_fade_out.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.horizontalLayout_29.addWidget(self.lable_fade_out)


        self.verticalLayout_10.addWidget(self.frame_fade_in_out_sub_2)

        self.frame_fade_in_out_sub_3 = QFrame(self.frame_fade_in_out)
        self.frame_fade_in_out_sub_3.setObjectName(u"frame_fade_in_out_sub_3")
        self.frame_fade_in_out_sub_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_fade_in_out_sub_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_27 = QHBoxLayout(self.frame_fade_in_out_sub_3)
        self.horizontalLayout_27.setSpacing(0)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.alter_fade_in_out = DoubleSlider(self.frame_fade_in_out_sub_3)
        self.alter_fade_in_out.setObjectName(u"alter_fade_in_out")
        sizePolicy.setHeightForWidth(self.alter_fade_in_out.sizePolicy().hasHeightForWidth())
        self.alter_fade_in_out.setSizePolicy(sizePolicy)
        self.alter_fade_in_out.setMinimumSize(QSize(160, 20))
        self.alter_fade_in_out.setMaximumSize(QSize(160, 20))
        self.alter_fade_in_out.setFrameShape(QFrame.Shape.StyledPanel)
        self.alter_fade_in_out.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_27.addWidget(self.alter_fade_in_out)


        self.verticalLayout_10.addWidget(self.frame_fade_in_out_sub_3)


        self.verticalLayout_7.addWidget(self.frame_fade_in_out)

        self.frame_int_weight = QFrame(self.frame)
        self.frame_int_weight.setObjectName(u"frame_int_weight")
        sizePolicy1.setHeightForWidth(self.frame_int_weight.sizePolicy().hasHeightForWidth())
        self.frame_int_weight.setSizePolicy(sizePolicy1)
        self.frame_int_weight.setMinimumSize(QSize(0, 40))
        self.frame_int_weight.setMaximumSize(QSize(16777215, 40))
        self.frame_int_weight.setStyleSheet(u"")
        self.frame_int_weight.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_int_weight.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.frame_int_weight)
        self.horizontalLayout_20.setSpacing(0)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.label_frame_int_weight = QLabel(self.frame_int_weight)
        self.label_frame_int_weight.setObjectName(u"label_frame_int_weight")
        sizePolicy1.setHeightForWidth(self.label_frame_int_weight.sizePolicy().hasHeightForWidth())
        self.label_frame_int_weight.setSizePolicy(sizePolicy1)
        self.label_frame_int_weight.setMinimumSize(QSize(60, 20))
        self.label_frame_int_weight.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_20.addWidget(self.label_frame_int_weight)

        self.int_weight_able = QCheckBox(self.frame_int_weight)
        self.int_weight_able.setObjectName(u"int_weight_able")

        self.horizontalLayout_20.addWidget(self.int_weight_able)

        self.horizontalSpacer_int_weight = QSpacerItem(20, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_int_weight)


        self.verticalLayout_7.addWidget(self.frame_int_weight)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)


        self.verticalLayout_5.addWidget(self.frame)

        icon10 = QIcon()
        icon10.addFile(u":/icons/resource/icon/setting.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.star_trail_option_box.addTab(self.star_trail_input_setting, icon10, "")
        self.star_trail_output = QWidget()
        self.star_trail_output.setObjectName(u"star_trail_output")
        self.star_trail_output.setStyleSheet(u"#star_trail_output *{\n"
"	font-size: 11px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(30,30,30,200);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 500;\n"
"}\n"
"#star_trail_output QLabel{\n"
"	font-size: 12px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(20,20,20,220);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 550;\n"
"}\n"
"#star_trail_output QComboBox{\n"
"	font-size: 11px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(30,30,30,200);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 500;\n"
"}\n"
"\n"
"#star_trail_output > QFrame{\n"
"    margin: 0px 5px 0px 5px;\n"
"	padding:10px 10px 10px 10px;\n"
"	border-bottom:1px solid rgba(199,199,199,100);\n"
"	border-radius:5px\n"
"}\n"
"\n"
"#star_trail_output QComboBox {\n"
"    border: 0px solid rgba(199,199,199,100);    /* \u8bbe\u7f6e\u8fb9\u6846 */\n"
"    padding: 3px;              /* \u8bbe"
                        "\u7f6e\u5185\u8fb9\u8ddd */\n"
"    border-radius: 3px;        /* \u8bbe\u7f6e\u5706\u89d2 */\n"
"	min-width:50px;\n"
"	max-width:50px\n"
"}\n"
"#star_trail_output QComboBox::down-arrow {\n"
"    image: url('E:/Code/python/HoshinoWeaver-main/HoshinoWeaver-main/ui/resource/icon/combobox-down-arrow.png');  /* \u66ff\u6362\u4e0b\u62c9\u7bad\u5934\u7684\u56fe\u6807 */\n"
"	margin-left:-10px;\n"
"	min-width:0px;\n"
"	max-width:0px\n"
"}\n"
"#star_trail_output QComboBox::hover {\n"
"	background-color: rgba(0, 212, 254, 10);\n"
"}\n"
"\n"
"#star_trail_output QComboBox QAbstractItemView {\n"
"    background-color:  rgba(255,255,255,1);  /* \u4fee\u6539\u4e0b\u62c9\u5217\u8868\u7684\u80cc\u666f\u8272 */\n"
"    border: 0px solid rgba(199,199,199,100);       /* \u4fee\u6539\u4e0b\u62c9\u5217\u8868\u7684\u8fb9\u6846 */\n"
"\n"
"}")
        self.verticalLayout_4 = QVBoxLayout(self.star_trail_output)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.frame_output_type = QFrame(self.star_trail_output)
        self.frame_output_type.setObjectName(u"frame_output_type")
        sizePolicy1.setHeightForWidth(self.frame_output_type.sizePolicy().hasHeightForWidth())
        self.frame_output_type.setSizePolicy(sizePolicy1)
        self.frame_output_type.setMinimumSize(QSize(0, 40))
        self.frame_output_type.setMaximumSize(QSize(16777215, 40))
        self.frame_output_type.setStyleSheet(u"")
        self.frame_output_type.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_output_type.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_output_type)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.label_output_type = QLabel(self.frame_output_type)
        self.label_output_type.setObjectName(u"label_output_type")
        sizePolicy1.setHeightForWidth(self.label_output_type.sizePolicy().hasHeightForWidth())
        self.label_output_type.setSizePolicy(sizePolicy1)
        self.label_output_type.setMinimumSize(QSize(60, 20))
        self.label_output_type.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_14.addWidget(self.label_output_type)

        self.alter_output_type_2 = QComboBox(self.frame_output_type)
        self.alter_output_type_2.addItem("")
        self.alter_output_type_2.addItem("")
        self.alter_output_type_2.addItem("")
        self.alter_output_type_2.setObjectName(u"alter_output_type_2")
        sizePolicy1.setHeightForWidth(self.alter_output_type_2.sizePolicy().hasHeightForWidth())
        self.alter_output_type_2.setSizePolicy(sizePolicy1)
        self.alter_output_type_2.setMinimumSize(QSize(56, 20))
        self.alter_output_type_2.setMaximumSize(QSize(56, 20))
        self.alter_output_type_2.setStyleSheet(u"")
        self.alter_output_type_2.setIconSize(QSize(40, 40))

        self.horizontalLayout_14.addWidget(self.alter_output_type_2)

        self.horizontalSpacer_output_type = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_output_type)


        self.verticalLayout_4.addWidget(self.frame_output_type)

        self.frame_output_path = QFrame(self.star_trail_output)
        self.frame_output_path.setObjectName(u"frame_output_path")
        sizePolicy1.setHeightForWidth(self.frame_output_path.sizePolicy().hasHeightForWidth())
        self.frame_output_path.setSizePolicy(sizePolicy1)
        self.frame_output_path.setMinimumSize(QSize(0, 60))
        self.frame_output_path.setMaximumSize(QSize(16777215, 60))
        self.frame_output_path.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_output_path.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_2 = QGridLayout(self.frame_output_path)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.output_path_2 = QLineEdit(self.frame_output_path)
        self.output_path_2.setObjectName(u"output_path_2")
        self.output_path_2.setMinimumSize(QSize(160, 20))
        self.output_path_2.setMaximumSize(QSize(160, 20))
        self.output_path_2.setStyleSheet(u"QLineEdit {\n"
"	border:1px solid  rgba(220,220,220,200);\n"
"	border-radius:3px;\n"
"	background-color: rgba(250,250,250,200);\n"
"	margin-left:10px\n"
"}")

        self.gridLayout_2.addWidget(self.output_path_2, 1, 0, 1, 3)

        self.label_output_path = QLabel(self.frame_output_path)
        self.label_output_path.setObjectName(u"label_output_path")
        sizePolicy1.setHeightForWidth(self.label_output_path.sizePolicy().hasHeightForWidth())
        self.label_output_path.setSizePolicy(sizePolicy1)
        self.label_output_path.setMinimumSize(QSize(60, 20))
        self.label_output_path.setMaximumSize(QSize(60, 20))

        self.gridLayout_2.addWidget(self.label_output_path, 0, 0, 1, 1)

        self.alter_output_2 = QPushButton(self.frame_output_path)
        self.alter_output_2.setObjectName(u"alter_output_2")
        sizePolicy1.setHeightForWidth(self.alter_output_2.sizePolicy().hasHeightForWidth())
        self.alter_output_2.setSizePolicy(sizePolicy1)
        self.alter_output_2.setMinimumSize(QSize(25, 20))
        self.alter_output_2.setMaximumSize(QSize(25, 20))
        self.alter_output_2.setStyleSheet(u"QPushButton {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	padding-left:5px;\n"
"}\n"
"QPushButton:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"QPushButton:pressed {\n"
"	padding-bottom: 0px;\n"
"}")
        self.alter_output_2.setIcon(icon9)
        self.alter_output_2.setIconSize(QSize(20, 20))

        self.gridLayout_2.addWidget(self.alter_output_2, 1, 3, 1, 1)

        self.horizontalSpacer_output_path_1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_output_path_1, 1, 4, 1, 1)

        self.horizontalSpacer_output_path_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_output_path_2, 0, 1, 1, 4)


        self.verticalLayout_4.addWidget(self.frame_output_path)

        self.frame_png_level = QFrame(self.star_trail_output)
        self.frame_png_level.setObjectName(u"frame_png_level")
        sizePolicy1.setHeightForWidth(self.frame_png_level.sizePolicy().hasHeightForWidth())
        self.frame_png_level.setSizePolicy(sizePolicy1)
        self.frame_png_level.setMinimumSize(QSize(0, 60))
        self.frame_png_level.setMaximumSize(QSize(16777215, 60))
        self.frame_png_level.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_png_level.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_8 = QGridLayout(self.frame_png_level)
        self.gridLayout_8.setSpacing(0)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.label_png_level = QLabel(self.frame_png_level)
        self.label_png_level.setObjectName(u"label_png_level")
        self.label_png_level.setMinimumSize(QSize(60, 20))
        self.label_png_level.setMaximumSize(QSize(60, 20))

        self.gridLayout_8.addWidget(self.label_png_level, 0, 0, 1, 1)

        self.png_level = QLabel(self.frame_png_level)
        self.png_level.setObjectName(u"png_level")
        self.png_level.setMinimumSize(QSize(30, 20))
        self.png_level.setMaximumSize(QSize(30, 20))
        self.png_level.setStyleSheet(u"QLabel {\n"
"	padding: 3px\n"
"}\n"
"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.gridLayout_8.addWidget(self.png_level, 0, 1, 1, 1)

        self.horizontalSpacer_png_level_1 = QSpacerItem(114, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_png_level_1, 0, 2, 1, 2)

        self.horizontalSpacer_png_level_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_png_level_2, 1, 3, 1, 1)

        self.alter_png_level = QSlider(self.frame_png_level)
        self.alter_png_level.setObjectName(u"alter_png_level")
        self.alter_png_level.setMinimumSize(QSize(180, 20))
        self.alter_png_level.setMaximumSize(QSize(180, 20))
        self.alter_png_level.setStyleSheet(u"")
        self.alter_png_level.setMinimum(1)
        self.alter_png_level.setMaximum(9)
        self.alter_png_level.setPageStep(1)
        self.alter_png_level.setValue(7)
        self.alter_png_level.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.alter_png_level, 1, 0, 1, 3)


        self.verticalLayout_4.addWidget(self.frame_png_level)

        self.frame_jpg_level = QFrame(self.star_trail_output)
        self.frame_jpg_level.setObjectName(u"frame_jpg_level")
        sizePolicy1.setHeightForWidth(self.frame_jpg_level.sizePolicy().hasHeightForWidth())
        self.frame_jpg_level.setSizePolicy(sizePolicy1)
        self.frame_jpg_level.setMinimumSize(QSize(0, 60))
        self.frame_jpg_level.setMaximumSize(QSize(16777215, 60))
        self.frame_jpg_level.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_jpg_level.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_4 = QGridLayout(self.frame_jpg_level)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.jpg_level = QLabel(self.frame_jpg_level)
        self.jpg_level.setObjectName(u"jpg_level")
        self.jpg_level.setMinimumSize(QSize(30, 20))
        self.jpg_level.setMaximumSize(QSize(30, 20))
        self.jpg_level.setStyleSheet(u"QLabel {\n"
"	padding: 3px\n"
"}\n"
"\n"
"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}")

        self.gridLayout_4.addWidget(self.jpg_level, 0, 1, 1, 1)

        self.alter_jpg_level = QSlider(self.frame_jpg_level)
        self.alter_jpg_level.setObjectName(u"alter_jpg_level")
        self.alter_jpg_level.setMinimumSize(QSize(180, 20))
        self.alter_jpg_level.setMaximumSize(QSize(180, 20))
        self.alter_jpg_level.setStyleSheet(u"")
        self.alter_jpg_level.setMinimum(1)
        self.alter_jpg_level.setMaximum(100)
        self.alter_jpg_level.setPageStep(10)
        self.alter_jpg_level.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_4.addWidget(self.alter_jpg_level, 1, 0, 1, 3)

        self.label_jpg_level = QLabel(self.frame_jpg_level)
        self.label_jpg_level.setObjectName(u"label_jpg_level")
        self.label_jpg_level.setMinimumSize(QSize(60, 20))
        self.label_jpg_level.setMaximumSize(QSize(60, 20))

        self.gridLayout_4.addWidget(self.label_jpg_level, 0, 0, 1, 1)

        self.horizontalSpacer_jpg_level_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_jpg_level_2, 1, 3, 1, 1)

        self.horizontalSpacer_jpg_level_1 = QSpacerItem(97, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer_jpg_level_1, 0, 2, 1, 2)


        self.verticalLayout_4.addWidget(self.frame_jpg_level)

        self.frame_output_bits = QFrame(self.star_trail_output)
        self.frame_output_bits.setObjectName(u"frame_output_bits")
        sizePolicy1.setHeightForWidth(self.frame_output_bits.sizePolicy().hasHeightForWidth())
        self.frame_output_bits.setSizePolicy(sizePolicy1)
        self.frame_output_bits.setMinimumSize(QSize(0, 40))
        self.frame_output_bits.setMaximumSize(QSize(16777215, 40))
        self.frame_output_bits.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_output_bits.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_16 = QHBoxLayout(self.frame_output_bits)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.label_output_bits = QLabel(self.frame_output_bits)
        self.label_output_bits.setObjectName(u"label_output_bits")
        self.label_output_bits.setMinimumSize(QSize(60, 20))
        self.label_output_bits.setMaximumSize(QSize(60, 20))

        self.horizontalLayout_16.addWidget(self.label_output_bits)

        self.alter_output_bits = QComboBox(self.frame_output_bits)
        self.alter_output_bits.addItem("")
        self.alter_output_bits.addItem("")
        self.alter_output_bits.addItem("")
        self.alter_output_bits.setObjectName(u"alter_output_bits")
        self.alter_output_bits.setMinimumSize(QSize(56, 20))
        self.alter_output_bits.setMaximumSize(QSize(56, 20))
        self.alter_output_bits.setStyleSheet(u"")

        self.horizontalLayout_16.addWidget(self.alter_output_bits)

        self.horizontalSpacer_output_bits = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_output_bits)


        self.verticalLayout_4.addWidget(self.frame_output_bits)

        self.frame_resize = QFrame(self.star_trail_output)
        self.frame_resize.setObjectName(u"frame_resize")
        sizePolicy1.setHeightForWidth(self.frame_resize.sizePolicy().hasHeightForWidth())
        self.frame_resize.setSizePolicy(sizePolicy1)
        self.frame_resize.setMinimumSize(QSize(0, 100))
        self.frame_resize.setMaximumSize(QSize(16777215, 100))
        self.frame_resize.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_resize.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_5 = QGridLayout(self.frame_resize)
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_resize_1 = QSpacerItem(27, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_resize_1, 0, 2, 1, 1)

        self.alter_resize = QSlider(self.frame_resize)
        self.alter_resize.setObjectName(u"alter_resize")
        self.alter_resize.setMinimumSize(QSize(180, 0))
        self.alter_resize.setMaximumSize(QSize(180, 16777215))
        self.alter_resize.setMinimum(1)
        self.alter_resize.setMaximum(100)
        self.alter_resize.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.alter_resize, 3, 0, 1, 3)

        self.resize_x_y = QLabel(self.frame_resize)
        self.resize_x_y.setObjectName(u"resize_x_y")
        self.resize_x_y.setMinimumSize(QSize(100, 20))
        self.resize_x_y.setMaximumSize(QSize(100, 20))

        self.gridLayout_5.addWidget(self.resize_x_y, 0, 1, 1, 1)

        self.horizontalSpacer_resize_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_resize_2, 1, 2, 1, 1)

        self.resize_x = QLabel(self.frame_resize)
        self.resize_x.setObjectName(u"resize_x")
        self.resize_x.setMinimumSize(QSize(100, 0))
        self.resize_x.setMaximumSize(QSize(100, 20))

        self.gridLayout_5.addWidget(self.resize_x, 1, 1, 1, 1)

        self.label_resize = QLabel(self.frame_resize)
        self.label_resize.setObjectName(u"label_resize")
        self.label_resize.setMinimumSize(QSize(60, 0))
        self.label_resize.setMaximumSize(QSize(60, 16777215))
        self.label_resize.setStyleSheet(u"#label_12 {\n"
"width:60px\n"
"}")

        self.gridLayout_5.addWidget(self.label_resize, 0, 0, 1, 1)

        self.resize_y = QLabel(self.frame_resize)
        self.resize_y.setObjectName(u"resize_y")
        self.resize_y.setMinimumSize(QSize(100, 0))
        self.resize_y.setMaximumSize(QSize(100, 20))

        self.gridLayout_5.addWidget(self.resize_y, 2, 1, 1, 1)

        self.horizontalSpacer_resize_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_resize_3, 2, 2, 1, 1)


        self.verticalLayout_4.addWidget(self.frame_resize)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer)

        icon11 = QIcon()
        icon11.addFile(u":/icons/resource/icon/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.star_trail_option_box.addTab(self.star_trail_output, icon11, "")

        self.horizontalLayout_8.addWidget(self.star_trail_option_box)

        self.splitter.addWidget(self.args_area)
        self.preview_area = QFrame(self.splitter)
        self.preview_area.setObjectName(u"preview_area")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(255)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.preview_area.sizePolicy().hasHeightForWidth())
        self.preview_area.setSizePolicy(sizePolicy4)
        self.preview_area.setMinimumSize(QSize(750, 600))
        self.preview_area.setStyleSheet(u"#preview_area {\n"
"	background-color: rgba(255, 255, 255,255);\n"
"border:none;\n"
"}")
        self.preview_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_area.setFrameShadow(QFrame.Shadow.Raised)
        self.preview_area.setLineWidth(0)
        self.horizontalLayout_9 = QHBoxLayout(self.preview_area)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.img_view_label = imgDisplayQFrame(self.preview_area)
        self.img_view_label.setObjectName(u"img_view_label")
        sizePolicy1.setHeightForWidth(self.img_view_label.sizePolicy().hasHeightForWidth())
        self.img_view_label.setSizePolicy(sizePolicy1)
        self.img_view_label.setStyleSheet(u" QPushButton {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border:None;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"QPushButton:pressed {\n"
"	padding-bottom: 0px;\n"
"}\n"
"* QToolTip {\n"
"	/* \u5168\u5c40\u5e94\u7528 */\n"
"    background-color: rgba(250, 250, 250, 240) !important;  /* \u8bbe\u7f6e\u5de5\u5177\u63d0\u793a\u7684\u80cc\u666f\u4e3a\u534a\u900f\u660e\u767d\u8272 */\n"
"    color:rgba(35,35,35,210) !important;  /* \u8bbe\u7f6e\u6587\u5b57\u989c\u8272\u4e3a\u9ed1\u8272 */\n"
"    border: 1px solid rgba(220, 220, 220, 250) !important;  /* \u53ef\u9009\uff1a\u4e3a\u5de5\u5177\u63d0\u793a\u6dfb\u52a0\u8fb9\u6846 */\n"
"}")
        self.img_view_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.img_view_label.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.img_view_label)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.view_pre_img = hoverDisplayButton(self.img_view_label)
        self.view_pre_img.setObjectName(u"view_pre_img")
        self.view_pre_img.setMinimumSize(QSize(50, 500))
        self.view_pre_img.setMaximumSize(QSize(50, 800))
        icon12 = QIcon()
        icon12.addFile(u":/icons/resource/icon/left.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.view_pre_img.setIcon(icon12)
        self.view_pre_img.setIconSize(QSize(0, 0))

        self.gridLayout.addWidget(self.view_pre_img, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        self.view_next_img = hoverDisplayButton(self.img_view_label)
        self.view_next_img.setObjectName(u"view_next_img")
        self.view_next_img.setMinimumSize(QSize(50, 500))
        self.view_next_img.setMaximumSize(QSize(50, 800))
        icon13 = QIcon()
        icon13.addFile(u":/icons/resource/icon/right.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.view_next_img.setIcon(icon13)
        self.view_next_img.setIconSize(QSize(0, 0))

        self.gridLayout.addWidget(self.view_next_img, 1, 2, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 3, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_3, 0, 1, 1, 1)


        self.horizontalLayout_9.addWidget(self.img_view_label)

        self.splitter.addWidget(self.preview_area)

        self.horizontalLayout.addWidget(self.splitter)


        self.verticalLayout_2.addWidget(self.workspace)

        self.status_bar = QFrame(self.main_frame)
        self.status_bar.setObjectName(u"status_bar")
        sizePolicy1.setHeightForWidth(self.status_bar.sizePolicy().hasHeightForWidth())
        self.status_bar.setSizePolicy(sizePolicy1)
        self.status_bar.setMinimumSize(QSize(0, 50))
        self.status_bar.setMaximumSize(QSize(16777215, 50))
        self.status_bar.setStyleSheet(u"#status_bar {\n"
"	background-color: rgba(200, 200, 200, 200);\n"
"	border-width:0px;\n"
"	border-color:rgba(199,199,199,0);\n"
"	border-style:inset\n"
"}\n"
"\n"
"#status_bar #btn_star_trail_start:hover {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"}\n"
"#status_bar #btn_star_trail_preview:hover {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"}\n"
"\n"
"#status_bar QPushButton {\n"
"}\n"
"\n"
"#status_bar #btn_star_trail_start:pressed {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"	padding-top:5px;\n"
"}\n"
"#status_bar #btn_star_trail_preview:pressed {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"	padding-top:3px;\n"
"}\n"
"\n"
"#status_bar QLabel{\n"
"	font-size: 14px;           /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"    color:  rgba(20,20,20,220);              /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 550;\n"
"}\n"
"")
        self.status_bar.setFrameShape(QFrame.Shape.StyledPanel)
        self.status_bar.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.status_bar)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.status_bar)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy1.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy1)
        self.frame_6.setMinimumSize(QSize(0, 3))
        self.frame_6.setMaximumSize(QSize(10000, 3))
        self.frame_6.setStyleSheet(u"QFrame {\n"
"padding-left:5px;\n"
"padding-right:5px;\n"
"}")
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.star_trail_process_bar = QProgressBar(self.frame_6)
        self.star_trail_process_bar.setObjectName(u"star_trail_process_bar")
        sizePolicy1.setHeightForWidth(self.star_trail_process_bar.sizePolicy().hasHeightForWidth())
        self.star_trail_process_bar.setSizePolicy(sizePolicy1)
        self.star_trail_process_bar.setMinimumSize(QSize(0, 5))
        self.star_trail_process_bar.setMaximumSize(QSize(16777215, 5))
        self.star_trail_process_bar.setStyleSheet(u"QProgressBar {\n"
"	padding: 0px; \n"
"    border: 0px solid rgba(100,100,100,30);  /* \u8bbe\u7f6e\u8fb9\u6846 */\n"
"    border-radius: 10px;  /* \u8bbe\u7f6e\u5706\u89d2\u8fb9\u6846 */\n"
"    background-color: ;  /* \u8bbe\u7f6e\u8fdb\u5ea6\u6761\u80cc\u666f\u8272 */\n"
"	background-color: rgba(2, 53, 57,50);\n"
"\n"
"	min-height: 5px;\n"
"	max-height: 5px;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"	min-height: 5px;\n"
"	max-height: 5px;\n"
"	background-color: rgb(96, 200, 120);\n"
"	border: 0px solid rgba(100,100,100,30);\n"
"}\n"
"")
        self.star_trail_process_bar.setValue(24)

        self.horizontalLayout_12.addWidget(self.star_trail_process_bar)


        self.verticalLayout_6.addWidget(self.frame_6)

        self.frame_4 = QFrame(self.status_bar)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy1.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy1)
        self.frame_4.setMinimumSize(QSize(50, 50))
        self.frame_4.setMaximumSize(QSize(10000, 16777215))
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.frame_status = QFrame(self.frame_4)
        self.frame_status.setObjectName(u"frame_status")
        sizePolicy1.setHeightForWidth(self.frame_status.sizePolicy().hasHeightForWidth())
        self.frame_status.setSizePolicy(sizePolicy1)
        self.frame_status.setMinimumSize(QSize(200, 0))
        self.frame_status.setMaximumSize(QSize(200, 16777215))
        self.frame_status.setStyleSheet(u"QFrame {\n"
"	padding-left: 10px\n"
"}")
        self.frame_status.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_status.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_21 = QHBoxLayout(self.frame_status)
        self.horizontalLayout_21.setSpacing(0)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.status_icon = QPushButton(self.frame_status)
        self.status_icon.setObjectName(u"status_icon")
        sizePolicy1.setHeightForWidth(self.status_icon.sizePolicy().hasHeightForWidth())
        self.status_icon.setSizePolicy(sizePolicy1)
        self.status_icon.setMinimumSize(QSize(45, 45))
        self.status_icon.setMaximumSize(QSize(45, 45))
        self.status_icon.setStyleSheet(u"QPushButton:hover {\n"
"	border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"QPushButton {\n"
"	border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"}\n"
"")
        icon14 = QIcon()
        icon14.addFile(u":/icons/resource/icon/status-notready-.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.status_icon.setIcon(icon14)
        self.status_icon.setIconSize(QSize(20, 20))

        self.horizontalLayout_21.addWidget(self.status_icon)

        self.status_text = QLabel(self.frame_status)
        self.status_text.setObjectName(u"status_text")
        sizePolicy1.setHeightForWidth(self.status_text.sizePolicy().hasHeightForWidth())
        self.status_text.setSizePolicy(sizePolicy1)

        self.horizontalLayout_21.addWidget(self.status_text)


        self.horizontalLayout_13.addWidget(self.frame_status)

        self.frame_21 = QFrame(self.frame_4)
        self.frame_21.setObjectName(u"frame_21")
        sizePolicy1.setHeightForWidth(self.frame_21.sizePolicy().hasHeightForWidth())
        self.frame_21.setSizePolicy(sizePolicy1)
        self.frame_21.setMinimumSize(QSize(100, 50))
        self.frame_21.setMaximumSize(QSize(100, 50))
        self.frame_21.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_21.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_15 = QHBoxLayout(self.frame_21)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.btn_star_trail_preview = QPushButton(self.frame_21)
        self.btn_star_trail_preview.setObjectName(u"btn_star_trail_preview")
        self.btn_star_trail_preview.setMinimumSize(QSize(35, 35))
        self.btn_star_trail_preview.setMaximumSize(QSize(35, 35))
        icon15 = QIcon()
        icon15.addFile(u":/icons/resource/icon/fast-preview.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_star_trail_preview.setIcon(icon15)
        self.btn_star_trail_preview.setIconSize(QSize(18, 18))

        self.horizontalLayout_15.addWidget(self.btn_star_trail_preview)

        self.btn_star_trail_start = QPushButton(self.frame_21)
        self.btn_star_trail_start.setObjectName(u"btn_star_trail_start")
        sizePolicy1.setHeightForWidth(self.btn_star_trail_start.sizePolicy().hasHeightForWidth())
        self.btn_star_trail_start.setSizePolicy(sizePolicy1)
        self.btn_star_trail_start.setMinimumSize(QSize(45, 45))
        self.btn_star_trail_start.setMaximumSize(QSize(45, 45))
        self.btn_star_trail_start.setStyleSheet(u"QPushButton:hover {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"\n"
"}\n"
"\n"
"QPushButton {\n"
"\n"
"}\n"
"QPushButton:pressed {\n"
"	border-radius: 5px;\n"
"	background-color: rgba(220, 220, 220, 150);\n"
"	padding-top:5px;\n"
"\n"
"}\n"
"")
        icon16 = QIcon()
        icon16.addFile(u":/icons/resource/icon/start.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_star_trail_start.setIcon(icon16)
        self.btn_star_trail_start.setIconSize(QSize(40, 40))

        self.horizontalLayout_15.addWidget(self.btn_star_trail_start)


        self.horizontalLayout_13.addWidget(self.frame_21)

        self.frame_11 = QFrame(self.frame_4)
        self.frame_11.setObjectName(u"frame_11")
        sizePolicy1.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy1)
        self.frame_11.setStyleSheet(u"QFrame {\n"
"	padding-left: 30px;\n"
"}")
        self.frame_11.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_22 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_22.setSpacing(0)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.frame_11)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setMinimumSize(QSize(20, 0))
        self.label_7.setMaximumSize(QSize(20, 16777215))
        self.label_7.setStyleSheet(u"QLabel {\n"
"	margin-top: 5px;\n"
"	margin-bottom: 5px;\n"
"	border-left : 1px solid rgba(220,220,220,255);\n"
"}")

        self.horizontalLayout_22.addWidget(self.label_7)

        self.star_trial_tips = QLabel(self.frame_11)
        self.star_trial_tips.setObjectName(u"star_trial_tips")
        self.star_trial_tips.setStyleSheet(u"QLabel {\n"
"	padding-left:0px\n"
"}")

        self.horizontalLayout_22.addWidget(self.star_trial_tips)


        self.horizontalLayout_13.addWidget(self.frame_11)

        self.frame_20 = QFrame(self.frame_4)
        self.frame_20.setObjectName(u"frame_20")
        sizePolicy1.setHeightForWidth(self.frame_20.sizePolicy().hasHeightForWidth())
        self.frame_20.setSizePolicy(sizePolicy1)
        self.frame_20.setMinimumSize(QSize(10, 0))
        self.frame_20.setMaximumSize(QSize(1, 16777215))
        self.frame_20.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_13.addWidget(self.frame_20)


        self.verticalLayout_6.addWidget(self.frame_4)


        self.verticalLayout_2.addWidget(self.status_bar)


        self.verticalLayout.addWidget(self.main_frame)

        HNW.setCentralWidget(self.centralwidget)

        self.retranslateUi(HNW)
        self.alter_jpg_level.valueChanged.connect(self.jpg_level.setNum)

        self.star_trail_option_box.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(HNW)
    # setupUi

    def retranslateUi(self, HNW):
        HNW.setWindowTitle(QCoreApplication.translate("HNW", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.label.setToolTip(QCoreApplication.translate("HNW", u"\u7ec7\u6b64\u661f\u8fb0\uff1a\u4e00\u4e2a\u7b80\u5355\u6613\u7528\u3001\u9ad8\u6548\u5feb\u6377\u7684\u56fe\u50cf\u53e0\u52a0\u8f6f\u4ef6\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("HNW", u"HoshiNoWeaver", None))
#if QT_CONFIG(tooltip)
        self.label_current_mode.setToolTip(QCoreApplication.translate("HNW", u"\u5207\u6362\u5de5\u4f5c\u573a\u666f\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_current_mode.setText(QCoreApplication.translate("HNW", u"\u661f\u8f68\u53e0\u52a0", None))
#if QT_CONFIG(tooltip)
        self.menu_setting.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u8bbe\u7f6e</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.menu_setting.setText("")
#if QT_CONFIG(tooltip)
        self.menu_about.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u5173\u4e8e</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.menu_about.setText("")
        self.label_3.setText("")
#if QT_CONFIG(tooltip)
        self.ui_min.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6700\u5c0f\u5316</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.ui_min.setText("")
#if QT_CONFIG(tooltip)
        self.ui_max.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6700\u5927\u5316</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.ui_max.setText("")
#if QT_CONFIG(tooltip)
        self.ui_close.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u9000\u51fa</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.ui_close.setText("")
#if QT_CONFIG(tooltip)
        self.star_trail_option_box.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.add_files.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6dfb\u52a0\u56fe\u7247</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.add_files.setWhatsThis(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6dfb\u52a0\u56fe\u7247</p></body></html>", None))
#endif // QT_CONFIG(whatsthis)
        self.add_files.setText("")
#if QT_CONFIG(tooltip)
        self.add_folder.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6dfb\u52a0\u6587\u4ef6\u5939</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.add_folder.setWhatsThis(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6dfb\u52a0\u6587\u4ef6\u5939</p></body></html>", None))
#endif // QT_CONFIG(whatsthis)
        self.add_folder.setText("")
#if QT_CONFIG(tooltip)
        self.clear_files.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6e05\u7a7a\u5217\u8868</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.clear_files.setWhatsThis(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u6e05\u7a7a\u5217\u8868</p></body></html>", None))
#endif // QT_CONFIG(whatsthis)
        self.clear_files.setText("")
        self.star_trail_option_box.setTabText(self.star_trail_option_box.indexOf(self.star_trail_input_files), QCoreApplication.translate("HNW", u"\u56fe\u50cf\u5217\u8868", None))
#if QT_CONFIG(tooltip)
        self.label_19.setToolTip(QCoreApplication.translate("HNW", u"\u9009\u62e9\u661f\u8f68\u53e0\u52a0\u7b97\u6cd5\u3002\n"
"\u6700\u5927\u503c\uff1a\u53d6\u8f93\u5165\u56fe\u50cf\u5e8f\u5217\u5404\u50cf\u7d20\u70b9\u6700\u5927\u4eae\u5ea6\u4fe1\u606f\u8fdb\u884c\u53e0\u52a0\u3002\n"
"\u6df7\u5408\u6a21\u5f0f\uff1a\u57fa\u4e8e\u8499\u7248\u56fe\u50cf\uff0c\u5bf9\u4e8e\u8499\u7248\u533a\u57df\u91c7\u7528Sigma\u88c1\u526a\u5747\u503c\u53e0\u52a0\u7b97\u6cd5\uff0c\u5bf9\u4e8e\u5176\u4ed6\u90e8\u5206\u91c7\u7528\u6700\u5927\u503c\u53e0\u52a0\u7b97\u6cd5\uff0c\u5728\u53e0\u52a0\u661f\u8f68\u7684\u540c\u65f6\uff0c\u63d0\u9ad8\u8499\u7248\u90e8\u5206\uff08\u5982\u5730\u666f\uff09\u7684\u4fe1\u566a\u6bd4\u53ca\u7eaf\u51c0\u5ea6\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_19.setText(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65b9\u5f0f", None))
        self.alter_algorithm_startrail.setItemText(0, QCoreApplication.translate("HNW", u"\u6700\u5927\u503c", None))
        self.alter_algorithm_startrail.setItemText(1, QCoreApplication.translate("HNW", u"\u6df7\u5408\u6a21\u5f0f", None))

#if QT_CONFIG(tooltip)
        self.label_16.setToolTip(QCoreApplication.translate("HNW", u"\u9009\u62e9\u5806\u6808\u964d\u566a\u53e0\u52a0\u7b97\u6cd5\u3002\n"
"\u5e73\u5747\u503c\uff1a\u53d6\u8f93\u5165\u56fe\u50cf\u5e8f\u5217\u5404\u50cf\u7d20\u70b9\u5e73\u5747\u4eae\u5ea6\u4fe1\u606f\u8fdb\u884c\u53e0\u52a0\u3002\n"
"\u5e73\u5747\u503c-\u6392\u5f02\uff1a\u4f7f\u7528Sigma\u88c1\u5207\u5747\u503c\u7b97\u6cd5\uff0c\u57fa\u4e8e\u8f93\u5165\u56fe\u50cf\u5e8f\u5217\u5404\u50cf\u7d20\u70b9\u4eae\u5ea6\u4fe1\u606f\uff0c\u5bf9\u504f\u79bb\u5747\u503c\u8fc7\u9ad8\u7684\u50cf\u7d20\u70b9\u8fdb\u884c\u8fc7\u6ee4\u3002\u8be5\u7b97\u6cd5\u8bfe\u5254\u9664\u56fe\u50cf\u4e2d\u5b58\u5728\u7684\u98de\u673a\u7ebf\u3001\u536b\u661f\u7ebf\u3001\u6d41\u661f\u7b49\u7684\u5f71\u54cd\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_16.setText(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65b9\u5f0f", None))
        self.alter_algorithm_mean.setItemText(0, QCoreApplication.translate("HNW", u"\u5e73\u5747\u503c", None))
        self.alter_algorithm_mean.setItemText(1, QCoreApplication.translate("HNW", u"\u5e73\u5747\u503c-\u6392\u5f02", None))

        self.label_20.setText(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65b9\u5f0f", None))
        self.alter_algorithm_min.setItemText(0, QCoreApplication.translate("HNW", u"\u6700\u5c0f\u503c", None))

        self.label_mask.setText(QCoreApplication.translate("HNW", u"\u8499\u7248\u6587\u4ef6", None))
#if QT_CONFIG(tooltip)
        self.mask_able.setToolTip(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65f6\u4ee5\u6574\u5f62\u6743\u91cd\u4ee3\u66ff\u6d6e\u70b9\u6743\u91cd\u3002\n"
"\u5927\u591a\u6570\u60c5\u51b5\u4e0b\uff0c\u542f\u7528\u8be5\u9009\u9879\u53ef\u4ee5\u53ef\u63a5\u53d7\u7684\u7cbe\u5ea6\u635f\u5931\u6362\u53d6\u66f4\u5feb\u7684\u8fd0\u884c\u901f\u5ea6\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.mask_able.setText(QCoreApplication.translate("HNW", u"\u542f\u7528", None))
#if QT_CONFIG(tooltip)
        self.alter_mask_file.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u9009\u62e9\u8499\u7248</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.alter_mask_file.setText("")
#if QT_CONFIG(tooltip)
        self.label_max_iter.setToolTip(QCoreApplication.translate("HNW", u"Sigma\u88c1\u526a\u5747\u503c\u7684\u6700\u5927\u8fed\u4ee3\u8f6e\u6570\uff0c0~10\uff0c\u9ed8\u8ba45\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_max_iter.setText(QCoreApplication.translate("HNW", u"\u6700\u5927\u8fed\u4ee3\u6b21\u6570", None))
#if QT_CONFIG(tooltip)
        self.alter_max_iter.setToolTip(QCoreApplication.translate("HNW", u"\u9009\u62e9\u8f93\u51faJPG\u56fe\u50cf\u7684\u8d28\u91cf\uff08100\u4e3a\u6700\u9ad8\uff09", None))
#endif // QT_CONFIG(tooltip)
        self.max_iter.setText(QCoreApplication.translate("HNW", u"0", None))
#if QT_CONFIG(tooltip)
        self.label_28.setToolTip(QCoreApplication.translate("HNW", u"Sigma\u88c1\u526a\u5747\u503c\u53e0\u52a0\u7684\u62d2\u7edd\u500d\u7387\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_28.setText(QCoreApplication.translate("HNW", u"\u62d2\u7edd\u500d\u7387", None))
#if QT_CONFIG(tooltip)
        self.label_rejection_low.setToolTip(QCoreApplication.translate("HNW", u"Sigma\u88c1\u526a\u5747\u503c\u53e0\u52a0\u7684\u62d2\u7edd\u4e0b\u754c\u500d\u7387\uff0c-6~0\uff0c\u9ed8\u8ba4-3\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_rejection_low.setText(QCoreApplication.translate("HNW", u"\u4e0b\u754c", None))
        self.rejection_low.setText(QCoreApplication.translate("HNW", u"0", None))
        self.rejection_high.setText(QCoreApplication.translate("HNW", u"100", None))
#if QT_CONFIG(tooltip)
        self.label_rejection_high.setToolTip(QCoreApplication.translate("HNW", u"Sigma\u88c1\u526a\u5747\u503c\u53e0\u52a0\u7684\u62d2\u7edd\u4e0a\u754c\u500d\u7387\uff0c0~6\uff0c\u9ed8\u8ba43\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_rejection_high.setText(QCoreApplication.translate("HNW", u"\u4e0a\u754c", None))
#if QT_CONFIG(tooltip)
        self.lable_fade.setToolTip(QCoreApplication.translate("HNW", u"\u8bbe\u7f6e\u661f\u8f68\u7684\u8d77\u59cb\u90e8\u5206\u4e3a\u6e10\u5165\u6548\u679c\u3001\u672b\u7aef\u90e8\u5206\u4e3a\u6e10\u51fa\u6548\u679c\u3002\n"
"\u6e10\u5165\u3001\u6e10\u51fa\u4e2d\u95f4\u90e8\u5206\u4e3a\u6b63\u5e38\u53e0\u52a0\u3002\n"
"", None))
#endif // QT_CONFIG(tooltip)
        self.lable_fade.setText(QCoreApplication.translate("HNW", u"\u6e10\u9690\u9009\u9879", None))
#if QT_CONFIG(tooltip)
        self.lable_fade_in.setToolTip(QCoreApplication.translate("HNW", u"\u6e10\u5165\u90e8\u5206\u5360\u6574\u4e2a\u661f\u8f68\u7684\u957f\u5ea6\u767e\u5206\u6bd4\u3002\n"
"0%~100%\uff0c\u4e0e\u6e10\u51fa\u90e8\u5206\u4e4b\u548c\u4e0d\u5927\u4e8e100%\uff0c0%\u8868\u793a\u4e0d\u542f\u7528\u6e10\u5165\uff0c\u9ed8\u8ba40%\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.lable_fade_in.setText(QCoreApplication.translate("HNW", u"\u6e10\u5165", None))
        self.fade_in.setText(QCoreApplication.translate("HNW", u"0", None))
        self.fade_out.setText(QCoreApplication.translate("HNW", u"100", None))
#if QT_CONFIG(tooltip)
        self.lable_fade_out.setToolTip(QCoreApplication.translate("HNW", u"\u6e10\u51fa\u90e8\u5206\u5360\u6574\u4e2a\u661f\u8f68\u7684\u957f\u5ea6\u767e\u5206\u6bd4\u3002\n"
"0%~100%\uff0c\u4e0e\u6e10\u5165\u90e8\u5206\u4e4b\u548c\u4e0d\u5927\u4e8e100%\uff0c0%\u8868\u793a\u4e0d\u542f\u7528\u6e10\u51fa\uff0c\u9ed8\u8ba40%\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.lable_fade_out.setText(QCoreApplication.translate("HNW", u"\u6e10\u51fa", None))
#if QT_CONFIG(tooltip)
        self.label_frame_int_weight.setToolTip(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65f6\u4ee5\u6574\u5f62\u6743\u91cd\u4ee3\u66ff\u6d6e\u70b9\u6743\u91cd\u3002\n"
"\u5927\u591a\u6570\u60c5\u51b5\u4e0b\uff0c\u542f\u7528\u8be5\u9009\u9879\u53ef\u5b9e\u73b0\u4ee5\u53ef\u63a5\u53d7\u7684\u7cbe\u5ea6\u635f\u5931\u6362\u53d6\u66f4\u5feb\u7684\u8fd0\u884c\u901f\u5ea6\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_frame_int_weight.setText(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u52a0\u901f", None))
#if QT_CONFIG(tooltip)
        self.int_weight_able.setToolTip(QCoreApplication.translate("HNW", u"\u8ba1\u7b97\u65f6\u4ee5\u6574\u5f62\u6743\u91cd\u4ee3\u66ff\u6d6e\u70b9\u6743\u91cd\u3002\n"
"\u5927\u591a\u6570\u60c5\u51b5\u4e0b\uff0c\u542f\u7528\u8be5\u9009\u9879\u53ef\u5b9e\u73b0\u4ee5\u53ef\u63a5\u53d7\u7684\u7cbe\u5ea6\u635f\u5931\u6362\u53d6\u66f4\u5feb\u7684\u8fd0\u884c\u901f\u5ea6\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.int_weight_able.setText(QCoreApplication.translate("HNW", u"\u542f\u7528", None))
        self.star_trail_option_box.setTabText(self.star_trail_option_box.indexOf(self.star_trail_input_setting), QCoreApplication.translate("HNW", u"\u53e0\u52a0\u9009\u9879", None))
#if QT_CONFIG(tooltip)
        self.label_output_type.setToolTip(QCoreApplication.translate("HNW", u"\u9009\u62e9\u5b58\u50a8\u56fe\u50cf\u683c\u5f0f\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_output_type.setText(QCoreApplication.translate("HNW", u"\u6587\u4ef6\u683c\u5f0f", None))
        self.alter_output_type_2.setItemText(0, QCoreApplication.translate("HNW", u"JPG", None))
        self.alter_output_type_2.setItemText(1, QCoreApplication.translate("HNW", u"PNG", None))
        self.alter_output_type_2.setItemText(2, QCoreApplication.translate("HNW", u"TIFF", None))

#if QT_CONFIG(tooltip)
        self.label_output_path.setToolTip(QCoreApplication.translate("HNW", u"\u9009\u62e9\u5b58\u50a8\u8def\u5f84\u53ca\u5b58\u50a8\u6587\u4ef6\u540d\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_output_path.setText(QCoreApplication.translate("HNW", u"\u5b58\u50a8\u8def\u5f84", None))
#if QT_CONFIG(tooltip)
        self.alter_output_2.setToolTip(QCoreApplication.translate("HNW", u"<html><head/><body><p>\u9009\u62e9\u5b58\u50a8\u8def\u5f84</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.alter_output_2.setText("")
#if QT_CONFIG(tooltip)
        self.label_png_level.setToolTip(QCoreApplication.translate("HNW", u"PNG\u56fe\u50cf\u7684\u538b\u7f29\u7ea7\u522b\uff0c\u8d8a\u9ad8\u6587\u4ef6\u8d8a\u5c0f\u30021~9\uff0c\u9ed8\u8ba40\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_png_level.setText(QCoreApplication.translate("HNW", u"\u538b\u7f29\u7ea7\u522b", None))
        self.png_level.setText(QCoreApplication.translate("HNW", u"0", None))
        self.jpg_level.setText(QCoreApplication.translate("HNW", u"0", None))
#if QT_CONFIG(tooltip)
        self.label_jpg_level.setToolTip(QCoreApplication.translate("HNW", u"\u5b58\u50a8JPG\u56fe\u50cf\u7684\u8d28\u91cf\uff0c\u8d8a\u9ad8\u56fe\u50cf\u8d28\u91cf\u8d8a\u597d\u30021~100\uff0c\u9ed8\u8ba485\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_jpg_level.setText(QCoreApplication.translate("HNW", u"\u56fe\u50cf\u8d28\u91cf", None))
#if QT_CONFIG(tooltip)
        self.label_output_bits.setToolTip(QCoreApplication.translate("HNW", u"\u4fdd\u5b58\u7684\u56fe\u50cf\u7684\u8272\u6df1\u3002\u5f53\u5b58\u50a8JPG\u6587\u4ef6\u65f6\uff0c\u4ec5\u652f\u63018bit\u3002", None))
#endif // QT_CONFIG(tooltip)
        self.label_output_bits.setText(QCoreApplication.translate("HNW", u"\u56fe\u50cf\u8272\u6df1", None))
        self.alter_output_bits.setItemText(0, QCoreApplication.translate("HNW", u"8 bit", None))
        self.alter_output_bits.setItemText(1, QCoreApplication.translate("HNW", u"16 bit", None))
        self.alter_output_bits.setItemText(2, QCoreApplication.translate("HNW", u"32 bit", None))

        self.resize_x_y.setText(QCoreApplication.translate("HNW", u"\u7f29\u5c0f\u6bd4\u4f8b\uff1a100%", None))
        self.resize_x.setText(QCoreApplication.translate("HNW", u"\u957f\uff1a10000px ", None))
        self.label_resize.setText(QCoreApplication.translate("HNW", u"\u8f93\u51fa\u5c3a\u5bf8", None))
        self.resize_y.setText(QCoreApplication.translate("HNW", u"\u5bbd\uff1a10000px", None))
        self.star_trail_option_box.setTabText(self.star_trail_option_box.indexOf(self.star_trail_output), QCoreApplication.translate("HNW", u"\u8f93\u51fa\u9009\u9879", None))
        self.view_pre_img.setText("")
        self.view_next_img.setText("")
        self.star_trail_process_bar.setFormat("")
        self.status_icon.setText("")
        self.status_text.setText(QCoreApplication.translate("HNW", u"\u672a\u5c31\u7eea", None))
#if QT_CONFIG(tooltip)
        self.btn_star_trail_preview.setToolTip(QCoreApplication.translate("HNW", u"\u6309\u7740\u89e3\u538b\u7684\u6309\u94ae\uff08\u4e0d\u662f\uff09\n"
"\u8fd8\u6ca1\u5f00\u653e\u7684\u65b0\u529f\u80fd", None))
#endif // QT_CONFIG(tooltip)
        self.btn_star_trail_preview.setText("")
#if QT_CONFIG(tooltip)
        self.btn_star_trail_start.setToolTip(QCoreApplication.translate("HNW", u"\u5f00\u59cb\u53e0\u52a0", None))
#endif // QT_CONFIG(tooltip)
        self.btn_star_trail_start.setText("")
        self.label_7.setText("")
        self.star_trial_tips.setText(QCoreApplication.translate("HNW", u"\u6587\u5b57\u663e\u793a", None))
    # retranslateUi
class Ui_guide(object):
    def setupUi(self, guide):
        if not guide.objectName():
            guide.setObjectName(u"guide")
        guide.resize(815, 657)
        guide.setMinimumSize(QSize(815, 0))
        guide.setMaximumSize(QSize(815, 16777215))
        self.horizontalLayout = QHBoxLayout(guide)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.frame = QFrame(guide)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"#frame {\n"
"	background-color: rgba(230, 229, 228, 200);\n"
"	border: 1px solid rgba(220, 220, 220, 220);\n"
"	border-radius: 5px;\n"
"	padding: 5px 5px 5px 5px;\n"
"}\n"
"#frame *{\n"
"	background-color:  rgba(255, 255, 255, 0);\n"
"}")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.title = QFrame(self.frame)
        self.title.setObjectName(u"title")
        self.title.setMinimumSize(QSize(0, 20))
        self.title.setMaximumSize(QSize(16777215, 20))
        self.title.setStyleSheet(u"#title {\n"
"	\n"
"	border-bottom: 1px solid rgba(220, 220, 220, 150);\n"
"}")
        self.title.setFrameShape(QFrame.Shape.StyledPanel)
        self.title.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.title)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.title)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"QLabel {\n"
"	qproperty-alignment: 'AlignCenter';\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	border:None;\n"
"	font: 14pt \"\u5b8b\u4f53\";\n"
"\n"
"\n"
"}")

        self.horizontalLayout_4.addWidget(self.label)


        self.verticalLayout.addWidget(self.title)

        self.guide_area = QStackedWidget(self.frame)
        self.guide_area.setObjectName(u"guide_area")
        self.guide_area.setMinimumSize(QSize(0, 577))
        self.guide_area.setMaximumSize(QSize(16777215, 577))
        self.guide_area.setStyleSheet(u"")
        self.guide01 = QWidget()
        self.guide01.setObjectName(u"guide01")
        self.guide01.setStyleSheet(u"#guide01 {\n"
"	image: url(:/guide/resource/guide/01.png);\n"
"}")
        self.hboxLayout = QHBoxLayout(self.guide01)
        self.hboxLayout.setSpacing(0)
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.hboxLayout.setContentsMargins(0, 0, 0, 0)
        self.guide_area.addWidget(self.guide01)
        self.guide02 = QWidget()
        self.guide02.setObjectName(u"guide02")
        self.guide02.setStyleSheet(u"#guide02 {\n"
"	image: url(:/guide/resource/guide/02.png);\n"
"}")
        self.guide_area.addWidget(self.guide02)
        self.guide03 = QWidget()
        self.guide03.setObjectName(u"guide03")
        self.guide03.setStyleSheet(u"#guide03 {\n"
"	image: url(:/guide/resource/guide/03.png);\n"
"}")
        self.guide_area.addWidget(self.guide03)
        self.guide04 = QWidget()
        self.guide04.setObjectName(u"guide04")
        self.guide04.setStyleSheet(u"#guide04 {\n"
"	image: url(:/guide/resource/guide/04.png);\n"
"}")
        self.guide_area.addWidget(self.guide04)
        self.guide05 = QWidget()
        self.guide05.setObjectName(u"guide05")
        self.guide05.setStyleSheet(u"#guide05 {\n"
"	image: url(:/guide/resource/guide/05.png);\n"
"}")
        self.guide_area.addWidget(self.guide05)
        self.guide06 = QWidget()
        self.guide06.setObjectName(u"guide06")
        self.guide06.setStyleSheet(u"#guide06 {\n"
"	image: url(:/guide/resource/guide/06.png);\n"
"}")
        self.guide_area.addWidget(self.guide06)
        self.guide07 = QWidget()
        self.guide07.setObjectName(u"guide07")
        self.guide07.setStyleSheet(u"#guide07 {\n"
"	image: url(:/guide/resource/guide/07.png);\n"
"}")
        self.guide_area.addWidget(self.guide07)
        self.guide08 = QWidget()
        self.guide08.setObjectName(u"guide08")
        self.guide08.setStyleSheet(u"#guide08 {\n"
"	image: url(:/guide/resource/guide/08.png);\n"
"}")
        self.guide_area.addWidget(self.guide08)
        self.guide09 = QWidget()
        self.guide09.setObjectName(u"guide09")
        self.guide09.setStyleSheet(u"#guide09 {\n"
"	image: url(:/guide/resource/guide/09.png);\n"
"}")
        self.guide_area.addWidget(self.guide09)

        self.verticalLayout.addWidget(self.guide_area)

        self.btn = QFrame(self.frame)
        self.btn.setObjectName(u"btn")
        self.btn.setMinimumSize(QSize(0, 30))
        self.btn.setMaximumSize(QSize(16777215, 30))
        self.btn.setStyleSheet(u"QPushButton {\n"
"	width: 80px\n"
"}\n"
"QPushButton {\n"
"	background-color: rgba(250,250,250,220);\n"
"	color:rgba(35,35,35,210);\n"
"	padding: 0px;\n"
"	border: 0px solid #ccc;\n"
"	border-radius: 5px;\n"
"	width:25px;\n"
"	height:80px;\n"
"	padding-left: 3px;\n"
"	padding-top: 3px;\n"
"	margin-bottom: 1px;\n"
"	margin-left: 2px\n"
"\n"
"}\n"
"QPushButton:hover{\n"
"	padding-bottom: 3px;\n"
"}\n"
"QPushButton:disabled{\n"
"	background-color: rgba(210,210,210,200);\n"
"}")
        self.btn.setFrameShape(QFrame.Shape.StyledPanel)
        self.btn.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.btn)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_2 = QSpacerItem(257, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)

        self.pre = QPushButton(self.btn)
        self.pre.setObjectName(u"pre")
        self.pre.setMinimumSize(QSize(100, 0))
        self.pre.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_2.addWidget(self.pre)

        self.close_guide = QPushButton(self.btn)
        self.close_guide.setObjectName(u"close_guide")
        self.close_guide.setMinimumSize(QSize(100, 0))
        self.close_guide.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_2.addWidget(self.close_guide)

        self.next = QPushButton(self.btn)
        self.next.setObjectName(u"next")
        self.next.setMinimumSize(QSize(100, 0))
        self.next.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout_2.addWidget(self.next)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.display_always = QCheckBox(self.btn)
        self.display_always.setObjectName(u"display_always")
        self.display_always.setMinimumSize(QSize(120, 0))
        self.display_always.setMaximumSize(QSize(120, 16777215))
        self.display_always.setChecked(True)

        self.horizontalLayout_2.addWidget(self.display_always)


        self.verticalLayout.addWidget(self.btn)


        self.horizontalLayout.addWidget(self.frame)


        self.retranslateUi(guide)

        self.guide_area.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(guide)
    # setupUi

    def retranslateUi(self, guide):
        guide.setWindowTitle(QCoreApplication.translate("guide", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("guide", u"\u4f7f\u7528\u6307\u5357", None))
        self.pre.setText(QCoreApplication.translate("guide", u"\u4e0a\u4e00\u6b65", None))
        self.close_guide.setText(QCoreApplication.translate("guide", u"\u5173\u95ed", None))
        self.next.setText(QCoreApplication.translate("guide", u"\u5f00\u59cb", None))
        self.display_always.setText(QCoreApplication.translate("guide", u"\u6bcf\u6b21\u542f\u52a8\u65f6\u5f39\u51fa", None))
    # retranslateUi



class ui_choose_mode(object):
    def setupUi(self, HNW_choose_mode):
        if not HNW_choose_mode.objectName():
            HNW_choose_mode.setObjectName(u"HNW_choose_mode")
        HNW_choose_mode.resize(200, 120)
        HNW_choose_mode.setMinimumSize(QSize(200, 120))
        HNW_choose_mode.setMaximumSize(QSize(200, 150))
        self.centralwidget = QWidget(HNW_choose_mode)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"#centralwidget QPushButton, QLabel, ClickableLabel{\n"
"	font-size: 11px;                         /* \u8bbe\u7f6e\u5b57\u4f53\u5927\u5c0f */\n"
"     color:  rgba(30,30,30,200);       /* \u8bbe\u7f6e\u5b57\u4f53\u989c\u8272 */\n"
"	font-weight: 400;                      /* \u8bbe\u7f6e\u5b57\u4f53\u539a\u5ea6 */\n"
"	qproperty-alignment: 'AlignCenter';\n"
"}\n"
"\n"
"#centralwidget ClickableLabel {\n"
"	background-color: rgba(255, 255, 255, 0);\n"
"	margin:5px;\n"
"}\n"
"#centralwidget ClickableLabel:hover{\n"
"	padding-bottom: 5px;\n"
"}\n"
"\n"
"#centralwidget ClickableLabel:pressed {\n"
"	margin:0px 0px 0px 0px ;\n"
"	padding:0px 0px 0px 0px ;\n"
"}\n"
"#centralwidget {\n"
"	border: 1px solid rgba(240, 240, 240, 200);\n"
"}\n"
"")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"#frame {\n"
"	border-left: 1px solid rgba(0, 240, 240, 200);\n"
"}")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setStyleSheet(u"#frame_3 > QFrame {\n"
"	margin: 2px 2px 2px 2px;\n"
"	\n"
"	background-color: rgb(220, 220, 220);\n"
"}")
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.frame_4 = QFrame(self.frame_3)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_4)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.img_startrail = ClickableLabel(self.frame_4)
        self.img_startrail.setObjectName(u"img_startrail")
        self.img_startrail.setMinimumSize(QSize(90, 90))
        self.img_startrail.setMaximumSize(QSize(90, 90))
        self.img_startrail.setStyleSheet(u"#img_startrail {\n"
"	border:none;\n"
"	border-image: url(:/img/resource/img/01.jpg);\n"
"	background-repeat: no-repeat;                     /* \u4e0d\u91cd\u590d\u80cc\u666f\u56fe */\n"
"    background-position: center;                      /* \u5c45\u4e2d\u663e\u793a\u80cc\u666f\u56fe */\n"
"}")

        self.verticalLayout_2.addWidget(self.img_startrail, 0, Qt.AlignmentFlag.AlignHCenter)

        self.label_3 = ClickableLabel(self.frame_4)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(0, 20))
        self.label_3.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_2.addWidget(self.label_3)


        self.horizontalLayout_2.addWidget(self.frame_4)

        self.frame_5 = QFrame(self.frame_3)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame_5)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.img_avg = ClickableLabel(self.frame_5)
        self.img_avg.setObjectName(u"img_avg")
        self.img_avg.setMinimumSize(QSize(90, 90))
        self.img_avg.setMaximumSize(QSize(90, 90))
        self.img_avg.setStyleSheet(u"#img_avg {\n"
"	border:none;\n"
"	border-image: url(:/img/resource/img/02.jpg);\n"
"	background-repeat: no-repeat;                     /* \u4e0d\u91cd\u590d\u80cc\u666f\u56fe */\n"
"    background-position: center;                      /* \u5c45\u4e2d\u663e\u793a\u80cc\u666f\u56fe */\n"
"}")

        self.verticalLayout_3.addWidget(self.img_avg, 0, Qt.AlignmentFlag.AlignHCenter)

        self.label_4 = ClickableLabel(self.frame_5)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(0, 20))
        self.label_4.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.label_4)


        self.horizontalLayout_2.addWidget(self.frame_5)


        self.verticalLayout.addWidget(self.frame_3)


        self.horizontalLayout.addWidget(self.frame)

        HNW_choose_mode.setCentralWidget(self.centralwidget)

        self.retranslateUi(HNW_choose_mode)

        QMetaObject.connectSlotsByName(HNW_choose_mode)
    # setupUi

    def retranslateUi(self, HNW_choose_mode):
        HNW_choose_mode.setWindowTitle(QCoreApplication.translate("HNW_choose_mode", u"MainWindow", None))
        self.img_startrail.setText("")
        self.label_3.setText(QCoreApplication.translate("HNW_choose_mode", u"\u661f\u8f68\u53e0\u52a0", None))
        self.img_avg.setText("")
        self.label_4.setText(QCoreApplication.translate("HNW_choose_mode", u"\u5806\u6808\u964d\u566a", None))
    # retranslateUi

