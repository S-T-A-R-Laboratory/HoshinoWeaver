# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI_choose_mode.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QMainWindow,
    QSizePolicy, QVBoxLayout, QWidget)

from ui.UILibs import ClickableLabel
from ui import resource_choose_mode

class HNW_choose_mode(object):
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
"	border-image: url(:/imgs/resource/img/01.jpg);\n"
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
"	border-image: url(:/imgs/resource/img/02.jpg);\n"
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

