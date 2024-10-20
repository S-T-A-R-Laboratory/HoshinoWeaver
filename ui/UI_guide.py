# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI_guide.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QStackedWidget, QVBoxLayout, QWidget)
from ui import resource_choose_mode

class Ui_guide(object):
    def setupUi(self, guide):
        if not guide.objectName():
            guide.setObjectName(u"guide")
        guide.resize(815, 633)
        guide.setMinimumSize(QSize(815, 0))
        guide.setMaximumSize(QSize(815, 16777215))
        guide.setStyleSheet(u"")
        self.centralwidget = QWidget(guide)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"#centralwidget {\n"
"	background-color: rgba(230, 229, 228, 200);\n"
"	border: 1px solid rgba(220, 220, 220, 220);\n"
"	border-radius: 5px;\n"
"}\n"
"#centralwidget *{\n"
"	background-color:  rgba(255, 255, 255, 0);\n"
"}")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setStyleSheet(u"#frame{\n"
"	padding: 5px 5px 5px 5px;\n"
"\n"
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

        guide.setCentralWidget(self.centralwidget)

        self.retranslateUi(guide)

        self.guide_area.setCurrentIndex(8)


        QMetaObject.connectSlotsByName(guide)
    # setupUi

    def retranslateUi(self, guide):
        guide.setWindowTitle(QCoreApplication.translate("guide", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("guide", u"\u4f7f\u7528\u6307\u5357", None))
        self.pre.setText(QCoreApplication.translate("guide", u"\u4e0a\u4e00\u6b65", None))
        self.close_guide.setText(QCoreApplication.translate("guide", u"\u5173\u95ed", None))
        self.next.setText(QCoreApplication.translate("guide", u"\u5f00\u59cb", None))
        self.display_always.setText(QCoreApplication.translate("guide", u"\u6bcf\u6b21\u542f\u52a8\u65f6\u5f39\u51fa", None))
    # retranslateUi

