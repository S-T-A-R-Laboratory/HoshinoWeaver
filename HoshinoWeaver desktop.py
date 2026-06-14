'''

'''
from __future__ import annotations

import os
import sys
from pathlib import Path

import asyncio
import ctypes
import json
import platform
import time

from loguru import logger as _logger
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QMouseEvent
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QFrame, QHeaderView, QMainWindow, QScrollArea,
                               QTreeWidgetItem)
from qasync import QEventLoop

from hoshicore.component.utils import ORG_NAME, SOFTWARE_NAME, VERSION
from hoshicore.component.utils import init_logger as _init_logger
from ui.output_panel import OutputPanel
from ui.panel_builder import DynamicConfigPanel, PanelSchema
from ui.UI import Ui_guide, Ui_HNW
from ui.UILibs import borderFrame
from ui.UIUtils import SlotHandler

_BASE_DIR = Path(getattr(sys, '_MEIPASS', '')) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

MODE_MAP = {
    "星轨叠加": (_BASE_DIR / "hoshicore/dag/startrail.meta.yaml",
             _BASE_DIR / "hoshicore/dag/startrail.ui.yaml"),
    "堆栈降噪": (_BASE_DIR / "hoshicore/dag/stack.meta.yaml",
             _BASE_DIR / "hoshicore/dag/stack.ui.yaml"),
    "星点对齐叠加": (_BASE_DIR / "hoshicore/dag/sky_ground_stack.meta.yaml",
             _BASE_DIR / "hoshicore/dag/sky_ground_stack.ui.yaml"),
}

class SnapPreviewWindow(QFrame):
    """Half-transparent overlay that shows where the window will snap to."""
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet(
            "QFrame { background-color: rgba(0, 120, 215, 80); "
            "border: 2px solid rgba(0, 120, 215, 200); border-radius: 6px; }"
        )


class HNW_guide(QDialog, Ui_guide):
    def __init__(self, callback, display_always_flag=True,parent=None):
        super().__init__(parent)
        self.setupUi(self)  # 初始化通过 Qt Designer 生成的 UI
        self.setModal(True)

        self.setWindowTitle("使用指南")
        self.setWindowFlags(Qt.Dialog)  # 设置为弹出窗口
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("")

        self.next.clicked.connect(self.next_image)
        self.pre.clicked.connect(self.prev_image)
        self.close_guide.clicked.connect(self.close)

        self.set_guide_always_display = callback
        self.display_always.stateChanged.connect(self.guide_always_display)


        self.guide_area.setCurrentIndex(0)
        self.pre.setEnabled(False)
        self.pre.setText(f'前面没有了')
        self.next.setText(f'下一页（2/9）')
        if display_always_flag:
            self.display_always.setChecked(True)
        else:
            self.display_always.setChecked(False)


    def guide_always_display(self):
        if self.display_always.isChecked():
            val = True
        else:
            val = False
        self.set_guide_always_display('guide_always_display', val)
        
    def next_image(self):
        # 切换到下一张图片
        self.pre.setEnabled(True)
        current_index = self.guide_area.currentIndex()
        next_index = current_index + 1
        self.guide_area.setCurrentIndex(next_index)
        self.pre.setText(f'上一页（{current_index+1}/{self.guide_area.count()}）')
        if next_index == self.guide_area.count()-1:
            self.next.setEnabled(False)
            self.next.setText('后面没有了')
        else:
            self.next.setText(f'下一页（{next_index+2}/{self.guide_area.count()}）')


    def prev_image(self):
        # 切换到上一张图片
        self.next.setEnabled(True)
        current_index = self.guide_area.currentIndex()
        pre_index = current_index - 1
        self.guide_area.setCurrentIndex(pre_index)
        self.next.setText(f'下一页（{current_index+1}/{self.guide_area.count()}）')
        if pre_index == 0:
            self.pre.setEnabled(False)
            self.pre.setText('前面没有了')
        else:
            self.pre.setText(f'上一页（{pre_index}/{self.guide_area.count()}）')

def _build_mode_menu(callback):
    """构建工作流模式选择菜单（纯文本 + tooltip 说明）"""
    from PySide6.QtWidgets import QMenu

    menu = QMenu()
    menu.setToolTipsVisible(True)
    menu.setStyleSheet("""
        QMenu {
            font-size: 13px;
            padding: 4px 0px;
        }
        QMenu::item {
            padding: 6px 20px;
        }
        QMenu::item:selected {
            background-color: rgba(0, 120, 215, 60);
        }
    """)

    actions = [
        ("星轨叠加", "将多张照片合成星轨效果，支持多种叠加算法"),
        ("堆栈降噪", "对多张照片进行堆栈平均/中值降噪处理"),
        ("星点对齐叠加", "分离天地后对齐星点进行叠加，减少噪点并保持地景清晰"),
    ]
    for mode_name, tooltip in actions:
        action = menu.addAction(mode_name)
        action.setToolTip(tooltip)

    menu.triggered.connect(lambda act: callback(act.text()))
    return menu

class HNW_window(QMainWindow, Ui_HNW):

    def __init__(self):
        super().__init__()

        self.init_window()
        self.initial_attr()
        # 先绑定再初始化ui设置，避免初始化选项时部分关联槽函数未触发
        self.binding_slot()
        self.initial_settings()

        # self.alter_png_level.setEnabled(False)

        # 启动guide页面
        if self._CONFIG['guide_always_display']:
            time.sleep(0.5)
            self.slot_handler.show_guide_window()

    def hover_border_frame(self):
        '''
        创建覆盖在四周的8个边框frame 实现缩放检测并完成缩放
        '''

        def set_border_style(frame: borderFrame):
            """ 
            设置 QFrame 的外观 
            """
            frame.setStyleSheet(
                "background-color: rgba(200,200,0,0);border: 0px solid rgba(0, 220, 0, 250)"
            )

        # 创建8个 QFrame
        self.top_border = borderFrame(position='top', parent=self)
        self.bottom_border = borderFrame(position='bottom', parent=self)
        self.left_border = borderFrame(position='left', parent=self)
        self.right_border = borderFrame(position='right', parent=self)
        self.top_left_corner = borderFrame(position='top_left', parent=self)
        self.top_right_corner = borderFrame(position='top_right', parent=self)
        self.bottom_left_corner = borderFrame(position='bottom_left',
                                              parent=self)
        self.bottom_right_corner = borderFrame(position='bottom_right',
                                               parent=self)

        # 设置 QFrame 的样式 (红色背景)
        set_border_style(self.top_border)
        set_border_style(self.bottom_border)
        set_border_style(self.left_border)
        set_border_style(self.right_border)
        set_border_style(self.top_left_corner)
        set_border_style(self.top_right_corner)
        set_border_style(self.bottom_left_corner)
        set_border_style(self.bottom_right_corner)

        # 设置默认的初始大小和位置
        self.resizeEvent(None)

    def resizeEvent(self, event):
        """ 在窗口大小改变时调整 QFrame 的位置和大小 """
        # 获取当前窗口的尺寸
        window_width = self.width()
        window_height = self.height()
        border_width = 3  # 设定边框的宽度

        # 顶部
        self.top_border.setGeometry(border_width, 0,
                                    window_width - 2 * border_width,
                                    border_width)
        # 底部
        self.bottom_border.setGeometry(border_width,
                                       window_height - border_width,
                                       window_width - 2 * border_width,
                                       border_width)
        # 左侧
        self.left_border.setGeometry(0, border_width, border_width,
                                     window_height - 2 * border_width)
        # 右侧
        self.right_border.setGeometry(window_width - border_width,
                                      border_width, border_width,
                                      window_height - 2 * border_width)
        # 左上
        self.top_left_corner.setGeometry(0, 0, border_width, border_width)
        # 右上
        self.top_right_corner.setGeometry(window_width - border_width, 0,
                                          border_width, border_width)
        # 左下
        self.bottom_left_corner.setGeometry(0, window_height - border_width,
                                            border_width, border_width)
        # 右下
        self.bottom_right_corner.setGeometry(window_width - border_width,
                                             window_height - border_width,
                                             border_width, border_width)

        super().resizeEvent(event)  # 保持父类的 resizeEvent 行为

    def mousePressEvent(self, event: QMouseEvent):
        '''
        识别鼠标事件类型
        如果窗口未最大化 在鼠标按下时更新resize_x_y属性 
        以避免缩放过程中持续更新resize_x_y导致通过缩放进行最大化之后再最小化无法恢复正常大小
        拖拽事件仅在顶部生效
        '''
        self.resizing = False
        self.dragging = False
        self._hide_snap_preview()
        self.snap_zone = None
        if self.isMaximized() or self.snap_state:
            pass
        else:
            self.resize_x_y = [self.width(), self.height()]
        if event.button() == Qt.LeftButton:
            # 记录按下的位置，用于拖动或调整大小
            self.drag_position = event.globalPosition().toPoint(
            ) - self.frameGeometry().topLeft()
            # 如果鼠标在边缘，则标记为正在调整大小
            self.cursor_shape = {
                'top':
                True if self.top_border.cursor().shape() == Qt.SizeVerCursor
                else False,
                'top_right':
                True if self.top_right_corner.cursor().shape()
                == Qt.SizeBDiagCursor else False,
                'right':
                True if self.right_border.cursor().shape() == Qt.SizeHorCursor
                else False,
                'bottom_right':
                True if self.bottom_right_corner.cursor().shape()
                == Qt.SizeFDiagCursor else False,
                'bottom':
                True if self.bottom_border.cursor().shape() == Qt.SizeVerCursor
                else False,
                'bottom_left':
                True if self.bottom_left_corner.cursor().shape()
                == Qt.SizeBDiagCursor else False,
                'left':
                True if self.left_border.cursor().shape() == Qt.SizeHorCursor
                else False,
                'top_left':
                True if self.top_left_corner.cursor().shape()
                == Qt.SizeFDiagCursor else False
            }
            if any(self.cursor_shape.values()):
                self.resizing = True
            else:
                # 只在顶部生效
                if event.position().y() <= 40:
                    self.dragging = True
                else:
                    pass

    def mouseMoveEvent(self, event: QMouseEvent):
        '''
        响应鼠标移动事件
        包括拖拽 大小调整
        '''
        # 调整窗口大小
        if self.resizing:
            self.resize_window(event)
        # 拖动窗口
        elif self.dragging:
            # 如果是最大化 则进入窗口化
            if self.isMaximized():
                pos = event.globalPosition().toPoint()
                x_p = pos.x()
                y_p = pos.y()
                # 先调用 showNormal 恢复窗口化，再计算目标位置
                self.slot_handler.ui_max(target_type='window')
                # 计算窗口应该出现的位置（光标相对于窗口的比例位置）
                x_n = x_p - x_p / self.screen_width * self.resize_x_y[0]
                y_n = y_p - y_p / self.screen_height * self.resize_x_y[1]
                self.move(int(x_n), int(y_n))
                # 更新 drag_position 为光标相对于新窗口位置的偏移
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            # 如果是半屏 snap 状态，恢复原始尺寸
            elif self.snap_state:
                pos = event.globalPosition().toPoint()
                x_p = pos.x()
                y_p = pos.y()
                self.snap_state = None
                x_n = x_p - x_p / self.screen_width * self.resize_x_y[0]
                y_n = y_p - y_p / self.screen_height * self.resize_x_y[1]
                self.setGeometry(int(x_n), int(y_n), self.resize_x_y[0], self.resize_x_y[1])
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            else:
                pos = event.globalPosition().toPoint()
                x_p = pos.x()
                y_p = pos.y()

                # Snap zone detection (10px threshold)
                snap_threshold = 10
                snap_zone = None

                if x_p <= self.screen_x + snap_threshold:
                    snap_zone = 'left'
                elif x_p >= self.screen_x + self.screen_width - snap_threshold:
                    snap_zone = 'right'
                elif y_p <= self.screen_y + snap_threshold:
                    snap_zone = 'top'

                if snap_zone:
                    self._show_snap_preview(snap_zone)
                    self.snap_zone = snap_zone
                else:
                    self._hide_snap_preview()
                    self.snap_zone = None
                    self.move(event.globalPosition().toPoint() - self.drag_position)
        else:
            pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        '''
        鼠标释放后重置resizing dragging状态
        响应特殊最大化最小化事件并重置状态
        '''
        self.resizing = False
        self.dragging = False
        self._hide_snap_preview()
        # 更新窗口大小信息
        if self.min_flag:
            self.slot_handler.ui_min()
            self.min_flag = False
        if hasattr(self, 'snap_zone') and self.snap_zone:
            self._apply_snap(self.snap_zone)
            self.snap_zone = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        '''
        双击切换最大化/窗口化
        '''
        if event.position().y() <= 40:
            if self.isMaximized():
                pos = event.globalPosition().toPoint()
                x_p = pos.x()
                y_p = pos.y()
                self.slot_handler.ui_max(target_type='window')
                x_n = x_p - x_p / self.screen_width * self.resize_x_y[0]
                y_n = y_p - y_p / self.screen_height * self.resize_x_y[1]
                self.move(int(x_n), int(y_n))
            else:
                self.slot_handler.ui_max(target_type='max')

    def resize_window(self, event: QMouseEvent):
        '''
        窗口缩放事件
        '''
        # 获取窗口当前的几何信息
        rect = self.frameGeometry()
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()

        pos = event.globalPosition().toPoint()
        x_p = pos.x()
        y_p = pos.y()

        # 如果鼠标移动到屏幕边缘 最大化
        _detect_width = 2
        if (x_p >= self.screen_x + self.screen_width - _detect_width or
                y_p >= self.screen_y + self.screen_height - _detect_width or
                x_p <= self.screen_x + _detect_width or
                y_p <= self.screen_y + _detect_width):
            self.slot_handler.ui_max(target_type='max')
        # 如果从最大化移动 执行窗口化
        elif self.isMaximized():
            self.slot_handler.ui_max(target_type='window')
        else:
            for pressed_part, is_pressed in self.cursor_shape.items():
                if is_pressed:
                    break

            if pressed_part == 'top':
                h_n = h - (y_p - y)
                # 移动后高度超出最大最小范围 不更改高度和位置信息 其它同理
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n
                    y = y_p
            elif pressed_part == 'bottom':
                h_n = y_p - y
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n
            elif pressed_part == 'left':
                w_n = w - (x_p - x)
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
                    x = x_p
            elif pressed_part == 'right':
                w_n = x_p - x
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
            elif pressed_part == 'top_left':
                h_n = h - (y_p - y)
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n
                    y = y_p

                w_n = w - (x_p - x)
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
                    x = x_p
            elif pressed_part == 'bottom_right':
                h_n = y_p - y
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n

                w_n = x_p - x
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
            elif pressed_part == 'top_right':
                h_n = h - (y_p - y)
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n
                    y = y_p

                w_n = x_p - x
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
            elif pressed_part == 'bottom_left':
                h_n = y_p - y
                if h_n < self.minimumHeight() or h_n > self.maximumHeight():
                    pass
                else:
                    h = h_n

                w_n = w - (x_p - x)
                if w_n < self.minimumWidth() or w_n > self.maximumWidth():
                    pass
                else:
                    w = w_n
                    x = x_p

            # 更新窗口大小
            # 图标更新 避免在最大化时缩放窗口导致的图标未切换
            self.ui_max.setIcon(QIcon(u":/icons/resource/icon/max.png"))
            rect.setRect(x, y, w, h)
            self.setGeometry(rect)

    def _snap_geometry(self, zone: str) -> QRect:
        """Return the target geometry for a given snap zone."""
        screen = QApplication.primaryScreen().availableGeometry()
        w, h = screen.width(), screen.height()
        x, y = screen.x(), screen.y()
        if zone == 'left':
            return QRect(x, y, w // 2, h)
        elif zone == 'right':
            return QRect(x + w // 2, y, w - w // 2, h)
        else:  # 'top' → full screen (maximize)
            return QRect(x, y, w, h)

    def _show_snap_preview(self, zone: str):
        geo = self._snap_geometry(zone)
        self._snap_preview.setGeometry(geo)
        self._snap_preview.show()
        self._snap_preview.raise_()

    def _hide_snap_preview(self):
        self._snap_preview.hide()

    def _apply_snap(self, zone: str):
        if zone == 'top':
            if not self.isMaximized():
                self.resize_x_y = [self.width(), self.height()]
            self.snap_state = None
            self.slot_handler.ui_max(target_type='max')
        else:
            self.snap_state = zone
            geo = self._snap_geometry(zone)
            self.setGeometry(geo)

    def init_window(self):
        '''
        初始化子窗口
        '''
        self.setupUi(self)

        # 初始化软件配置信息
        self._CONFIG = {
            'config_file' : 'config',
            # 'config_path_win' : f'{os.path.expanduser("~")}\\AppData\\Roaming\\HoshiNoWeaver', 
            'config_path_win' : os.path.join(os.path.expanduser("~"),"AppData","Roaming","HoshiNoWeaver"),
            # 'config_path_mac' : f'{os.path.expanduser("~")}\\Library\\Application Support\\HoshiNoWeaver', 
            'config_path_mac' : os.path.join(os.path.expanduser("~"),"Library","Application Support","HoshiNoWeaver"),
            'guide_always_display' : True
        }
        if platform.system() == 'Windows':
            self._CONFIG['OS'] = 'Windows'
            self._CONFIG['config_path'] = self._CONFIG['config_path_win']
        elif platform.system() == 'Darwin':
            self._CONFIG['OS'] = 'MacOS' 
            self._CONFIG['config_path'] = self._CONFIG['config_path_mac']
        else:
            self._CONFIG['OS'] = 'Others' 
            self._CONFIG['config_path'] = ''
        # 读取配置信息
        self.read_config()
        # 更新配置信息 主要是将当前启动后获取到的新的配置信息写入配置文件
        self.update_config_file()

        # 设置窗口的标题
        self.setWindowTitle("HNW-织此星辰")
        # 设置窗口的图标
        self.setWindowIcon(QIcon(u":/icons/resource/icon/HNW.jpg"))
        # 设置无边框
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)

        # 控件基础属性
        # 标记是否正在调整窗口大小
        self.resizing = False
        # 标记是否正在拖动窗口
        self.dragging = False
        self.drag_position = QPoint()
        # 标记边缘缩放事件触发情况
        self.cursor_shape = {
            'top': False,
            'top_right': False,
            'right': False,
            'bottom_right': False,
            'bottom': False,
            'bottom_left': False,
            'left': False,
            'top_left': False,
        }
        # 记录最大化前的状态 不直接使用内置的normalGeometry
        self.resize_x_y = [self.width(), self.height()]
        # 屏幕大小（用 availableGeometry 支持多显示器和任务栏）
        screen_geo = QApplication.primaryScreen().availableGeometry()
        self.screen_height = screen_geo.height()
        self.screen_width = screen_geo.width()
        self.screen_x = screen_geo.x()
        self.screen_y = screen_geo.y()
        # 最小化标记 最大化标记 用于在缩放、拖动操作中控制特殊行为
        self.min_flag = False
        self.max_flag = False
        # Snap 区域标记（left / right / top / None）
        self.snap_zone = None
        # 半屏 Snap 状态（left / right / None），用于 drag-from-snap 时恢复原始尺寸
        self.snap_state = None

        # Snap 预览窗口
        self._snap_preview = SnapPreviewWindow()

        # 0 激活SlotHandler
        self.slot_handler = SlotHandler(self)
        # 1 模式切换菜单
        self.choose_mode_menu = _build_mode_menu(self.slot_handler.change_mode)
        # 2 添加主界面缩放检测边框
        self.hover_border_frame()
        # 3 guide页面
        self.guide_window = HNW_guide(callback=self.update_config, display_always_flag=self._CONFIG['guide_always_display'],parent=self)
        # 4 动态参数面板
        self._setup_dynamic_panel()
        # 5 动态输出面板
        self._setup_output_panel()

    def _setup_dynamic_panel(self):
        """Replace hardcoded parameter widgets with DynamicConfigPanel."""
        # Remove all children from self.frame (the settings container)
        layout = self.frame.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.hide()
                    w.deleteLater()

        # Create scroll area inside the existing frame
        scroll = QScrollArea(self.frame)
        scroll.setObjectName("config_scroll_area")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollBar:vertical { background-color: rgba(190,190,190,0); border: 0px; width: 6px; }"
            "QScrollBar::handle:vertical { background-color: rgba(190,190,190,190); border-radius: 3px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )

        self.config_panel = DynamicConfigPanel()
        scroll.setWidget(self.config_panel)
        layout.addWidget(scroll)

        self._current_meta_yaml_path = None

    def _setup_output_panel(self):
        """Replace static output tab widgets with OutputPanel."""
        layout = self.star_trail_output.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.hide()
                    w.deleteLater()

        scroll = QScrollArea(self.star_trail_output)
        scroll.setObjectName("output_scroll_area")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollBar:vertical { background-color: rgba(190,190,190,0); border: 0px; width: 6px; }"
            "QScrollBar::handle:vertical { background-color: rgba(190,190,190,190); border-radius: 3px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )

        self.output_panel = OutputPanel()
        scroll.setWidget(self.output_panel)
        layout.addWidget(scroll)

        # Re-evaluate readiness whenever an output value changes
        self.output_panel.values_changed.connect(
            lambda: self.slot_handler.detect_status() if hasattr(self, 'slot_handler') else None)

    def load_mode_panel(self, mode: str):
        """Load the dynamic panel for a given mode name."""
        if mode not in MODE_MAP:
            return
        meta_path, ui_path = MODE_MAP[mode]
        self._current_meta_yaml_path = meta_path
        schema = PanelSchema.from_yaml(meta_path, ui_path)
        self.config_panel.load_schema(schema)
        self.output_panel.load_specs(schema.outputs)
        from hoshicore.engine.wiring import load_output_defaults
        output_defaults = load_output_defaults()
        if output_defaults:
            self.output_panel.apply_defaults(output_defaults)

    def initial_attr(self, workspace='星轨叠加'):
        '''
        初始化实例属性
        '''
        # 属性定义
        # 任务运行状态(notStart/running/cancelled/successed/failed)和任务
        self._task = None
        self._status = 'notStart'
        self._status_n = {'status': '未就绪', 'tips': '请添加图像文件', 'tips_2': ''}

        self._workspace = workspace

        self._input_files = {
            '亮场': list(),
            '平场': list(),
            '暗场': list(),
            '偏置场': list(),
            '蒙版': list()
        }
        self._preview_useable = True

        # 进度条定义
        self.star_trail_process_bar.setValue(0)

        self._preview_img = ['', None]

    def initial_settings(self):
        '''
        初始化程序设置
        '''
        # 设置窗口为窗口化
        self.slot_handler.ui_max(target_type='window')

        # 页面初始化设置
        # 1 设置三个tab窗口的默认页面
        # self.main_tab.setCurrentIndex(0)
        self.star_trail_option_box.setCurrentIndex(0)

        # 2 设置按钮默认选中状态
        # 输出选项卡已由 OutputPanel 接管，无需手动初始化静态 widget


        # 4 文件列表初始化
        # 减少缩进
        self.star_trail_file_tree.setIndentation(10)
        # 隐藏标题行
        self.star_trail_file_tree.setHeaderHidden(True)
        self.star_trail_file_tree.header().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.star_trail_file_tree_l = QTreeWidgetItem(
            self.star_trail_file_tree, ['星空图像（0）'])
        # self.star_trail_file_tree_f = QTreeWidgetItem(self.star_trail_file_tree, ['平场（0）'])
        # self.star_trail_file_tree_d = QTreeWidgetItem(self.star_trail_file_tree, ['暗场（0）'])
        # self.star_trail_file_tree_b = QTreeWidgetItem(self.star_trail_file_tree, ['偏置场（0）'])
        # self.star_trail_file_tree_m = QTreeWidgetItem(self.star_trail_file_tree, ['蒙版（0）'])
        self.star_trail_file_tree_categore = {
            "亮场": self.star_trail_file_tree_l,
            # "平场" : self.star_trail_file_tree_f,
            # "暗场" : self.star_trail_file_tree_d,
            # "偏置场" : self.star_trail_file_tree_b,
            # "蒙版" : self.star_trail_file_tree_m
        }

        # 5 tip label字体设置
        font = QFont()
        font.setPointSize(12)
        self.star_trial_tips.setFont(font)

        # 6 设置图标

        # 7 设置文件列表允许允许多选
        self.star_trail_file_tree.setSelectionMode(
            QAbstractItemView.ExtendedSelection)

        # 旧的硬编码参数初始化已移除 — 由动态面板接管

        # 设置初始模式为星轨（使用新的动态面板）
        self.slot_handler.change_mode(self._workspace)


        # 设置进度条颜色
        self.star_trail_process_bar.setStyleSheet("#star_trail_process_bar {background-color: rgb(96, 200, 120);}")

    def binding_slot(self):
        '''
        绑定槽函数
        '''
        # 模式切换按钮
        self.label_current_mode.clicked.connect(
            self.slot_handler.show_choose_mode_window)
        # 最小化、最大化/窗口化、关闭按钮
        self.ui_close.clicked.connect(self.slot_handler.ui_close)
        self.ui_max.clicked.connect(lambda: self.slot_handler.ui_max(
            target_type='window' if self.isMaximized() else 'max'))
        self.ui_min.clicked.connect(self.slot_handler.ui_min)

        # setting按钮
        self.menu_setting.clicked.connect(self.slot_handler.show_setting_menu)
        self.menu_about.clicked.connect(self.slot_handler.show_about_dialog)

        # 图像列表选项卡
        # 6 添加文件
        self.add_files.clicked.connect(self.slot_handler.add_images)
        # 7 添加文件夹
        self.add_folder.clicked.connect(self.slot_handler.add_folder)
        # 8 清空文件列表
        self.clear_files.clicked.connect(
            lambda: self.slot_handler.clear_tree(categore_to_clear=None))

        # 10 文件列表菜单按钮
        self.star_trail_file_tree.menu_action_triggered_signal.connect(
            self.slot_handler.trigger_file_tree_item_menu)

        # 叠加选项 — 动态面板接管，旧绑定已移除

        # 输出选项 选项卡
        # 输出选项 — 由 OutputPanel 接管，原静态绑定已移除

        # 开始按钮
        self.btn_star_trail_start.clicked.connect(
            self.slot_handler.star_trail_start_process)

        # 分隔条拖动 先不用了
        # self.splitter.splitterMoved.connect(self.img_view_label.setImage)

        # 预览界面的左右按钮
        self.view_next_img.clicked.connect(lambda: self.slot_handler.view_next_img())
        self.view_pre_img.clicked.connect(lambda: self.slot_handler.view_pre_img())

    def read_config(self):
        config_path = self._CONFIG['config_path']
        config_file = self._CONFIG['config_file']
        if os.path.exists(os.path.join(config_path, config_file)):
            with open(os.path.join(config_path, config_file),'r',encoding='utf-8') as f:
                try:
                    data = json.loads(f.read())
                    self._CONFIG['guide_always_display'] = data['guide_always_display']
                except:
                    self.update_config()
        else:
            try:
                os.makedirs(config_path)
            except FileExistsError:
                pass
            with open(os.path.join(config_path, config_file),'w',encoding='utf-8') as f:
                f.write(json.dumps(self._CONFIG))

    def update_config(self,key,val):
        self._CONFIG[key] = val
        self.update_config_file()

    def update_config_file(self): 
        config_path = self._CONFIG['config_path']
        config_file = self._CONFIG['config_file']
        try:
            os.makedirs(config_path)
        except FileExistsError:
            pass
        with open(os.path.join(config_path, config_file),'w',encoding='utf-8') as f:
            f.write(json.dumps(self._CONFIG))


if __name__ == '__main__':
    if platform.system() == 'Windows':
        myappid = '.'.join(['org', ORG_NAME, SOFTWARE_NAME, VERSION.replace(".","_")])
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    _init_logger(_logger, debug_mode=False, trace_mode=False, log_path=None, task="gui")

    app = QApplication()
    app.setWindowIcon(QIcon(u":/icons/resource/icon/HNW.jpg"))

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window_inst = HNW_window()
    window_inst.show()

    with loop:
        loop.run_forever()
