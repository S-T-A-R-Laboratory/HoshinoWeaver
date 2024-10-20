'''

'''
from __future__ import annotations
import sys
import os

import time
import platform
import json

from qasync import QEventLoop
import asyncio

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QHeaderView,QTreeWidgetItem, QAbstractItemView, QDialog
from PySide6.QtGui import QFont, QMouseEvent, QCursor, QColor, QIcon

from ui.UI import Ui_HNW
from ui.UI_choose_mode import HNW_choose_mode
from ui.UI_guide import Ui_guide
from ui.UIUtils import SlotHandler
from ui.UILibs import qtProgressBar
from ui.UILibs import borderFrame

import ctypes


class HNW_guide(QMainWindow, Ui_guide):
    def __init__(self, callback, display_always_flag=True):
        super().__init__()
        self.setupUi(self)  # 初始化通过 Qt Designer 生成的 UI
        
        # self.setWindowFlags(Qt.Popup)  # 设置为弹出窗口
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
        self.pre.setText(f'上一页（1/9）')
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
        else:
            self.pre.setText(f'上一页（{pre_index}/{self.guide_area.count()}）')

class HNW_choose_mode_window(QMainWindow, HNW_choose_mode):

    def __init__(self, callback):
        super().__init__()
        self.setWindowFlags(Qt.Popup)  # 设置为弹出窗口
        self.setWindowTitle("")

        self.setMouseTracking(True)
        # 创建一个 QTimer，初始设置为单次触发，超时后执行 self.close
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)  # 超时后关闭窗口

        self.setupUi(self)
        # 回调函数
        self.callback = callback
        self.label_3.clicked.connect(self.startrail_clicked)
        self.img_startrail.clicked.connect(self.startrail_clicked)
        self.label_4.clicked.connect(self.avg_clicked)
        self.img_avg.clicked.connect(self.avg_clicked)
        # self.back.clicked.connect(self.close)   去掉了关闭按钮

    def startrail_clicked(self):
        # 当按钮点击时，调用传递的回调函数
        self.callback('星轨叠加')
        self.close()

    def avg_clicked(self):
        # 当按钮点击时，调用传递的回调函数
        self.callback('堆栈降噪')
        self.close()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            # 鼠标在窗口内，取消定时器
            self.timer.stop()
        else:
            # 鼠标移出窗口，启动1秒的定时器
            if self.timer.isActive():
                pass
            else:
                self.timer.start(3000)

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
            time.sleep(1)
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
        if self.isMaximized():
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
                x_n = x_p - x_p / self.screen_width * self.resize_x_y[0]
                y_n = y_p - y_p / self.screen_height * self.resize_x_y[1]
                self.setGeometry(x_n, y_n, self.resize_x_y[0],
                                 self.resize_x_y[1])
                self.slot_handler.ui_max(target_type='window')
                # 这里移动了窗口 需要更新drag_position
                self.drag_position = event.globalPosition().toPoint(
                ) - self.frameGeometry().topLeft()
            # 如果移动到窗口顶部、侧部 则最大化
            elif event.globalPosition().toPoint().x(
            ) <= 0 or event.globalPosition().toPoint().x(
            ) >= self.screen_width - 0 or event.globalPosition().toPoint().y(
            ) <= 0:
                self.max_flag = True
            # 如果移动到窗口下侧 则最小化
            # 不支持这个选项了 会出发一些bug（移动上边框到最底部最小化后 无法恢复窗口）
            # elif event.globalPosition().toPoint().y() >= self.screen_height - 0:
            # self.min_flag = True
            else:
                self.move(event.globalPosition().toPoint() -
                          self.drag_position)
                self.max_flag = False
        else:
            pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        '''
        鼠标释放后重置resizing dragging状态
        响应特殊最大化最小化事件并重置状态
        '''
        self.resizing = False
        self.dragging = False
        # 更新窗口大小信息
        if self.min_flag:
            self.slot_handler.ui_min()
            self.min_flag = False
        if self.max_flag:
            self.slot_handler.ui_max(target_type='max')
            self.max_flag = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        ''' 
        双击切换最大化/窗口化
        '''
        if event.position().y() <= 40:
            if self.isMaximized():
                pos = event.globalPosition().toPoint()
                x_p = pos.x()
                y_p = pos.y()
                x_n = x_p - x_p / self.screen_width * self.resize_x_y[0]
                y_n = y_p - y_p / self.screen_height * self.resize_x_y[1]
                self.setGeometry(x_n, y_n, self.resize_x_y[0],
                                 self.resize_x_y[1])
                self.slot_handler.ui_max(target_type='window')
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
        if x_p >= self.screen_width - _detect_width or y_p >= self.screen_height - _detect_width or x_p <= _detect_width or y_p <= _detect_width:
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

    def init_window(self):
        '''
        初始化子窗口
        '''
        self.setupUi(self)

        # 初始化软件配置信息
        self._CONFIG = {
            'config_file' : 'config',
            'config_path_win' : f'{os.path.expanduser("~")}\\AppData\\Roaming\\HoshiNoWeaver', 
            'config_path_mac' : f'{os.path.expanduser("~")}\\Library\\Application Support\\HoshiNoWeaver', 
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
        # 屏幕大小
        self.screen_height = QApplication.primaryScreen().geometry().bottom()
        self.screen_width = QApplication.primaryScreen().geometry().right()
        # 最小化标记 最大化标记 用于在缩放、拖动操作中控制特殊行为
        self.min_flag = False
        self.max_flag = False

        # 0 激活SlotHandler
        self.slot_handler = SlotHandler(self)
        # 1 模式切换窗口
        self.choose_mode_window = HNW_choose_mode_window(
            self.slot_handler.change_mode)
        # 2 添加主界面缩放检测边框
        self.hover_border_frame()
        # 3 guide页面
        self.guide_window = HNW_guide(callback=self.update_config, display_always_flag=self._CONFIG['guide_always_display'])

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
        # 文件格式及文件路径
        self._output_file_type = 'JPG'
        # 缓存数据 用于在切换格式后保存之前输入的数据 切换回来之后不用重新输入
        self._output_file_path_cache = {
            'JPG': None,
            'PNG': None,
            'TIFF': None,
        }
        self._output_file_path = self._output_file_path_cache[
            self._output_file_type]
        # 算法
        self._mode = 'max'
        # 最大迭代次数 1-10 默认5
        self._max_iter = 5
        # 蒙版是否可用 可用的前提下 传入参数self._input_files['蒙版']
        self._mask_able = False
        self._mask_able = True
        # 渐入渐出
        self._fade_in = 0
        self._fade_out = 0
        # 拒绝域
        self._rej_low = 3
        self._rej_high = 3
        # int weight选项
        self._int_weight = False
        self._jpg_quality = 85
        self._png_compressing = 0
        # 色深
        self._output_bits = 8
        # 预览是否可用
        self._preview_useable = True

        self._resize = None
        self._input_size = [0, 0]

        # 进度条定义
        self.star_trail_process_bar.setValue(0)
        self.qtbar_star_trail = qtProgressBar(tot_num=0)

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
        # 默认输出jpg
        # 更改setCurrentText为png再改回jpg确保槽函数触发
        self.alter_output_type_2.setCurrentText('PNG')
        self.alter_output_type_2.setCurrentText('JPG')

        # 3 设置图像质量滑块默认值
        self.alter_jpg_level.setValue(85)
        self.alter_png_level.setValue(8)
        self.alter_resize.setValue(100)

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

        # s设置输出文件路径框为不可修改
        self.output_path_2.setReadOnly(True)

        # s设置蒙版路径框为不可修改
        self.mask_file_path.setReadOnly(True)

        # 初始化fade in out 显示 以及双滑块显示
        self.alter_fade_in_out.left_value = self._fade_in
        self.alter_fade_in_out.right_value = 100 - self._fade_out
        self.alter_fade_in_out.track_width_margin_left = 20
        self.alter_fade_in_out.width_ = 160
        self.alter_fade_in_out.track_width = 120
        self.alter_fade_in_out.update_slider()
        self.slot_handler.alter_fade_in_out()

        # 初始化alter_rejection显示 以及双滑块显示
        self.alter_rejection.left_value = int(0 - self._rej_low * 10)
        self.alter_rejection.right_value = int(self._rej_high * 10)
        self.alter_rejection.width_ = 160
        self.alter_rejection.track_width_margin_left = 20
        self.alter_rejection.track_width = 120
        self.alter_rejection.track_color = QColor("#66cc66")
        self.alter_rejection.track_left_able_color = QColor(190, 190, 190, 190)
        self.alter_rejection.track_right_able_color = QColor(
            190, 190, 190, 190)
        self.alter_rejection.min_value = -60
        self.alter_rejection.max_value = 60
        self.alter_rejection.left_handdle_min_value = None
        self.alter_rejection.left_handdle_max_value = 0
        self.alter_rejection.right_handdle_min_value = 0
        self.alter_rejection.right_handdle_max_value = None
        self.alter_rejection.update_slider()
        self.slot_handler.alter_rejection()

        # 初始化蒙版选项卡
        # self.slot_handler.mask_able()
        # 隐藏蒙版选项开启按钮 不需要这个按钮
        self.mask_able.hide()
        # 初始化int_weight选项
        self.slot_handler.int_weight_able()
        # 初始化max_iter、
        self.slot_handler.alter_max_iter()
        # 初始化_output_bits
        self.slot_handler.alter_output_bits()

        # 设置初始模式为星轨
        self.slot_handler.change_mode(self._workspace)

        # 隐藏resize选项
        self.frame_resize.hide()
        self.frame_resize.setVisible(False)

        # 设置进度条颜色
        self.star_trail_process_bar.setStyleSheet("#star_trail_process_bar {background-color: rgb(96, 200, 120);}")

    def binding_slot(self):
        '''
        绑定槽函数
        '''
        # 0 激活SlotHandler
        self.slot_handler = SlotHandler(self)

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

        # 叠加选项 选项卡
        # 算法选择切换
        self.alter_algorithm_startrail.currentTextChanged.connect(
            lambda: self.slot_handler.choose_algorithm_max())
        self.alter_algorithm_mean.currentTextChanged.connect(
            lambda: self.slot_handler.choose_algorithm_mean())
        self.alter_algorithm_min.currentTextChanged.connect(
            lambda: self.slot_handler.choose_algorithm_min())
        # 3 fade in out双滑块
        self.alter_fade_in_out.valueChanged.connect(
            self.slot_handler.alter_fade_in_out)
        # rej low、high双滑块
        self.alter_rejection.valueChanged.connect(
            self.slot_handler.alter_rejection)
        # 9 质量与速度选项
        self.int_weight_able.stateChanged.connect(
            self.slot_handler.int_weight_able)
        # 启用蒙版
        self.mask_able.stateChanged.connect(self.slot_handler.mask_able)
        # 选择蒙版
        self.alter_mask_file.clicked.connect(self.slot_handler.alter_mask_file)
        # 最大迭代次数
        self.alter_max_iter.valueChanged.connect(
            self.slot_handler.alter_max_iter)

        # 输出选项 选项卡
        # 2 星轨页面的文件格式下拉框
        self.alter_output_type_2.currentTextChanged.connect(
            self.slot_handler.output_file_option_2_switch)
        # 4 输出文件选择框
        self.alter_output_2.clicked.connect(self.slot_handler.save_img)
        # 5 图像质量值显示
        self.alter_png_level.valueChanged.connect(
            lambda value: self.slot_handler.alter_png_level(int(value)))
        self.alter_jpg_level.valueChanged.connect(
            lambda value: self.slot_handler.alter_jpg_level(int(value)))

        # self.alter_png_level.valueChanged.connect(lambda value: self.slot_handler.update_png_compressing(int(value)))
        # self.alter_jpg_level.valueChanged.connect(lambda value: self.slot_handler.update_jpg_quality(int(value)))
        # self.alter_png_level.valueChanged.connect(lambda value: self.png_level.setText(str(value)))
        # self.alter_jpg_level.valueChanged.connect(lambda value: self.jpg_level.setText(str(value)))
        # 色深
        self.alter_output_bits.currentTextChanged.connect(
            self.slot_handler.alter_output_bits)
        # 输出尺寸
        self.alter_resize.valueChanged.connect(
            lambda value: self.slot_handler.update_resize(int(value)))

        # 开始按钮
        self.btn_star_trail_start.clicked.connect(
            self.slot_handler.star_trail_start_process)
        # 进度条
        self.qtbar_star_trail.progress_signal.connect(
            self.slot_handler.update_progress_bar)

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
    myappid = 'mycompany.myproduct.subproduct.version'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication()
    app.setWindowIcon(QIcon(u":/icons/resource/icon/HNW.jpg"))

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window_inst = HNW_window()
    window_inst.show()

    with loop:
        loop.run_forever()
