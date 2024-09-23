from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QSize, QPointF, QRect, QPoint
from PySide6.QtWidgets import QLabel, QTreeWidget, QAbstractItemView, QMenu, QTreeWidgetItem, QDialog
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QVBoxLayout, QPushButton, QFrame
from PySide6.QtGui import QAction, QPixmap, QWheelEvent, QMouseEvent, QPainter, QImage, QColor, QBrush, QPolygon, QCursor

from ezlib.progressbar import QueueProgressbar
import rawpy

class imgDisplayQFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.scale_factor = 1.0
        self.max_scale = 10.0
        self.min_scale = 1.0
        self.offset = QPointF(0, 0)
        self.dragging = False
        self.last_mouse_pos = QPointF(0, 0)
        self.img_path = None


    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # 双击时重置缩放和位置
        if event.button() == Qt.LeftButton:
            self.setImage()

    def clear(self):
        self.img_path = None
        self.initImg()

    def initImg(self,path = None):
        if path is not None:
            self.img_path = path
        
            if self.img_path.split('\\')[-1].split('.')[-1].lower() in ['cr2','cr3','arw','nef','dng']:
                with rawpy.imread(self.img_path) as raw:
                    rgb = raw.postprocess()
                    
                # 将RGB数据转换为QImage
                height, width, channel = rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    
                # 将QImage转换为QPixmap并设置到QLabel
                pixmap = QPixmap.fromImage(q_image)
            else:
                pixmap = QPixmap(self.img_path)
            if not pixmap.isNull():
                self.pixmap = pixmap
                self.setImage()
        else:
            self.pixmap = None
            self.setImage()

    def setImage(self, scale_factor = None):
        self.updateScaleLimits()
        if scale_factor is None:
            # 加载图片时以最小比例展示
            self.scale_factor = self.min_scale
            # self.centerImage()
            # 重置图片位置偏移
            self.offset = QPointF(0, 0)
        else:
            self.scale_factor = scale_factor
        self.update()

    def centerImage(self):
        """Centers the image based on the current scale."""
        if self.pixmap:
            scaled_pixmap_size = self.pixmap.size() * self.scale_factor
            self.offset = QPointF(0, 0)
            print(self.width(), scaled_pixmap_size.width())
            print(self.height(), scaled_pixmap_size.height())
            self.offset.setX(0)
            self.offset.setY(0)
            print(self.offset)
    
    def updateScaleLimits(self):
        if self.pixmap:
            self.min_scale = min(self.width() / self.pixmap.width(), 
                                 self.height() / self.pixmap.height())

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            scaled_pixmap = self.pixmap.scaled(self.pixmap.size() * self.scale_factor, 
                                               Qt.KeepAspectRatio, 
                                               Qt.SmoothTransformation)
            
            x = (self.width() - scaled_pixmap.width()) / 2 + self.offset.x()
            y = (self.height() - scaled_pixmap.height()) / 2 + self.offset.y()
            painter.drawPixmap(x, y, scaled_pixmap)
        else:
            super().paintEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.pixmap:
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 0.85

            old_scale_factor = self.scale_factor
            new_scale_factor = self.scale_factor * factor

            # 限制缩放比例
            if new_scale_factor < self.min_scale:
                new_scale_factor = self.min_scale
            elif new_scale_factor > self.max_scale:
                new_scale_factor = self.max_scale

            if new_scale_factor == old_scale_factor:
                return

            # 更新缩放比例
            self.scale_factor = new_scale_factor

            # 获取鼠标位置相对于QFrame的坐标
            old_mouse_pos = event.position()

            # 计算缩放后鼠标在图片中的位置
            new_pixmap_size = self.pixmap.size() * self.scale_factor
            delta = (old_mouse_pos - self.offset - QPointF(self.width(), self.height()) / 2) * (new_scale_factor / old_scale_factor - 1)

            self.offset -= delta
            
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.pixmap:
            self.dragging = True
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            delta = event.position() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def resizeEvent(self, event):
        self.updateScaleLimits()
        super().resizeEvent(event)

class hoverDisplayButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hover_size = QSize(0, 0)
        self.original_size = QSize(0, 0)

    def enterEvent(self, event):
        self.setIconSize(self.hover_size)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIconSize(self.original_size)
        super().leaveEvent(event)

class ClickableLabel(QLabel):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ImgTreeWidget(QTreeWidget):
    menu_action_triggered_signal = Signal(str, QTreeWidgetItem)
    def __init__(self, parent=None):
        
        super().__init__(parent)
        self._menu_content = {
            '类别': ['展开', '折叠', '清空', '添加文件', '添加文件夹'],
            '类别多选': ['展开', '折叠', '清空'],
            '图像': ['预览', '从列表删除'],
            '图像多选': ['从列表删除']
        }
        self._disabled_menu_items = set()
        self.setHeaderHidden(True)

    def enterEvent(self, event):
        # 当鼠标进入 TreeWidget 时，将滚动条设置非透明
        self.verticalScrollBar().setStyleSheet("QScrollBar:vertical { background-color: rgba(190,190,190,190);border: 0px}")
        super().enterEvent(event)

    def leaveEvent(self, event):
        # 当鼠标离开 TreeWidget 时，将滚动条设置为透明
        self.verticalScrollBar().setStyleSheet("QScrollBar:vertical { background-color: rgba(190,190,190,0  );border: 0px}")
        super().leaveEvent(event)
    
    def add_disabled_menu_items(self, menu_items : set):
        self._disabled_menu_items = self._disabled_menu_items | menu_items

    def remove_disabled_menu_items(self, menu_items : set):
        self._disabled_menu_items = self._disabled_menu_items - menu_items

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        item = self.itemAt(position)
        selectd_item_cnt = len(self.selectedItems())

        # 鼠标右键点击事件
        if event.button() == Qt.RightButton:
            if item:
                # 如果已选多项且当前项已选，不触发选择当前项事件，否则选中当前项
                if selectd_item_cnt>1 and item.isSelected():
                    pass
                else:
                    self.setCurrentItem(item)
                # 切换单选多选模式
                self.switch_selection_mode(item)
                # 打开菜单
                self.open_menu(item, position, selectd_item_cnt)
        # 鼠标左键点击事件
        elif event.button() == Qt.LeftButton:
            if item:
                # 折叠或展开
                if item.isExpanded():
                    item.setExpanded(False)
                else:
                    item.setExpanded(True)
                # 切换单选多选模式
                self.switch_selection_mode(item)
        
        super().mousePressEvent(event)

    def open_menu(self, item, position, selectd_item_cnt):
        if item.parent() is None:
            # 类别节点
            menu_type = '类别'
            # if self.selectedItems() and all(i.parent() is not None for i in self.selectedItems()):
            #     menu_type = '类别多选'
            # else:
            #     menu_type = '类别'
        elif item.parent().parent() is None:
            # 图像节点
            if selectd_item_cnt > 1:
                menu_type = '图像多选'
            else:
                menu_type = '图像'

        menu = QMenu()
        if menu_type in self._menu_content:
            for menu_item in self._menu_content[menu_type]:
                action = QAction(menu_item, self)
                if menu_item in self._disabled_menu_items:
                    action.setEnabled(False)
                menu.addAction(action)
        
        action = menu.exec(self.viewport().mapToGlobal(position))
        if action:
            # 点击时触发menu_action_triggered_signal信号
            menu_text = action.text()
            self.menu_action_triggered_signal.emit(action.text(), item)

    def switch_selection_mode(self,item):
        current_mode = self.selectionMode()
        if item.parent() is None:
            # 类别节点
            # 如果当前是多选，切换为单选，否则不更改
            if current_mode == QAbstractItemView.ExtendedSelection:
                self.setSelectionMode(QAbstractItemView.SingleSelection)
            else:
                pass
        else: 
            # 图像节点
            # 如果当前是单选，切换为多选，否则不更改
            if current_mode == QAbstractItemView.SingleSelection:
                for selected_item in self.selectedItems():
                    if selected_item.parent() is None:
                        selected_item.setSelected(False)
                self.setSelectionMode(QAbstractItemView.ExtendedSelection)
            else:
                pass

class qtProgressBar(QueueProgressbar,QObject):
    progress_signal = Signal(int)

    def __init__(self, tot_num: int, desc: str = "") -> None:
        QObject.__init__(self)
        QueueProgressbar.__init__(self, tot_num, desc)

    def reset(self, tot_num: Optional[int] = 0, desc: str = "") -> None:
        super().reset(tot_num, desc)
        self.progress_signal.emit(0)

    def update(self):
        val = round(self.progress/self.tot_num*100,0)
        if val == 100:
            val = 99.9
        self.progress_signal.emit(val)
        
        # QApplication.processEvents()
        # time.sleep(1)

    def finish(self):
        self.progress_signal.emit(100)

class uQDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    def center_on_screen(self):
        # 获取主窗口的位置和大小
        parent_geometry = self.parent().window.geometry()
        parent_center = parent_geometry.center()

        # 计算对话框的位置
        x = parent_center.x() - self.width() // 2
        y = parent_center.y() - self.height() // 2
        self.move(x, y)

class exifCheckDialog(uQDialog):
    # 检查文件exif，弹出警告
    def __init__(self, parent = None, exif_check_result : dict = {}):
        super().__init__(parent)
        self._exif_check_result = exif_check_result
        self.setWindowTitle('警告！')
        self.setGeometry(100, 100, 200, 100)
        self.center_on_screen()
        
        text = '输入图像的'
        temp = []
        for item in self._exif_check_result:
            if item['attr_name'] == 'suffix' and item['other_dist'] != []:
                temp.append('文件格式')
            if item['attr_name'] == 'size' and item['other_dist'] != []:
                temp.append('像素尺寸')
            if item['attr_name'] == 'bits' and item['other_dist'] != []:
                temp.append('色深')
        text += '、'.join(temp)
        text += '存在不一致的情况，\n若继续叠加，可能叠加失败或叠加结果偏离预期，是否继续叠加？'

        layout = QVBoxLayout(self)
        self.info_lable = QLabel(self)
        self.info_lable.setText(text)
        layout.addWidget(self.info_lable)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = self.button_box.button(QDialogButtonBox.Ok)
        self.cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        self.ok_button.setText("继续")
        self.cancel_button.setText("取消")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)

class CategoryDialog(uQDialog):
    # 导入文件时选择文件类别
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择图像类别")
        self.setGeometry(100, 100, 200, 100)
        self.center_on_screen()

        layout = QVBoxLayout(self)
        self.combo_box = QComboBox(self)
        # self.combo_box.addItems(["星空图像", "平场", "暗场", "偏置场", "蒙版"])
        self.combo_box.addItems(["星空图像"])
        # self.combo_box.setMaximumWidth(80)
        layout.addWidget(self.combo_box)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = self.button_box.button(QDialogButtonBox.Ok)
        self.cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        self.ok_button.setText("下一步")
        self.cancel_button.setText("返回")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
        self.setLayout(layout)

    def selected_category(self):
        return '亮场' if self.combo_box.currentText() == '星空图像' else self.combo_box.currentText()

class borderFrame(QFrame):
    def __init__(self, position, parent=None):
        super().__init__(parent)
        # 设置 Frame 样式
        self.position = position
        
    
    def enterEvent(self, event):
        """ 当鼠标进入 Frame 时，将鼠标样式改为 SizeFDiagCursor """
        if self.position == 'top' or self.position == 'bottom': 
            self.setCursor(QCursor(Qt.SizeVerCursor))
        elif self.position == 'right' or self.position == 'left': 
            self.setCursor(QCursor(Qt.SizeHorCursor))
        elif self.position == 'top_left' or self.position == 'bottom_right': 
            self.setCursor(QCursor(Qt.SizeFDiagCursor))
        elif self.position == 'top_right' or self.position == 'bottom_left': 
            self.setCursor(QCursor(Qt.SizeBDiagCursor))
        super().enterEvent(event)

    def leaveEvent(self, event):
        """ 当鼠标离开 Frame 时，恢复鼠标样式为默认的箭头光标 """
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().leaveEvent(event)

class borderFrame_N():
    def __init__(self, window_width, window_height, border_width, parent=None):

        # 保存窗口宽度，高度和边框宽度
        self.window_width = window_width
        self.window_height = window_height
        self.border_width = border_width

        # 创建四个QFrame（上下左右边框）
        self.top_border = QFrame(parent)
        self.bottom_border = QFrame(parent)
        self.left_border = QFrame(parent)
        self.right_border = QFrame(parent)
        self.top_left_corner = QFrame(parent)
        self.top_right_corner = QFrame(parent)
        self.bottom_left_corner = QFrame(parent)
        self.bottom_right_corner = QFrame(parent)

        # 设置边框的样式，可以自定义颜色和样式
        self.set_frame_style(self.top_border)
        self.set_frame_style(self.bottom_border)
        self.set_frame_style(self.left_border)
        self.set_frame_style(self.right_border)
        self.set_frame_style(self.top_left_corner)
        self.set_frame_style(self.top_right_corner)
        self.set_frame_style(self.bottom_left_corner)
        self.set_frame_style(self.bottom_right_corner)

        # 调整边框位置和大小
        self.update_frames()

    def set_frame_style(self, frame):
        """ 设置边框的外观 """
        frame.setStyleSheet("background-color: red;")  # 红色背景
        frame.setFrameShape(QFrame.Box)  # 可选：添加边框效果

    def update_frames(self):
        """ 根据窗口的宽度、高度和边框宽度更新四个边框的位置和大小 """
        # 顶部 QFrame: 在窗口上边缘
        self.top_border.setGeometry(self.border_width, 0, self.window_width - 2 * self.border_width, self.border_width)
        # 底部 QFrame: 在窗口下边缘
        self.bottom_border.setGeometry(self.border_width, self.window_height - self.border_width, self.window_width - 2 * self.border_width, self.border_width)
        # 左侧 QFrame: 在窗口左边缘
        self.left_border.setGeometry(0, self.border_width, self.border_width, self.window_height - 2 * self.border_width)
        # 右侧 QFrame: 在窗口右边缘
        self.right_border.setGeometry(self.window_width - self.border_width, self.border_width, self.border_width, self.window_height - 2 * self.border_width)
        self.top_left_corner.setGeometry(0, 0, self.border_width, self.border_width)
        self.top_right_corner.setGeometry(self.window_width - self.border_width, 0, self.border_width, self.border_width)
        self.bottom_left_corner.setGeometry(0, self.window_height - self.border_width, self.border_width, self.border_width)
        self.bottom_right_corner.setGeometry(self.window_width - self.border_width, self.window_height - self.border_width, self.border_width, self.border_width)

    def resizeEvent(self, event):
        """ 当窗口大小改变时，自动调整四个边框的大小和位置 """
        self.window_width = self.width()
        self.window_height = self.height()
        self.update_frames()
        super().resizeEvent(event)

class DoubleSlider(QFrame):
    valueChanged = Signal(int, int)

    def __init__(self, parent=None):
        super(DoubleSlider, self).__init__(parent)

        # 控件的高度和宽度
        self.height_ = 20
        self.width_ = 130

        # 滑轨的最大最小值
        self.min_value = 0
        self.max_value = 100
        # 滑块的初始化值
        self.left_value = 20
        self.right_value = 80
        # 左右滑块的可移动范围
        # None表示不设置 此时会自动取min_value/max_value
        self.left_handdle_min_value = None
        self.left_handdle_max_value = None
        self.right_handdle_min_value = None
        self.right_handdle_max_value = None

        # 滑槽的高度宽度
        self.track_height = 2
        self.track_width = 110
        # 滑槽距离控件顶部、底部、左、右的距离
        # 该距离应给滑块的空间占用留足距离
        # self.height__track、height_track_margin_top、height_track_margin_bottom之和应等于self.height_，
        # 若不等于，优先级为self.height__track、height_track_margin_top、height_track_margin_bottom
        # width同理
        self.track_height_margin_top = 8
        self.track_height_margin_bottom = 9
        self.track_width_margin_left = 10
        self.track_width_margin_right = 10
        # 滑槽的中心线高度
        # self.track_center_height = self.track_height//2 if self.track_height%2 == 0 else self.track_height//2+1
        self.track_center_height = self.track_height/2
        self.track_center_height += self.track_height_margin_top
        # 滑槽颜色
        self.track_color = QColor(190, 190, 190, 190)

        # 左三角滑块的形状定义 分别为左顶点 上顶点 下顶点距离
        # 右滑块同理
        self.handdle_left_width = 5
        self.handdle_left_height_top = 7
        self.handdle_left_height_bottom = 7
        self.handdle_right_width = 5
        self.handdle_right_height_top = 7
        self.handdle_right_height_bottom = 7

        # 左右滑块的hover标志
        self.handdle_left_hovered = False
        self.handdle_right_hovered = False

        # 左右滑块的颜色、鼠标放置上去的颜色
        self.handdle_left_color = QColor("#5c5c5c")
        self.handdle_left_color_hover = QColor("#66cc66")
        self.handdle_right_color = QColor("#5c5c5c")
        self.handdle_right_color_hover = QColor("#66cc66")

        # 左划过部分滑槽距离滑槽顶部、底部、左、右的距离，高度、宽度 及颜色
        # 宽度应由当前的位置决定
        self.track_left_able_height_margin_top = 0
        self.track_left_able_height_margin_bottom = 0
        self.track_left_able_width_margin_left = 0
        self.track_left_able_width_margin_right = 0
        self.track_left_able_height = self.track_height
        self.track_left_able_width = (self.left_value - self.min_value) / (self.max_value - self.min_value) * self.track_width
        self.track_left_able_color = QColor("#66cc66")
        # 右划过部分滑槽距离控件顶部、底部、右的距离 及颜色
        self.track_right_able_height_margin_top = 0
        self.track_right_able_height_margin_bottom = 0
        self.track_right_able_width_margin_left = 0
        self.track_right_able_width_margin_right = 0
        self.track_right_able_height = self.track_height
        self.track_right_able_width = (self.max_value - self.right_value) / (self.max_value - self.min_value) * self.track_width
        self.track_right_able_color = QColor("#66cc66")

       
        # 滑块的初始位置
        self.left_handdle_pos = self.valueToPixel(self.left_value)
        self.right_handle_pos = self.valueToPixel(self.right_value)

        # 设置固定大小
        self.setFixedSize(self.width_, self.height_)
        # 启用鼠标跟踪
        self.setMouseTracking(True)

    def update_slider(self):
        self.track_center_height = self.track_height/2
        self.track_center_height += self.track_height_margin_top
        self.track_left_able_height = self.track_height
        self.track_left_able_width = (self.left_value - self.min_value) / (self.max_value - self.min_value) * self.track_width
        self.track_right_able_height = self.track_height
        self.track_right_able_width = (self.max_value - self.right_value) / (self.max_value - self.min_value) * self.track_width
        self.left_handdle_pos = self.valueToPixel(self.left_value)
        self.right_handle_pos = self.valueToPixel(self.right_value)
        self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        # 启用抗锯齿
        painter.setRenderHint(QPainter.Antialiasing)
        # 无边框
        painter.setPen(Qt.NoPen) 
        # 圆角
        xRadius = 2
        yRadius=2 

        # 绘制滑槽
        track_rect = QRect(self.track_width_margin_left, self.track_height_margin_top, self.track_width, self.track_height)
        painter.setBrush(QBrush(self.track_color))
        # 绘制圆角矩形
        painter.drawRoundedRect(track_rect, xRadius, yRadius)

        # 绘制左边已划过部分
        # 左上顶点x设置为滑槽距控件左边距离+已划过部分滑轨距离滑轨左边距离，y同理、
        # 宽度使用track_left_able_width，高度使用track_left_able_height
        track_left_able = QRect(
            self.track_width_margin_left - self.track_left_able_width_margin_left,
            self.track_height_margin_top - self.track_left_able_height_margin_bottom, 
            self.track_left_able_width,
            self.track_left_able_height)
        painter.setBrush(QBrush(self.track_left_able_color))
        # 绘制圆角矩形
        painter.drawRoundedRect(track_left_able, xRadius, yRadius)

        # 绘制右边已划过部分 类似左
        # 其左上顶点的x为右滑块位置+已划过部分滑轨距离滑轨左边距离，y同理、
        track_right_able = QRect(
            self.right_handle_pos - self.track_right_able_width_margin_left,
            self.track_height_margin_top - self.track_right_able_height_margin_bottom, 
            self.track_right_able_width,
            self.track_right_able_height)
        painter.setBrush(QBrush(self.track_right_able_color))
        # 绘制圆角矩形
        painter.drawRoundedRect(track_right_able, xRadius, yRadius)

        # 绘制左三角滑块
        self.drawTriangle(painter, 'left')
        # 绘制右三角滑块
        self.drawTriangle(painter, 'right')

        
    def drawTriangle(self, painter, direction):
        points = []

        # 定义顶点 按顺序为上 左/右 下
        # center_pos为滑块当前位置（x） 
        if direction == 'right':
            stard_x = self.right_handle_pos
            stard_y = self.track_center_height
            points = [
                QPoint(stard_x, stard_y-self.handdle_right_height_top),
                QPoint(stard_x+self.handdle_right_width, stard_y),
                QPoint(stard_x, stard_y+self.handdle_right_height_bottom)
            ]
            color = self.handdle_right_color_hover if self.handdle_right_hovered else self.handdle_right_color
        else: 
            stard_x = self.left_handdle_pos
            stard_y = self.track_center_height
            points = [
                QPoint(stard_x, stard_y-self.handdle_left_height_top),
                QPoint(stard_x-self.handdle_left_width, stard_y),
                QPoint(stard_x, stard_y+self.handdle_left_height_bottom)
            ]
            color = self.handdle_left_color_hover if self.handdle_left_hovered else self.handdle_left_color

        # 绘制多边形
        polygon = QPolygon(points)
        painter.setBrush(QBrush(color)) 
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(polygon)


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_pos = event.position().toPoint()
            self.click_handle = None

            # 距离
            left_diff = self.click_pos.x() - self.left_handdle_pos
            right_diff = self.right_handle_pos - self.click_pos.x()
            # 左滑块左边点击且距离为左滑块宽度的1.5倍定义为左边触发 右边同理
            if left_diff < 0 and abs(left_diff) <= self.handdle_left_width * 1.5:
                self.click_handle = 'left'
            elif right_diff < 0 and abs(right_diff) <= self.handdle_right_width * 1.5:
                self.click_handle = 'right'
            

    def handle_hover_detect(self, event):
        
        def is_in_triangle(self, event, vertext_1, vertext_2, vertext_3):
            def cross_product(x1, y1, x2, y2, x3, y3):
                # 向量叉积函数
                return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            in_triangle = False
            x1 = vertext_1[0]
            y1 = vertext_1[1]
            x2 = vertext_2[0]
            y2 = vertext_2[1]
            x3 = vertext_3[0]
            y3 = vertext_3[1]
            pos_x = event.position().toPoint().x()
            pos_y = event.position().toPoint().y()
            # 计算叉积
            cp1 = cross_product(x1, y1, x2, y2, pos_x, pos_y)
            cp2 = cross_product(x2, y2, x3, y3, pos_x, pos_y)
            cp3 = cross_product(x3, y3, x1, y1, pos_x, pos_y)
            # 判断点是否在三角形内
            if (cp1 > 0 and cp2 > 0 and cp3 > 0) or (cp1 < 0 and cp2 < 0 and cp3 < 0):
                in_triangle = True  # 点在三角形内
            else:
                in_triangle = False  # 点在三角形外
            return in_triangle
        # 检测是否在左滑块
        stard_x = self.left_handdle_pos
        stard_y = self.track_center_height
        # 扩大检测范围2像素
        vertext_1 = (stard_x, stard_y-self.handdle_left_height_top - 2)
        vertext_2 = (stard_x-self.handdle_left_width - 2, stard_y)
        vertext_3 = (stard_x, stard_y+self.handdle_left_height_bottom + 2)
        in_left_handdle = is_in_triangle(self, event, vertext_1, vertext_2, vertext_3)
        # 检测是否在右滑块
        stard_x = self.right_handle_pos
        stard_y = self.track_center_height
        vertext_1 = (stard_x, stard_y-self.handdle_right_height_top - 2)
        vertext_2 = (stard_x+self.handdle_right_width + 2, stard_y)
        vertext_3 = (stard_x, stard_y+self.handdle_right_height_bottom + 2)
        in_right_handdle = is_in_triangle(self, event, vertext_1, vertext_2, vertext_3)

        if in_left_handdle:
            self.handdle_left_hovered = True
        else:
            self.handdle_left_hovered = False
        if in_right_handdle:
            self.handdle_right_hovered = True
        else:
            self.handdle_right_hovered = False

    def mouseMoveEvent(self, event):
        self.handle_hover_detect(event)

        if event.buttons() & Qt.LeftButton:
            # pos为当前位置 横坐标
            pos = event.position().toPoint().x()
            # value为pos对应的值 但是可能超出滑块的最大最小范围
            value = round(self.pixelToValue(pos), 0)

            if self.click_handle == 'left':
                # 新的值应在滑轨最小、左滑块最小之上 滑轨最大、右滑块当前值、左滑块最大之下 right同理
                new_left_handdle_pos = self.valueToPixel(
                    max(
                        max(self.min_value,self.left_handdle_min_value if self.left_handdle_min_value is not None else self.min_value), 
                        min(value, self.right_value, self.left_handdle_max_value if self.left_handdle_max_value is not None else self.max_value)
                    )
                )
                if new_left_handdle_pos != self.left_handdle_pos:
                    self.left_handdle_pos = new_left_handdle_pos
                    if value != self.left_value:
                        # 避免检测精度不足导致的数值溢出 right同理
                        if value <= max(self.min_value,self.left_handdle_min_value if self.left_handdle_min_value is not None else self.min_value):
                            value = max(self.min_value,self.left_handdle_min_value if self.left_handdle_min_value is not None else self.min_value)
                        elif value >= min(self.right_value, self.left_handdle_max_value if self.left_handdle_max_value is not None else self.max_value):
                            value = min(self.right_value, self.left_handdle_max_value if self.left_handdle_max_value is not None else self.max_value)
                        self.left_value = value
                        self.track_left_able_width = (self.left_value - self.min_value) / (self.max_value - self.min_value) * self.track_width
                        self.valueChanged.emit(self.left_value, self.right_value)

            elif self.click_handle == 'right':
                new_right_handle_pos = self.valueToPixel(
                    min(
                        min(self.max_value, self.right_handdle_max_value if self.right_handdle_max_value is not None else self.max_value), 
                        max(value, self.left_value, self.right_handdle_min_value if self.right_handdle_min_value is not None else self.min_value)
                    )
                )
                if new_right_handle_pos != self.right_handle_pos:
                    self.right_handle_pos = new_right_handle_pos
                    if value != self.right_value:
                        if value >= min(self.max_value, self.right_handdle_max_value if self.right_handdle_max_value is not None else self.max_value):
                            value = min(self.max_value, self.right_handdle_max_value if self.right_handdle_max_value is not None else self.max_value)
                        elif value <= max(self.left_value, self.right_handdle_min_value if self.right_handdle_min_value is not None else self.min_value):
                            value = max(self.left_value, self.right_handdle_min_value if self.right_handdle_min_value is not None else self.min_value)
                        self.right_value = value
                        self.track_right_able_width = (self.max_value - self.right_value) / (self.max_value - self.min_value) * self.track_width
                        self.valueChanged.emit(self.left_value, self.right_value)

        self.update_slider()

    def mouseReleaseEvent(self, event):
        self.click_handle = None

    def enterEvent(self, event):
        pass

    def leaveEvent(self, event):
        pass

    def valueToPixel(self, value):
        return self.track_width_margin_left + (value - self.min_value) / (self.max_value - self.min_value) * self.track_width

    def pixelToValue(self, pos):
        return self.min_value + (pos - self.track_width_margin_left) / self.track_width * (self.max_value - self.min_value)