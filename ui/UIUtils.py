'''
界面操作响应函数或方法
'''

import re
import asyncio
from qasync import asyncSlot
from PySide6.QtCore import Slot,QSize, Qt, QPoint, QSize
from PySide6.QtWidgets import QFileDialog, QMainWindow,QDialog,QTreeWidgetItem,QPushButton,QHBoxLayout,QWidget,QMenu
from PySide6.QtGui import QIcon, QCursor, QBrush, QColor

# 导入自定义组件
from ui.UILibs import ClickableLabel,exifCheckDialog,CategoryDialog,QtSignalTracker
# 导入图标资源
from ui import resource

# 导入Core接口
from hoshicore.engine.wiring import run_from_yaml
from hoshicore.component.image_io import scan_all_exif

class SlotHandler(QMainWindow):
    # 文件检查约束级别：normal-异常情况仅提示；strong-异常情况不允许叠加
    _file_constraint = {
        'suffix':'normal',
        'size':'normal',
        'bits':'normal'
    }

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("槽函数")


    @Slot()
    def ui_close(self):
        self.window.close()

    @Slot()
    def ui_min(self):
        self.window.showMinimized()

    @Slot()
    def ui_max(self, target_type = 'window'):
        '''
        窗口最大化、窗口化切换
        '''
        if target_type == 'window':
            self.window.showNormal()
            self.window.ui_max.setIcon(QIcon(u":/icons/resource/icon/max.png"))
            self.window.ui_max.setToolTip('最大化')
        else:
            self.window.showMaximized()
            self.window.ui_max.setIcon(QIcon(u":/icons/resource/icon/win.png"))
            self.window.ui_max.setToolTip('窗口化')

    # choose_algorithm_mean/max/min — 已移除，由 DynamicConfigPanel 接管

    @Slot()
    def show_choose_mode_window(self):
        '''
        显示选择模式的弹窗 置于标题栏的居中位置
        '''
        button_pos = self.window.label_current_mode.mapToGlobal(QPoint(0, self.window.label_current_mode.height()))
        new_x = button_pos.x() - (self.window.choose_mode_window.width() - self.window.label_current_mode.width()) / 2
        new_button_pos = QPoint(new_x, button_pos.y())
        self.window.choose_mode_window.move(new_button_pos)
        self.window.choose_mode_window.show()
        self.window.choose_mode_window.timer.start(5000)

    @Slot()
    def show_guide_window(self):
        '''
        显示使用说明的的弹窗 置于主窗口的中心位置
        '''
        guide_geometry = self.window.guide_window.frameGeometry()
        HNW_geometry = self.window.frameGeometry()
        pos_x = HNW_geometry.x() + HNW_geometry.width() / 2 - guide_geometry.width() / 2
        pos_y = HNW_geometry.y() + HNW_geometry.height() / 2 - guide_geometry.height() / 2
        guide_window_pos = QPoint(0 if pos_x < 0 else pos_x, 0 if pos_y < 0 else pos_y)
        self.window.guide_window.move(guide_window_pos)
        self.window.guide_window.show()

    @Slot()
    def change_mode(self, mode):
        '''
        响应选择的模式 — 加载对应的动态参数面板 + 设置背景图
        '''
        self.window.m_flag = False
        self.window.setCursor(QCursor(Qt.ArrowCursor))
        self.window.label_current_mode.setText(mode)

        # 加载动态面板
        self.window.load_mode_panel(mode)

        # 设置背景图
        bg_map = {
            '星轨叠加': 'url(:/img/resource/img/皿仓山星轨-s.jpg)',
            '堆栈降噪': 'url(:/img/resource/img/back02.jpg)',
            '天地分离': 'url(:/img/resource/img/皿仓山星轨-s.jpg)',
        }
        bg = bg_map.get(mode, 'url(:/img/resource/img/back02.jpg)')
        self.window.main_frame.setStyleSheet(f"""
            #main_frame {{
                border:none;
                background-image: {bg};
                background-repeat: no-repeat;
                background-position: center;
            }}
        """)

        self.window.dragging = False
        self.window.resizing = False

    @Slot()
    def show_setting_menu(self):
        self.menu = QMenu(self)
        self.menu_show_guide = self.menu.addAction("使用指南")
        self.menu_show_guide.triggered.connect(self.show_guide_window)

        # 将按钮点击与显示菜单绑定
        button_pos = self.window.menu_setting.mapToGlobal(QPoint(0, self.window.menu_setting.height()))
        self.menu.popup(button_pos)  # 弹出菜单，位置是按钮下方

    @Slot()
    def output_file_option_2_switch(self):
        '''
        响应文件类型选择
        '''
        # # 设置tab页
        # 改用隐藏控件实现 不再通过tab页实现
        output_file_type = self.window.alter_output_type_2.currentText()
        if output_file_type == 'TIFF':
            # 设置压缩级别隐藏 图片质量隐藏
            self.window.frame_png_level.hide()
            self.window.frame_png_level.setVisible(False)
            self.window.frame_jpg_level.hide()
            self.window.frame_jpg_level.setVisible(False)
            # 启用色深下拉选项
            self.window.alter_output_bits.model().item(1).setEnabled(True)
            self.window.alter_output_bits.model().item(1).setForeground(QBrush(QColor(35,35,35,210)))
            self.window.alter_output_bits.model().item(2).setEnabled(True)
            self.window.alter_output_bits.model().item(2).setForeground(QBrush(QColor(35,35,35,210)))
        elif output_file_type == 'JPG':
            # 设置压缩级别隐藏 图片质量可见
            self.window.frame_png_level.hide()
            self.window.frame_png_level.setVisible(False)
            self.window.frame_jpg_level.show()
            self.window.frame_jpg_level.setVisible(True)
            # 设置色深下拉选项为8bit并禁用其他选项
            self.window.alter_output_bits.setCurrentText('8 bit')
            self.window.alter_output_bits.model().item(1).setEnabled(False)
            self.window.alter_output_bits.model().item(1).setForeground(QBrush(QColor(35,35,35,140)))
            self.window.alter_output_bits.model().item(2).setEnabled(False)
            self.window.alter_output_bits.model().item(2).setForeground(QBrush(QColor(35,35,35,140)))
        elif output_file_type == 'PNG':
            # 设置压缩级别可见 图片质量隐藏
            self.window.frame_png_level.show()
            self.window.frame_png_level.setVisible(True)
            self.window.frame_jpg_level.hide()
            self.window.frame_jpg_level.setVisible(False)
            # 启用色深下拉选项
            # 在选择32bit时，设置色深下拉选项为16bit。此外 禁用32bit选项
            if self.window.alter_output_bits.currentText() == '32 bit':
                self.window.alter_output_bits.setCurrentText('16 bit')
            self.window.alter_output_bits.model().item(1).setEnabled(True)
            self.window.alter_output_bits.model().item(1).setForeground(QBrush(QColor(35,35,35,210)))
            self.window.alter_output_bits.model().item(2).setEnabled(False)
            self.window.alter_output_bits.model().item(2).setForeground(QBrush(QColor(35,35,35,140)))
        else:
            pass
        
        # 将所选的文件格式存入变量，根据选择的文件格式更新输出文件路径 如果新的文件格式下，此前已经填过路径，用之前的，如果之前没填过，则为空
        # 同步更新文件路径框显示的路径和tooltip显示文字
        self.window._output_file_type = output_file_type
        self.window._output_file_path = self.window._output_file_path_cache[output_file_type]
        self.window.output_path_2.setText(self.window._output_file_path)
        self.window.output_path_2.setToolTip((self.window._output_file_path))
        self.detect_status()
    
    # alter_fade_in_out, alter_rejection, mask_able, alter_mask_file,
    # int_weight_able, alter_max_iter, alter_output_bits — 已移除，由动态面板接管

    @Slot()
    def alter_png_level(self,val=None):
        if val:
            self.window._png_compressing = val 
        else:
            self.window._png_compressing = int(self.window.png_level.text())
        self.window.png_level.setText(str(self.window._png_compressing))
        self.detect_status()

    @Slot()
    def alter_jpg_level(self,val=None):
        if val:
            self.window._jpg_quality = int(val)
        else:
            self.window._jpg_quality = int(self.window.jpg_level.text())
        self.window.jpg_level.setText(str(self.window._jpg_quality))
        self.detect_status()

    @Slot(int, str)
    def update_progress_bar(self, percent, desc=''):
        '''
        更新进度条的槽函数
        '''
        self.window.star_trail_process_bar.setValue(percent)
        if percent < 100:
            self.window._status_n['tips'] = desc or f'当前进度{percent}%，请勿操作'
            self.window._status_n['status'] = f'处理中'
        else:
            self.window._status_n['tips'] = f'已完成~(文件路径：{self.window._output_file_path_cache[self.window._output_file_type]})'
            self.window._status_n['status'] = f'任务完成'
        self.update_status_display()

    @Slot()
    def view_next_img(self, img_list = None):
        category = self.window._preview_img[0]
        current_img = self.window._preview_img[1]
        if category == '':
            pass
        else:
            if img_list is None:
                img_list = self.window._input_files[category]
            flag = False
            if current_img == self.window._input_files[category][-1]:
                pass
            elif category is not None:
                for img in img_list:
                    if flag:
                        self.view_file(file_path = img, category = category)
                        break
                    elif img == current_img:
                        flag = True 

    @Slot()
    def view_pre_img(self, img_list = None):
        category = self.window._preview_img[0]
        current_img = self.window._preview_img[1]
        pre_img = None
        if img_list is None:
            img_list = self.window._input_files[category]
        if current_img == self.window._input_files[category][0]:
            pass
        elif category is not None:
            for img in img_list:
                if img == current_img:
                    self.view_file(file_path = pre_img, category = category)
                    break
                else:
                    pre_img = img

    @Slot()
    def save_img(self):
        # 打开文件浏览对话框
        options = QFileDialog.Options(0)
        # 允许保存的文件类型
        filter = 'JPG (*.jpg);;PNG (*.png);;TIFF (*.tif)'
        # 根据当前已选择的文件类型选择默认类型
        current_file_type = self.window.alter_output_type_2.currentText()
        selectedFilter = {'JPG':'JPG (*.jpg)','PNG':'PNG (*.png)','TIFF':'TIFF (*.tif)'}[current_file_type]
        
        file_path, choosed_file_type = QFileDialog.getSaveFileName(self, "保存文件", "", filter = filter, selectedFilter = selectedFilter, options=options)
        # 用户在该页面重新选择文件类型后，修改页面的文件格式和格式对应的额外选项tab页
        choosed_file_type = choosed_file_type.split(' ')[0]
        if current_file_type != choosed_file_type:
            self.window.alter_output_type_2.setCurrentText(choosed_file_type)
            self.update_output_file_type(choosed_file_type)
        # 将文件路径写入output_path_2
        if file_path:
            self.update_output_file_path_cache(choosed_file_type,file_path)
            # print(file_path)
        else:
            pass
        self.detect_status()

    @Slot()
    def add_folder(self, category = None):
        def open_dialog(self,category):
            folder_dialog = QFileDialog(self,caption='添加%s' % '星空图像' if category=='亮场' else category)
            folder_dialog.setFileMode(QFileDialog.Directory)
            if folder_dialog.exec_() == QDialog.Accepted:
                folder_path = folder_dialog.selectedUrls()[0].toLocalFile()
                self.add_file_to_tree_from_floder(folder_path, category)
            self.update_star_trail_file_tree_title(category)
            # 添加完成展开当前类别
            self.window.star_trail_file_tree_categore[category].setExpanded(True)
            temp = list(self.window.star_trail_file_tree_categore.keys())
            temp.remove(category)
            for _category in temp:
                self.window.star_trail_file_tree_categore[_category].setExpanded(False)

        if category:
            open_dialog(self,category)
        else:
            category_dialog = CategoryDialog(self)
            selected_items = self.window.star_trail_file_tree.selectedItems()
            if len(selected_items) == 1 and selected_items[0].parent() is None:
                selected_category = selected_items[0].text(0)
                category_dialog.combo_box.setCurrentText(selected_category.split('（')[0])
            if category_dialog.exec_() == QDialog.Accepted:
                category = category_dialog.selected_category()
                open_dialog(self,category)
        self.detect_status()

    def open_add_file_dialog(self,category):
        file_dialog = QFileDialog(self,caption='添加%s'%'星空图像' if category=='亮场' else category)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters([
                '全部支持文件(*.cr2 *.cr3 *.arw *.nef *.dng *.tiff *.tif *.jpeg *.jpg *.png *.bmp *.gif *.fits)',
                'RAW文件(*.cr2 *.cr3 *.arw *.nef *.dng)',
                'tif文件(*.tiff *.tif)',
                'jpg文件(*.jpeg *.jpg)',
                'png文件(*.png)',
                '其它图片文件(*.bmp *.gif *.fits)'
        ])
        if file_dialog.exec_() == QDialog.Accepted:
            file_paths = [url.toLocalFile() for url in file_dialog.selectedUrls()]
            for file_path in file_paths:
                # 如果文件路径已存在，不允许重复添加
                if file_path not in self.window._input_files[category]:
                    self.add_file_to_tree(file_path, category)
                else:
                    pass
            self.update_star_trail_file_tree_title(category)
            # 添加完成展开当前类别
            self.window.star_trail_file_tree_categore[category].setExpanded(True)
            temp = list(self.window.star_trail_file_tree_categore.keys())
            temp.remove(category)
            for _category in temp:
                self.window.star_trail_file_tree_categore[_category].setExpanded(False)
        self.detect_status()

    @Slot()
    def add_images(self, category = None):
        if category:
            self.open_add_file_dialog(category)
        else:
            category_dialog = CategoryDialog(self)
            selected_items = self.window.star_trail_file_tree.selectedItems()
            if len(selected_items) == 1 and selected_items[0].parent() is None:
                selected_category = selected_items[0].text(0)
                category_dialog.combo_box.setCurrentText(selected_category.split('（')[0])
            if category_dialog.exec_() == QDialog.Accepted:
                category = category_dialog.selected_category()
                self.open_add_file_dialog(category)
        self.detect_status()
                
    @Slot()
    def add_file_to_tree(self, file_path, category):
        # image = Image.open(file_path)
        # # 获取图像尺寸
        # width, height = image.size
        # print(f"Width: {width}, Height: {height}")

        category_item = self.window.star_trail_file_tree_categore[category]
        # 将文件添加至树
        file_name = file_path.split('/')[-1]

        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # 去除边距

        remove_button = QPushButton()
        remove_button.setToolTip('从列表移除该图像')
        icon_remove_button = (QIcon(u":/icons/resource/icon/delete.png"))
        remove_button.setIcon(icon_remove_button)
        remove_button.setMinimumSize(QSize(20, 20))
        remove_button.setMaximumSize(QSize(20, 20))
        remove_button.setGeometry(0,0,0,0)
        remove_button.clicked.connect(lambda: self.remove_file_from_tree(file_item, mode = 'SingleImg'))
        layout.addWidget(remove_button)

        view_button = QPushButton()
        view_button.setToolTip('预览')
        icon_view_button = (QIcon(u":/icons/resource/icon/preview.png"))
        view_button.setIcon(icon_view_button)
        view_button.setMinimumSize(QSize(20, 20))
        view_button.setMaximumSize(QSize(20, 20))
        view_button.clicked.connect(lambda: self.view_file(file_path, category))
        layout.addWidget(view_button)

        file_label = ClickableLabel(file_name)
        file_label.setToolTip(file_path)  # 设置鼠标悬停提示
        file_label.clicked.connect(lambda: self.view_file(file_path, category))
        layout.addWidget(file_label)

        layout.addStretch()  # 添加弹性空间
        widget.setLayout(layout)

        file_item = QTreeWidgetItem(category_item)
        file_item.__file_path = file_path
        file_item.__category = category
        file_item.__remove_bnt = remove_button
        file_item.__view_bnt = view_button
        file_item.__file_label = file_label
        self.window.star_trail_file_tree.setItemWidget(file_item, 0, widget)

        # 点击时将该条选中，取消其他已选中
        view_button.clicked.connect(lambda: [selected_item.setSelected(False) for selected_item in self.window.star_trail_file_tree.selectedItems()])
        view_button.clicked.connect(lambda: file_item.setSelected(True))
        # 添加文件时将文件路径添加至_input_files
        self.window._input_files[category].append(file_path)
        # print(self.window._input_files)

    @Slot()
    def add_file_to_tree_from_floder(self, folder_path, category):
        # 从文件夹添加所有符合格式的文件
        import os
        for root, _, files in os.walk(folder_path):
            for file in files:
                # 按支持的文件类型进行过滤
                if re.search('\.((cr2)|(cr3)|(arw)|(nef)|(dng)|(tiff)|(tif)|(jpeg)|(jpg)|(png)|(bmp)|(gif)|(fits))$', file.lower()):
                    file_path = '%s/%s'%(root, file)
                    file_path = file_path.replace('\\','/')
                    # 不允许重复添加
                    if file_path not in self.window._input_files[category]:
                        self.add_file_to_tree(file_path.replace('\\','/'), category)
                    else:
                        pass
        self.detect_status()

    @Slot()
    def remove_file_from_tree(self, file_item, mode='SingleImg'):
        category = file_item.__category
        file_path = file_item.__file_path
        # 从列表删除文件
        tree = file_item.parent()
        index = self.window.star_trail_file_tree.indexOfTopLevelItem(file_item)
        if index != -1:
            self.window.star_trail_file_tree.takeTopLevelItem(index)
        else:
            parent = file_item.parent()
            if parent:
                parent.removeChild(file_item)
        # 更新数量
        self.update_star_trail_file_tree_title(category)
        # 先从列表删除再预览下一张 避免加载下一张的时间开销影响流畅度
        # 从列表删除后无法再根据当前显示图片寻找上一张或下一张 因此先备份一个删除之前的文件列表 传入view_next_img/view_pre_img
        # 但不知道为什么还是会卡顿。。
        temp = [img for img in self.window._input_files[category]]# 删除文件时将文件路径从_input_files中删除
        # 从列表删除
        self.window._input_files[category].remove(file_path)
        # 清空时 若正在预览则清空预览 若删除单张 切换至当前类别的下一张 若无下一张 切换至上一张 若空了 清空
        if mode == 'SingleImg':
            if file_path == self.window._preview_img[1] and category == self.window._preview_img[0]:
                if file_path != temp[-1]:
                    self.view_next_img(img_list = temp)
                elif file_path != temp[0]:
                    self.view_pre_img(img_list = temp)
                else:
                    self.view_file()
            elif self.window._input_files[category][0] == self.window._preview_img[1] and category == self.window._preview_img[0]:
                # 如果删除后显示图片变成事实上的第一张，设置pre img按钮不再可点击
                self.window.view_pre_img.setStyleSheet("#view_pre_img:pressed {padding-bottom: 5px;}")
        else:
            self.view_file()
        self.detect_status()
        
    @Slot()
    def view_file(self, file_path : str = None, category : str = None):
        if not self.window._preview_useable:
            self.window._preview_img = ['', None] 
            self.window.view_next_img.hover_size = QSize(0, 0)
            self.window.view_pre_img.hover_size = QSize(0, 0)
        elif file_path is not None:
            self.window._preview_img = [category, file_path]
            self.window.img_view_label.initImg(file_path)
            if category is None:
                # 如果不传入类别 即不是从文件列表选项卡点进去的，隐藏左右按钮
                self.window.view_next_img.hover_size = QSize(0, 0)
                self.window.view_pre_img.hover_size = QSize(0, 0)
            else:
                # 如果传入类别 显示左右按钮
                self.window.view_next_img.hover_size = QSize(40, 40)
                self.window.view_pre_img.hover_size = QSize(40, 40)
                # 如果是各类别第一张图片 view_pre_img无法点击
                if self.window._input_files[category][0] == file_path:
                    self.window.view_pre_img.setStyleSheet("#view_pre_img:pressed {padding-bottom: 5px;}")
                else:
                    self.window.view_pre_img.setStyleSheet("#view_pre_img:pressed {padding-bottom: 0px;}")
                # 如果是各类别最后一张图片 view_next_img无法点击
                if self.window._input_files[category][-1] == file_path:
                    self.window.view_next_img.setStyleSheet("#view_next_img:pressed {padding-bottom: 5px;}")
                else:
                    self.window.view_next_img.setStyleSheet("#view_next_img:pressed {padding-bottom: 0px;}")
        else:
            self.window._preview_img = ['', None]
            self.window.img_view_label.clear()
            self.window.view_next_img.hover_size = QSize(0, 0)
            self.window.view_pre_img.hover_size = QSize(0, 0)

    @Slot()
    def clear_tree(self, categore_to_clear = None):
        # 清空文件列表
        # print(categore_to_clear)
        if categore_to_clear is None:
            categore_to_clear = self.window.star_trail_file_tree_categore.keys()
        # print(categore_to_clear)
        for category in categore_to_clear:
            tree = self.window.star_trail_file_tree_categore[category]
            tree.takeChildren()
            self.update_star_trail_file_tree_title(category)
            # 清空列表时清空_input_files
            self.window._input_files[category] = list()
        # 清空预览
        self.view_file()
        self.detect_status()
    
    @Slot()
    def update_star_trail_file_tree_title(self, category):
        category_item = self.window.star_trail_file_tree_categore[category]
        # 更新文件树的文件数量
        categore = category_item.text(0)
        file_cnt = category_item.childCount()
        new_categore = re.sub('\d+',str(file_cnt),categore)
        category_item.setText(0,new_categore)

    @Slot()
    def star_trail_start_process_bak(self):
        if self.window._input_files['亮场'] == []:
            self.display_star_trail_tips('请添加星空图像文件！',color='red')
        elif self.window._output_file_path is None :
            self.display_star_trail_tips('请设置输出路径！',color='red')
        else:
            continue_flag = True
            # 调用检查api
            exif_check_result = scan_all_exif(self.window._input_files['亮场'])
            # print(exif_check_result)
            if all([True if item['other_dist'] == [] else False  for item in exif_check_result]):
                continue_flag = True
            else:
                exif_check_dialog = exifCheckDialog(self, exif_check_result)
                if exif_check_dialog.exec_() == QDialog.Accepted:
                    continue_flag = True
                else:
                    continue_flag = False
            if continue_flag:
                self.display_star_trail_tips('正在叠加>>>>',color='red')
                self.window._task = asyncio.ensure_future(self.start_task())

    @Slot()
    def star_trail_start_process(self):
        if self.window._status_n['status'] == '未就绪':
            self.window.status_text.setStyleSheet("#status_text {color:rgba(200,0,0,200)}")
            self.window.star_trial_tips.setStyleSheet("#star_trial_tips {color:rgba(200,0,0,200)}")
        else:
            continue_flag = True
            # 调用检查api
            exif_check_result = scan_all_exif(self.window._input_files['亮场'])
            # print(exif_check_result)
            if all([True if item['other_dist'] == [] else False  for item in exif_check_result]):
                continue_flag = True
            else:
                exif_check_dialog = exifCheckDialog(self, exif_check_result)
                if exif_check_dialog.exec_() == QDialog.Accepted:
                    continue_flag = True
                else:
                    continue_flag = False
            if continue_flag:
                self.window._status_n['status'] = '处理中'
                self.update_status_display()
                # self.display_star_trail_tips('正在叠加>>>>',color='red')
                self.window.star_trail_process_bar.setStyleSheet("#star_trail_process_bar {background-color: rgba(2, 53, 57,50);}")
                self.window._task = asyncio.ensure_future(self.start_task())

    @asyncSlot()
    async def start_task(self):
        # 清空预览
        self.view_file()
        # 设置界面不可操作
        self.set_widget_handleable(handleable = False, task_type='star_trail')
        self.window._status = 'running'

        # 创建 Qt 进度追踪器并连接信号
        qt_tracker = QtSignalTracker()
        qt_tracker.progress_updated.connect(self.update_progress_bar)
        qt_tracker.finished.connect(lambda: self.update_progress_bar(100, ''))

        # 创建取消事件，供 cancel_task 触发
        self.window._cancel_event = asyncio.Event()

        # 从动态面板收集参数
        yaml_path = self.window._current_meta_yaml_path
        global_inputs = {"fnames": self.window._input_files['亮场']}
        global_configs = self.window.config_panel.collect_configs()
        route_choices = self.window.config_panel.collect_route_choices()

        # 确保 output_filename 有值（面板可能收集到 None）
        if not global_configs.get("output_filename"):
            global_configs["output_filename"] = self.window._output_file_path

        try:
            await run_from_yaml(
                yaml_path, global_inputs, global_configs,
                tracker=qt_tracker, progress=False,
                cancel_event=self.window._cancel_event,
                route_choices=route_choices)

            # 执行成功
            output_path = global_configs.get("output_filename", "")
            self.view_file(output_path)
            self.window._status = 'successed'
            self.window._status_n['status'] = '任务完成'
            self.window._status_n['tips_2'] = ''
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.window.status_text.setStyleSheet("#status_text {color:rgba(200,0,0,200)}")
            self.window.star_trial_tips.setStyleSheet("#star_trial_tips {color:rgba(200,0,0,200)}")
            self.view_file(file_path = '')
            self.window._status = 'failed'
            self.window._status_n['tips_2'] = ''
            self.window._status_n['status'] = '任务失败'
        finally:
            self.set_widget_handleable(handleable = True)
            self.update_status_display()
            

    @Slot()
    def cancel_task(self):
        if self.window._status == 'running':
            # 触发协作式取消：Op 在下一个 _run_cpu 检查点退出
            if hasattr(self.window, '_cancel_event'):
                self.window._cancel_event.set()
            # 取消 asyncio task 以打断 gather 层面的 await
            self.window._task.cancel()
            # 取消任务后 修改tips为已停止，修改状态为cancelled，不展示任何图像，恢复按钮为可点击，修改开始按钮为"开始叠加"
            self.display_star_trail_tips('已停止叠加！',color='red')
            self.view_file(file_path = '')
            self.window._status = 'cancelled'
            self.set_widget_handleable(handleable = True)
            
    @Slot()
    def display_star_trail_tips(self,text,color='red'):
        self.window.star_trial_tips.setStyleSheet("color: %s;" % color)
        self.window.star_trial_tips.setText(text)

    @Slot()
    def update_output_file_type(self,val='JPG'):
        self.window._output_file_type = val
        self.update_output_file_path_cache(file_type = val)
        self.detect_status()
        
    @Slot()
    def update_output_file_path_cache(self,file_type,val=None):
        # print(val)
        # print(self.window._output_file_type)
        # print(self.window._output_file_path_cache)
        if val:
            self.window._output_file_path_cache[file_type] = val 
        self.window._output_file_path = self.window._output_file_path_cache[self.window._output_file_type]
        self.window.output_path_2.setText(self.window._output_file_path)
        self.window.output_path_2.setToolTip((self.window._output_file_path))
        self.detect_status()

    @Slot()
    def update_resize(self,val=None):
        if val:
            self.window._resize = val 
        else:
            self.window._png_compressing = int(self.window.png_level.text())

    @Slot()
    def update_qua_speed_option(self,val='speed'):
        self.window._qua_speed_option = val
        self.window._int_weight = {'speed':True,'quality':False}[self.window._qua_speed_option]

    @Slot()
    def update_fade_out(self,val=None):
        if val: 
            self.window._fade_out = val
        else:
            self.window._fade_out = int(self.window.fade_out.text())

    @Slot()
    def update_fade_in(self,val : int = None):
        if val: 
            self.window._fade_in = val
        else:
            self.window._fade_in = int(self.window.fade_in.text())

    @Slot()
    def trigger_file_tree_item_menu(self, menu_text : str, menu_item : QTreeWidgetItem):
        '''
        文件列表的菜单选项触发逻辑
        '''
        categore = menu_item.text(0).split('（')[0]
        if categore == '星空图像':
            categore = '亮场'

        if menu_text == '展开':
            menu_item.setExpanded(True)
        elif menu_text == '折叠':
            menu_item.setExpanded(False)
        elif menu_text == '清空':
            self.clear_tree(categore_to_clear = [categore])
        elif menu_text == '添加文件':
            self.add_images(category = categore)
        elif menu_text == '添加文件夹':
            self.add_folder(category = categore)
        elif menu_text == '预览':
            self.view_file(menu_item.__file_path)
        elif menu_text == '从列表删除':
            selected_img_items = self.window.star_trail_file_tree.selectedItems()
            for selected_img_item in selected_img_items:
                self.remove_file_from_tree(selected_img_item, mode = 'SingleImg')

    # 设置文件列表的文件的删除按钮和预览按钮是否可用，点击文件名预览是否可用
    @Slot()
    def set_file_list_clickable(self, clickable : bool = True):
        if clickable:
            # 禁用全局预览是否可用以使文件名点击不再触发预览（为了不使file_label的鼠标右击被禁用，不使用setEnable
            self.window._preview_useable = True
            for category, file_tree in self.window.star_trail_file_tree_categore.items():
                for i in range(file_tree.childCount()):
                    file_item = file_tree.child(i)
                    file_item.__remove_bnt.setEnabled(True)
                    file_item.__view_bnt.setEnabled(True)
        else:
            self.window._preview_useable = False
            for category, file_tree in self.window.star_trail_file_tree_categore.items():
                for i in range(file_tree.childCount()):
                    file_item = file_tree.child(i)
                    file_item.__remove_bnt.setEnabled(False)
                    file_item.__view_bnt.setEnabled(False)

    @Slot()
    def set_widget_handleable(self, handleable : bool = True, task_type : str = None):
        handleable_widget_content = {
            '01' : {'widget':self.window.label_current_mode,        'type' : 'operable_widget'},
            '02' : {'widget':self.window.menu_setting,              'type' : 'operable_widget'},
            '03' : {'widget':self.window.menu_about,                'type' : 'operable_widget'},
            '04' : {'widget':self.window.ui_min,                    'type' : 'operable_widget'},
            '05' : {'widget':self.window.ui_max,                    'type' : 'operable_widget'},
            '06' : {'widget':self.window.ui_close,                  'type' : 'operable_widget'},
            '07' : {'widget':self.window.star_trail_file_tree,      'type' : 'tree_wieget'},
            '08' : {'widget':self.window.add_files,                 'type' : 'operable_widget'},
            '09' : {'widget':self.window.add_folder,                'type' : 'operable_widget'},
            '10' : {'widget':self.window.clear_files,               'type' : 'operable_widget'},
            '11' : {'widget':self.window.config_panel,              'type' : 'operable_widget'},
            '12' : {'widget':self.window.alter_output_type_2,       'type' : 'operable_widget'},
            '13' : {'widget':self.window.alter_output_2,            'type' : 'operable_widget'},
            '14' : {'widget':self.window.alter_png_level,           'type' : 'operable_widget'},
            '15' : {'widget':self.window.alter_jpg_level,           'type' : 'operable_widget'},
            '16' : {'widget':self.window.alter_output_bits,         'type' : 'operable_widget'},
            '17' : {'widget':self.window.btn_star_trail_preview,    'type' : 'operable_widget'},
            '18' : {'widget':self.window.btn_star_trail_start,      'type' : 'operable_widget'}
        }
        dis_handleable_widget_content = {
            'star_trail_fast_preview' : ['01','02','03','06','07','08','09','10','11','12','13','14','15','16','17','18'],
            'star_trail' : ['01','02','03','06','07','08','09','10','11','12','13','14','15','16','17','18']
        }

        if handleable:
            for _, widget in handleable_widget_content.items():
                if widget['type'] == 'operable_widget':
                    widget['widget'].setEnabled(handleable)
                elif widget['type'] == 'tree_wieget':
                    self.set_file_list_clickable(clickable=handleable)
                    self.window.star_trail_file_tree.remove_disabled_menu_items({'展开', '折叠', '清空', '添加文件', '添加文件夹','预览', '从列表删除'})
        else:
            for w_id in dis_handleable_widget_content[task_type]:
                widget = handleable_widget_content[w_id]
                if widget['type'] == 'clickable_widget':
                    widget['widget'].setClickable(handleable)
                elif widget['type'] == 'operable_widget':
                    widget['widget'].setEnabled(handleable)
                elif widget['type'] == 'tree_wieget':
                    self.set_file_list_clickable(clickable=handleable)
                    self.window.star_trail_file_tree.add_disabled_menu_items({'展开', '折叠', '清空', '添加文件', '添加文件夹','预览', '从列表删除'})

    @Slot()
    def update_status(self, status : str = 'notStart'):
        self.window._status = status

    @Slot()
    def alter_start_bnt(self, text : str = '叠加'):
        self.window.btn_star_trail_start.setText(text)

    def detect_status(self):
        if len(self.window._input_files['亮场']) == 0:
            self.window._status_n['status'] = '未就绪'
            self.window._status_n['tips'] = '请添加图像文件'
        elif len(self.window._input_files['亮场']) < 3:
            self.window._status_n['status'] = '未就绪'
            self.window._status_n['tips'] = '请添加3张或以上图像文件'
        elif self.window._output_file_path_cache[self.window._output_file_type] is None:
            self.window._status_n['status'] = '未就绪'
            self.window._status_n['tips'] = '请选择存储路径'
        else:
            self.window._status_n['status'] = '就绪'
            self.window._status_n['tips'] = '点击开始按钮进行图像处理'
        self.update_status_display()

    def update_status_display(self):
        self.window.status_text.setStyleSheet("#status_text {color:  rgba(20,20,20,220);}")
        self.window.star_trial_tips.setStyleSheet("#star_trial_tips {color:  rgba(20,20,20,220);}")
        self.window.status_text.setText(self.window._status_n['status'])
        self.window.star_trial_tips.setText(self.window._status_n['tips'])
        self.window.status_icon.setToolTip(self.window._status_n['status'])
        if self.window._status_n['status'] == '就绪':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-finish-stop.png"))
            self.window.star_trail_process_bar.setStyleSheet("#star_trail_process_bar {background-color: rgb(96, 200, 120);}")
        elif self.window._status_n['status'] == '未就绪':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-notready-.png"))
            self.window.star_trail_process_bar.setStyleSheet("#star_trail_process_bar {background-color: rgb(96, 200, 120);}")
        elif self.window._status_n['status'] == '处理中':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-working&checking.png"))
        elif self.window._status_n['status'] == '任务失败':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-finish-failed-02.png"))
        elif self.window._status_n['status'] == '任务完成':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-finish-success-01.png"))
        elif self.window._status_n['status'] == '任务取消':
            self.window.status_icon.setIcon(QIcon(u":/icons/resource/icon/status-ready.png"))
