# 用于存放函数

import re
import asyncio
from qasync import asyncSlot
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PySide6.QtCore import Slot,QSize, Qt, QPoint, QSize
from PySide6.QtWidgets import QFileDialog, QMainWindow,QDialog,QTreeWidgetItem,QPushButton,QHBoxLayout,QWidget
from PySide6.QtGui import QIcon, QCursor, QBrush, QColor

# 导入自定义组件
from ui.UILibs import ClickableLabel,exifCheckDialog,CategoryDialog
# 导入图标资源
from ui import resource

# 导入Core接口
from launcher import launch

from ezlib.progressbar import QueueProgressbar
from ezlib import scan_all_exif

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

    @Slot()
    def choose_algorithm_mean(self):
        '''
        处理均值叠加算法选择
        '''
        val = self.window.alter_algorithm_mean.currentText()
        if val == '平均值-排异':
            self.window._mode = 'sigmaclip-mean'
            # 迭代次数 隐藏
            self.window.frame_max_iter.show()
            self.window.frame_max_iter.setVisible(True)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.show()
            self.window.frame_rejection.setVisible(True)
            # 渐隐选项
            self.window.frame_fade_in_out.hide()
            self.window.frame_fade_in_out.setVisible(False)
            # 计算加速
            self.window.frame_int_weight.hide()
            self.window.frame_int_weight.setVisible(False)
            # 蒙版选项 隐藏
            self.window.frame_mask.hide()
            self.window.frame_mask.setVisible(False)
            # 启用拒绝倍率和迭代次数
            self.window.alter_max_iter.setEnabled(True)
            self.window.alter_rejection.setEnabled(True)
        else:
            self.window._mode = 'mean'
            # 迭代次数 隐藏
            self.window.frame_max_iter.hide()
            self.window.frame_max_iter.setVisible(False)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.hide()
            self.window.frame_rejection.setVisible(False)
            # 渐隐选项
            self.window.frame_fade_in_out.hide()
            self.window.frame_fade_in_out.setVisible(False)
            # 计算加速
            self.window.frame_int_weight.hide()
            self.window.frame_int_weight.setVisible(False)
            # 蒙版选项 隐藏
            self.window.frame_mask.hide()
            self.window.frame_mask.setVisible(False)
        self.detect_status()
    
    @Slot()
    def choose_algorithm_max(self):
        '''
        处理最大值叠加算法选择
        '''
        val = self.window.alter_algorithm_startrail.currentText()
        if val == '混合模式':
            self.window._mode = 'mask-mix'
            # 迭代次数 隐藏
            self.window.frame_max_iter.show()
            self.window.frame_max_iter.setVisible(True)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.show()
            self.window.frame_rejection.setVisible(True)
            # 渐隐选项
            self.window.frame_fade_in_out.show()
            self.window.frame_fade_in_out.setVisible(True)
            # 计算加速
            self.window.frame_int_weight.show()
            self.window.frame_int_weight.setVisible(True)
            # 蒙版选项 隐藏
            self.window.frame_mask.show()
            self.window.frame_mask.setVisible(True)
            # 混合模式时 如果没有添加蒙版 则迭代次数 拒绝倍率不可用
            if len(self.window._input_files['蒙版']) == 0:
                self.window.alter_max_iter.setEnabled(False)
                self.window.alter_rejection.setEnabled(False)
            else:
                self.window.alter_max_iter.setEnabled(True)
                self.window.alter_rejection.setEnabled(True)
        else:
            self.window._mode = 'max'
            # 迭代次数 隐藏
            self.window.frame_max_iter.hide()
            self.window.frame_max_iter.setVisible(False)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.hide()
            self.window.frame_rejection.setVisible(False)
            # 渐隐选项
            self.window.frame_fade_in_out.show()
            self.window.frame_fade_in_out.setVisible(True)
            # 计算加速
            self.window.frame_int_weight.show()
            self.window.frame_int_weight.setVisible(True)
            # 蒙版选项 隐藏
            self.window.frame_mask.hide()
            self.window.frame_mask.setVisible(False)
        self.detect_status()

    @Slot()
    def choose_algorithm_min(self):
        '''
        处理最小值叠加算法选择
        '''
        val = self.window.alter_algorithm_mean.currentText()
        if val == '最小值':
            self.window._mode = 'min'
            # 迭代次数 隐藏
            self.window.frame_max_iter.hide()
            self.window.frame_max_iter.setVisible(False)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.hide()
            self.window.frame_rejection.setVisible(False)
            # 渐隐选项
            self.window.frame_fade_in_out.hide()
            self.window.frame_fade_in_out.setVisible(False)
            # 计算加速
            self.window.frame_int_weight.hide()
            self.window.frame_int_weight.setVisible(False)
            # 蒙版选项 隐藏
            self.window.frame_mask.hide()
            self.window.frame_mask.setVisible(False)
        else:
            self.window._mode = 'min'
            # 迭代次数 隐藏
            self.window.frame_max_iter.hide()
            self.window.frame_max_iter.setVisible(False)
            # 拒绝倍率 隐藏
            self.window.frame_rejection.hide()
            self.window.frame_rejection.setVisible(False)
            # 渐隐选项
            self.window.frame_fade_in_out.hide()
            self.window.frame_fade_in_out.setVisible(False)
            # 计算加速
            self.window.frame_int_weight.hide()
            self.window.frame_int_weight.setVisible(False)
            # 蒙版选项 隐藏
            self.window.frame_mask.hide()
            self.window.frame_mask.setVisible(False)
        self.detect_status()

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
    def change_mode(self, mode):
        '''
        响应选择的模式
        根据选择的模式，重置算法为默认算法，设置背景图 设置默认算法下的控件显示与隐藏
        '''
        self.window.m_flag = False
        self.window.setCursor(QCursor(Qt.ArrowCursor))
        self.window.label_current_mode.setText(mode)
        if mode == '星轨叠加':
            # 设置算法选项框可见性
            self.window.frame_algorithm_mean.hide()
            self.window.frame_algorithm_mean.setVisible(False)
            self.window.frame_algorithm_min.hide()
            self.window.frame_algorithm_min.setVisible(False)
            self.window.frame_algorithm_startrail.show()
            self.window.frame_algorithm_startrail.setVisible(True)

            # 设置默认叠加算法 最大值
            self.window.alter_algorithm_startrail.setCurrentIndex(0)
            self.choose_algorithm_max()

            # # 迭代次数 隐藏
            # self.window.frame_max_iter.hide()
            # self.window.frame_max_iter.setVisible(False)
            # # 拒绝倍率 隐藏
            # self.window.frame_rejection.hide()
            # self.window.frame_rejection.setVisible(False)
            # # 渐隐选项
            # self.window.frame_fade_in_out.show()
            # self.window.frame_fade_in_out.setVisible(True)
            # # 计算加速
            # self.window.frame_int_weight.show()
            # self.window.frame_int_weight.setVisible(True)
            # # 蒙版选项 隐藏
            # self.window.frame_mask.hide()
            # self.window.frame_mask.setVisible(False)

            # 设置背景
            self.window.main_frame.setStyleSheet("""
                #main_frame {
                        border:none;
                        background-image: url(:/img/resource/img/皿仓山星轨-s.jpg);
                        background-repeat: no-repeat;                     /* 不重复背景图 */
                        background-position: center;                      /* 居中显示背景图 */
                    }
                """
            )

        elif mode == '堆栈降噪':
            # 设置算法选项框可见性
            self.window.frame_algorithm_mean.show()
            self.window.frame_algorithm_mean.setVisible(True)
            self.window.frame_algorithm_min.hide()
            self.window.frame_algorithm_min.setVisible(False)
            self.window.frame_algorithm_startrail.hide()
            self.window.frame_algorithm_startrail.setVisible(False)

            # 设置默认叠加算法 最大值
            self.window.alter_algorithm_mean.setCurrentIndex(0)
            self.choose_algorithm_mean()

            # # 迭代次数 隐藏
            # self.window.frame_max_iter.hide()
            # self.window.frame_max_iter.setVisible(False)
            # # 拒绝倍率 隐藏
            # self.window.frame_rejection.hide()
            # self.window.frame_rejection.setVisible(False)
            # # 渐隐选项 隐藏
            # self.window.frame_fade_in_out.hide()
            # self.window.frame_fade_in_out.setVisible(False)
            # # 计算加速 隐藏
            # self.window.frame_int_weight.hide()
            # self.window.frame_int_weight.setVisible(False)
            # # 蒙版选项 隐藏
            # self.window.frame_mask.hide()
            # self.window.frame_mask.setVisible(False)

            # 设置背景
            self.window.main_frame.setStyleSheet("""
                #main_frame {
                        border:none;
                        background-image: url(:/img/resource/img/back02.jpg);
                        background-repeat: no-repeat;                     /* 不重复背景图 */
                        background-position: center;                      /* 居中显示背景图 */
                    }
                """
            )
        # 更新两个标记 避免点击顶部按钮切换到子页面后 主页面无法响应鼠标release事件导致两个参数保持True，后续引发预期之外的事件
        self.window.dragging = False
        self.window.resizing = False

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
            self.window.alter_output_bits.model().item(1).setEnabled(True)
            self.window.alter_output_bits.model().item(1).setForeground(QBrush(QColor(35,35,35,210)))
            self.window.alter_output_bits.model().item(2).setEnabled(True)
            self.window.alter_output_bits.model().item(2).setForeground(QBrush(QColor(35,35,35,210)))
        else:
            pass
        
        # 将所选的文件格式存入变量，根据选择的文件格式更新输出文件路径 如果新的文件格式下，此前已经填过路径，用之前的，如果之前没填过，则为空
        # 同步更新文件路径框显示的路径和tooltip显示文字
        self.window._output_file_type = output_file_type
        self.window._output_file_path = self.window._output_file_path_cache[output_file_type]
        self.window.output_path_2.setText(self.window._output_file_path)
        self.window.output_path_2.setToolTip((self.window._output_file_path))
        self.detect_status()
    
    @Slot()
    def alter_fade_in_out(self, fade_in=None, fade_out=None):
        '''
        响应双滑块，修改fade_in fade_out参数，或初始化双滑块（不传参时）
        '''
        if fade_in is None:
            fade_in = self.window._fade_in
        if fade_out is None:
            fade_out = 100 - self.window._fade_out
        self.window.fade_in.setText(f'{fade_in}%')
        self.window.fade_out.setText(f'{100 - fade_out}%')
        self.window._fade_in = fade_in
        self.window._fade_out = 100 - fade_out
        self.detect_status()

    @Slot()
    def alter_rejection(self, rej_low=None, rej_high=None):
        '''
        响应双滑块，修改rej_low rej_high参数，或初始化双滑块（不传参时）
        '''
        if rej_low is None:
            rej_low = 0 - self.window._rej_low * 10
        if rej_high is None:
            rej_high = self.window._rej_high * 10
        self.window._rej_low = 0 - round(rej_low / 10, 1)
        self.window._rej_high = round(rej_high / 10, 1)
        self.window.rejection_low.setText('%.1f' % round(rej_low / 10, 1))
        self.window.rejection_high.setText('%.1f' % self.window._rej_high)
        self.detect_status()

    Slot()
    def mask_able(self):
        if self.window.mask_able.isChecked():
            self.window.frame_mask_sub2.show()
            self.window.frame_mask_sub2.setVisible(True)
            self.window.frame_mask.setMinimumHeight(60)
            self.window.frame_mask.setMaximumHeight(60)
            self.window._mask_able = True
        else:
            self.window.frame_mask_sub2.hide()
            self.window.frame_mask_sub2.setVisible(False)
            self.window.frame_mask.setMinimumHeight(40)
            self.window.frame_mask.setMaximumHeight(40)
            self.window._mask_able = False
        self.detect_status()

    @Slot()
    def alter_mask_file(self):
        file_dialog = QFileDialog(self,caption='添加蒙版')
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilters([
                '全部支持文件(*.cr2 *.cr3 *.arw *.nef *.dng *.tiff *.tif *.jpeg *.jpg *.png *.bmp *.gif *.fits)',
                'RAW文件(*.cr2 *.cr3 *.arw *.nef *.dng)',
                'tif文件(*.tiff *.tif)',
                'jpg文件(*.jpeg *.jpg)',
                'png文件(*.png)',
                '其它图片文件(*.bmp *.gif *.fits)'
        ])
        if file_dialog.exec_() == QDialog.Accepted:
            file_path = [url.toLocalFile() for url in file_dialog.selectedUrls()]
            self.window._input_files['蒙版'] = file_path
            self.window.mask_file_path.setText(file_path[0])
            self.window.mask_file_path.setToolTip((file_path[0]))
            # 启用拒绝倍率和迭代次数
            self.window.alter_max_iter.setEnabled(True)
            self.window.alter_rejection.setEnabled(True)
        self.detect_status()

    @Slot()
    def int_weight_able(self):
        if self.window.int_weight_able.isChecked():
            self.window._int_weight = True
        else:
            self.window._int_weight = False 
        self.detect_status()

    @Slot()
    def alter_max_iter(self, value = None):
        if value is None:
            value = self.window._max_iter
        self.window.max_iter.setText(str(value))
        self.window._max_iter = value
        self.detect_status()

    @Slot()
    def alter_output_bits(self, value = None):
        '''
        响应色深下拉框的修改
        当传入的value为空时，将_output_bits传入下拉框
        '''
        if value is None:
            self.window.alter_output_bits.setCurrentText(f'{value} bit')
        else:
            self.window._output_bits = int(value.replace(' bit',''))
        self.detect_status()

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

    @Slot()
    def update_progress_bar(self, value):
        '''
        更新进度条的槽函数
        '''
        self.window.star_trail_process_bar.setValue(value)
        if value < 100:
            self.window._status_n['tips'] = f'当前进度{value}%，请勿操作'
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
        options = QFileDialog.Options()
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

        self.window.qtbar_star_trail.reset(len(self.window._input_files['亮场']))
        # 调用叠加api
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            partial_func = partial(
                launch, 
                img_files = self.window._input_files['亮场'],
                mode = self.window._mode,
                output_fname = self.window._output_file_path,
                fin_ratio = self.window._fade_in/100,
                fout_ratio = self.window._fade_out/100,
                int_weight = self.window._int_weight,
                resize = None,
                output_bits = self.window._output_bits,
                ground_mask = self.window._input_files['蒙版'][0] if len(self.window._input_files['蒙版'])>0 and self.window._mask_able else None,
                debug_mode = False,
                rej_high = self.window._rej_high,
                rej_low = self.window._rej_low,
                max_iter = self.window._max_iter,
                png_compressing = self.window._png_compressing,
                jpg_quality = self.window._jpg_quality,
                progressbar = self.window.qtbar_star_trail
            )
            star_trail_res = await loop.run_in_executor(pool,partial_func)

            # temp = {
            #     'img_files' : self.window._input_files['亮场'],
            #     'mode' : self.window._mode,
            #     'output_fname' : self.window._output_file_path,
            #     'fin_ratio' : self.window._fade_in/100,
            #     'fout_ratio' : self.window._fade_out/100,
            #     'int_weight' : self.window._int_weight,
            #     'output_bits' : self.window._output_bits,
            #     'ground_mask' : self.window._input_files['蒙版'][0] if len(self.window._input_files['蒙版'])>1 and self.window._mask_able else None,
            #     'rej_high' : self.window._rej_high,
            #     'rej_low' : self.window._rej_low,
            #     'max_iter' : self.window._max_iter,
            #     'png_compressing' : self.window._png_compressing,
            #     'jpg_quality' : self.window._jpg_quality
            # }
            # print(temp)
            # time.sleep(10)
            # star_trail_res = {'status':True}
            
            # 解除界面控件禁用
            self.set_widget_handleable(handleable = True)
            
            if star_trail_res['status'] is True:
                # 执行成功时 修改tips为已完成，修改状态为已完成，展示图片
                # self.display_star_trail_tips('已完成！',color='green')
                self.view_file(self.window._output_file_path)
                self.window._status = 'successed'
                self.window._status_n['status'] = '任务完成'
                self.window._status_n['tips_2'] = ''
                self.window.qtbar_star_trail.finish()
            else:
                self.window.status_text.setStyleSheet("#status_text {color:rgba(200,0,0,200)}")
                self.window.star_trial_tips.setStyleSheet("#star_trial_tips {color:rgba(200,0,0,200)}")
                # 执行成功时 修改tips为失败，修改状态为失败，不展示任何图像
                # self.display_star_trail_tips('叠加失败！',color='red')
                self.view_file(file_path = '')
                self.window._status = 'failed'
                self.window._status_n['tips_2'] = ''
                self.window._status_n['status'] = '任务失败'
            
            self.update_status_display()
            

    @Slot()
    def cancel_task(self):
        if self.window._status == 'running':
            # 取消任务
            self.window._task.cancel()
            # 取消任务后 修改tips为已停止，修改状态为cancelled，不展示任何图像，恢复按钮为可点击，修改开始按钮为“开始叠加”
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

    # 更改界面的组件的可操作性，用于在执行快速预览、叠加过程中屏蔽大部分组件的可操作性
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
            '11' : {'widget':self.window.alter_algorithm_startrail, 'type' : 'operable_widget'},
            '12' : {'widget':self.window.alter_algorithm_mean,      'type' : 'operable_widget'},
            '13' : {'widget':self.window.alter_algorithm_min,       'type' : 'operable_widget'},
            '14' : {'widget':self.window.alter_mask_file,           'type' : 'operable_widget'},
            '15' : {'widget':self.window.alter_max_iter,            'type' : 'operable_widget'},
            '16' : {'widget':self.window.alter_rejection,           'type' : 'operable_widget'},
            '17' : {'widget':self.window.alter_fade_in_out,         'type' : 'operable_widget'},
            '18' : {'widget':self.window.int_weight_able,           'type' : 'operable_widget'},
            '19' : {'widget':self.window.alter_output_type_2,       'type' : 'operable_widget'},
            '20' : {'widget':self.window.alter_output_2,            'type' : 'operable_widget'},
            '21' : {'widget':self.window.alter_png_level,           'type' : 'operable_widget'},
            '22' : {'widget':self.window.alter_jpg_level,           'type' : 'operable_widget'},
            '23' : {'widget':self.window.alter_output_bits,         'type' : 'operable_widget'},
            '24' : {'widget':self.window.btn_star_trail_preview,    'type' : 'operable_widget'},
            '25' : {'widget':self.window.btn_star_trail_start,      'type' : 'operable_widget'}
        }
        dis_handleable_widget_content = {
            'star_trail_fast_preview' : ['01','02','03','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'],
            # 'star_trail_fast_preview' : [],
            'star_trail' : ['01','02','03','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
            # 'star_trail' : []
        }

        if handleable:
            for _, widget in handleable_widget_content.items():
                if widget['type'] == 'operable_widget':
                    widget['widget'].setEnabled(handleable)
                elif widget['type'] == 'tree_wieget':
                    # 禁用文件的点击操作
                    self.set_file_list_clickable(clickable=handleable)
                    # 设置菜单栏的不可点击项目
                    self.window.star_trail_file_tree.remove_disabled_menu_items({'展开', '折叠', '清空', '添加文件', '添加文件夹','预览', '从列表删除'})
        else:
            for w_id in dis_handleable_widget_content[task_type]:
                widget = handleable_widget_content[w_id]
                if widget['type'] == 'clickable_widget':
                    widget['widget'].setClickable(handleable)
                elif widget['type'] == 'operable_widget':
                    widget['widget'].setEnabled(handleable)
                elif widget['type'] == 'tree_wieget':
                    # 禁用文件的点击操作
                    self.set_file_list_clickable(clickable=handleable)
                    # 设置菜单栏的不可点击项目
                    self.window.star_trail_file_tree.add_disabled_menu_items({'展开', '折叠', '清空', '添加文件', '添加文件夹','预览', '从列表删除'})

    @Slot()
    def update_status(self, status : str = 'notStart'):
        self.window._status = status

    @Slot()
    def alter_start_bnt(self, text : str = '叠加'):
        self.window.btn_star_trail_start.setText(text)

    def detect_status(self):
        if self.window._mode == 'max':
            if len(self.window._input_files['亮场']) == 0:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加图像文件'
            elif len(self.window._input_files['亮场']) < 5:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加5张或以上图像文件'
            elif self.window._output_file_path_cache[self.window._output_file_type] is None:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请选择存储路径'
            else:
                self.window._status_n['status'] = '就绪'
                self.window._status_n['tips'] = '点击开始按钮进行图像处理'
        elif self.window._mode == 'mask-mix':
            if len(self.window._input_files['亮场']) == 0:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加图像文件'
            elif len(self.window._input_files['亮场']) < 5:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加5张或以上图像文件'
            elif len(self.window._input_files['蒙版']) == 0:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请选择蒙版文件'
            elif self.window._output_file_path_cache[self.window._output_file_type] is None:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请选择存储路径'
            else:
                self.window._status_n['status'] = '就绪'
                self.window._status_n['tips'] = '点击开始按钮进行图像处理'
        elif self.window._mode == 'mean':
            if len(self.window._input_files['亮场']) == 0:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加图像文件'
            elif len(self.window._input_files['亮场']) < 5:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加5张或以上图像文件'
            elif self.window._output_file_path_cache[self.window._output_file_type] is None:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请选择存储路径'
            else:
                self.window._status_n['status'] = '就绪'
                self.window._status_n['tips'] = '点击开始按钮进行图像处理'
        elif self.window._mode == 'sigmaclip-mean':
            if len(self.window._input_files['亮场']) == 0:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加图像文件'
            elif len(self.window._input_files['亮场']) < 5:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请添加5张或以上图像文件'
            elif self.window._output_file_path_cache[self.window._output_file_type] is None:
                self.window._status_n['status'] = '未就绪'
                self.window._status_n['tips'] = '请选择存储路径'
            else:
                self.window._status_n['status'] = '就绪'
                self.window._status_n['tips'] = '点击开始按钮进行图像处理'
        # print(self.window._status_n)
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
