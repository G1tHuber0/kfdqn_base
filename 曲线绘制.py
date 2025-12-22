#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RL Episode Returns Plotter App
------------------------------
一个独立的 GUI 工具，用于可视化强化学习训练过程中的 raw_episode_returns.csv。
支持拖拽导入、多文件对比、TensorBoard 风格平滑、自定义透明度及一键保存。

依赖: PyQt5, matplotlib, pandas, numpy
运行: python kfdqn_base/episode_returns_plot_app.py
"""

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

# ==========================================
# 依赖检测
# ==========================================
def check_dependencies():
    missing = []
    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    if missing:
        print("="*60)
        print("错误：缺少必要的依赖库，无法启动 App。")
        print(f"缺少的库: {', '.join(missing)}")
        print("\n请运行以下命令进行安装：")
        print(f"pip install {' '.join(missing)}")
        print("="*60)
        sys.exit(1)

check_dependencies()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QCheckBox, QSlider, 
                             QComboBox, QFileDialog, QFrame, QSplitter, QListWidget, 
                             QListWidgetItem, QMessageBox, QGroupBox, QStackedWidget,
                             QFormLayout)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ==========================================
# 核心算法：平滑逻辑
# ==========================================
class SmoothingUtils:
    @staticmethod
    def moving_average(data, window_size):
        """标准滑动平均"""
        if window_size <= 1:
            return data
        series = pd.Series(data)
        # min_periods=1 保证开头也有数据
        return series.rolling(window=int(window_size), min_periods=1).mean().values

    @staticmethod
    def tensorboard_smooth(data, weight=0.6):
        """
        TensorBoard 风格的 EMA (Exponential Moving Average) with Debias
        Formula:
            shadow = shadow * w + (1 - w) * x
            debias = 1 - w^step
            y = shadow / debias
        """
        if weight <= 0:
            return data
        if weight >= 1.0:
            weight = 0.99  # 保护

        scalar = np.array(data)
        last = 0  # TensorBoard starts with 0 accumulator
        smoothed = []
        debias_weight = 1.0

        for i, point in enumerate(scalar):
            last = last * weight + (1 - weight) * point
            debias_weight = debias_weight * weight
            
            # 避免除以0 (理论上 debias_weight 随 step 增大而减小，1-debias 增大)
            debias = 1.0 - debias_weight
            if debias == 0:
                smoothed_val = point
            else:
                smoothed_val = last / debias
            
            smoothed.append(smoothed_val)
        
        return np.array(smoothed)

# ==========================================
# UI 组件：支持拖拽的文件列表
# ==========================================
class FileListWidget(QListWidget):
    def __init__(self, parent_callback):
        super().__init__()
        self.parent_callback = parent_callback
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setStyleSheet("""
            QListWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 5px;
            }
            QListWidget:item {
                padding: 5px;
            }
        """)
        
        # 提示标签
        self.placeholder = QLabel("拖入 raw_episode_returns.csv\n或点击“添加文件”", self)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; background: transparent;")
        self.placeholder.setAttribute(Qt.WA_TransparentForMouseEvents)

    def resizeEvent(self, event):
        self.placeholder.resize(self.size())
        super().resizeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and path.lower().endswith('.csv'):
                    file_paths.append(path)
            
            if file_paths:
                self.parent_callback(file_paths)
                self.placeholder.hide()
            event.accept()
        else:
            super().dropEvent(event)

    def add_item_custom(self, path):
        # 避免重复添加
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == path:
                return
        
        filename = os.path.basename(path)
        parent_dir = os.path.basename(os.path.dirname(path))
        display_text = f"{parent_dir}/{filename}"
        
        item = QListWidgetItem(display_text)
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.addItem(item)
        self.placeholder.hide()

# ==========================================
# 主窗口 App
# ==========================================
class RLPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Training Plotter (PyQt5)")
        self.resize(1280, 800)
        
        # 数据存储
        self.loaded_data = {}  # path -> {'df': dataframe, 'x': col, 'y': col}
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === 左侧控制面板 (Splitter Left) ===
        control_panel = QFrame()
        control_panel.setFixedWidth(320)
        control_panel.setFrameShape(QFrame.StyledPanel)
        ctrl_layout = QVBoxLayout(control_panel)
        ctrl_layout.setContentsMargins(10, 10, 10, 10)

        # 1. 顶部功能区
        top_group = QGroupBox("窗口设置")
        top_layout = QVBoxLayout()
        self.check_top = QCheckBox("窗口置顶 (Always on Top)")
        self.check_top.stateChanged.connect(self.toggle_always_on_top)
        top_layout.addWidget(self.check_top)
        top_group.setLayout(top_layout)
        ctrl_layout.addWidget(top_group)

        # 2. 文件管理区
        file_group = QGroupBox("数据文件 (拖拽导入)")
        file_layout = QVBoxLayout()
        
        self.file_list = FileListWidget(self.load_files)
        file_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("添加文件")
        self.btn_add.clicked.connect(self.open_file_dialog)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_files)
        self.btn_remove = QPushButton("移除选中")
        self.btn_remove.clicked.connect(self.remove_selected_files)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)
        
        file_group.setLayout(file_layout)
        ctrl_layout.addWidget(file_group, stretch=1) # 让文件列表占据主要空间

        # 3. 绘图设置区
        plot_group = QGroupBox("绘图选项")
        plot_layout = QFormLayout()

        # Overlay
        self.check_overlay = QCheckBox("合并绘制 (Overlay)")
        self.check_overlay.setChecked(True)
        self.check_overlay.toggled.connect(self.update_plot)
        plot_layout.addRow("显示模式:", self.check_overlay)

        # Smooth Type
        self.combo_smooth = QComboBox()
        self.combo_smooth.addItems(["None", "Moving Average", "TensorBoard Smooth"])
        self.combo_smooth.currentIndexChanged.connect(self.on_smooth_type_changed)
        plot_layout.addRow("平滑算法:", self.combo_smooth)
        
        plot_group.setLayout(plot_layout)
        ctrl_layout.addWidget(plot_group)

        # 4. 平滑参数区 (动态显示)
        self.smooth_param_stack = QStackedWidget()
        
        # Page 0: Empty (None)
        self.smooth_param_stack.addWidget(QWidget())
        
        # Page 1: Moving Average
        page_ma = QWidget()
        layout_ma = QFormLayout()
        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setRange(1, 200)
        self.slider_window.setValue(10)
        self.label_window = QLabel("10")
        self.slider_window.valueChanged.connect(lambda v: (self.label_window.setText(str(v)), self.update_plot()))
        layout_ma.addRow("窗口大小:", self.label_window)
        layout_ma.addRow(self.slider_window)
        page_ma.setLayout(layout_ma)
        self.smooth_param_stack.addWidget(page_ma)

        # Page 2: TensorBoard
        page_tb = QWidget()
        layout_tb = QFormLayout()
        self.slider_weight = QSlider(Qt.Horizontal)
        self.slider_weight.setRange(0, 99) # 0.00 - 0.99
        self.slider_weight.setValue(60) # 0.6
        self.label_weight = QLabel("0.60")
        self.slider_weight.valueChanged.connect(lambda v: (self.label_weight.setText(f"{v/100:.2f}"), self.update_plot()))
        layout_tb.addRow("平滑系数 (Weight):", self.label_weight)
        layout_tb.addRow(self.slider_weight)
        page_tb.setLayout(layout_tb)
        self.smooth_param_stack.addWidget(page_tb)

        ctrl_layout.addWidget(self.smooth_param_stack)

        # 5. 透明度控制
        self.group_alpha = QGroupBox("原始数据可见度")
        alpha_layout = QFormLayout()
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(30) # 0.3
        self.label_alpha = QLabel("0.30")
        self.slider_alpha.valueChanged.connect(lambda v: (self.label_alpha.setText(f"{v/100:.2f}"), self.update_plot()))
        alpha_layout.addRow("透明度:", self.label_alpha)
        alpha_layout.addRow(self.slider_alpha)
        self.group_alpha.setLayout(alpha_layout)
        self.group_alpha.setEnabled(False) # 初始禁用，仅平滑开启时启用
        ctrl_layout.addWidget(self.group_alpha)

        # 6. 保存按钮
        self.btn_save = QPushButton("保存图片 (Save)")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)

        # === 右侧绘图区 (Splitter Right) ===
        self.canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        # 主 Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.canvas_panel)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

    # ==========================================
    # 逻辑功能
    # ==========================================
    
    def toggle_always_on_top(self, state):
        if state == Qt.Checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.show()
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.show()

    def open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择 CSV 文件", "", "CSV Files (*.csv)")
        if files:
            self.load_files(files)

    def load_files(self, file_paths):
        for path in file_paths:
            if path in self.loaded_data:
                continue
            
            try:
                df = pd.read_csv(path)
                
                # 识别列名
                cols = [c.lower() for c in df.columns]
                y_col = None
                x_col = None
                
                # 找 Return/Reward
                for target in ['return', 'reward', 'score', 'value']:
                    for c in df.columns:
                        if target in c.lower():
                            y_col = c
                            break
                    if y_col: break
                
                # 如果没找到 return 列，尝试取最后一列
                if not y_col and len(df.columns) > 0:
                    y_col = df.columns[-1]

                # 找 Episode/Step
                for target in ['episode', 'step', 'epoch', 'iter']:
                    for c in df.columns:
                        if target in c.lower():
                            x_col = c
                            break
                    if x_col: break
                
                # 存储数据
                self.loaded_data[path] = {
                    'df': df,
                    'y': y_col,
                    'x': x_col,
                    'filename': os.path.basename(path),
                    'parent': os.path.basename(os.path.dirname(path))
                }
                
                self.file_list.add_item_custom(path)
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                QMessageBox.warning(self, "加载失败", f"无法读取文件:\n{path}\n错误: {e}")

        self.update_plot()

    def remove_selected_files(self):
        for item in self.file_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path in self.loaded_data:
                del self.loaded_data[path]
            self.file_list.takeItem(self.file_list.row(item))
        
        if self.file_list.count() == 0:
            self.file_list.placeholder.show()
            
        self.update_plot()

    def clear_files(self):
        self.file_list.clear()
        self.loaded_data.clear()
        self.file_list.placeholder.show()
        self.update_plot()

    def on_smooth_type_changed(self, index):
        self.smooth_param_stack.setCurrentIndex(index)
        self.group_alpha.setEnabled(index != 0)
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        
        # 获取当前列表中的路径顺序，保证颜色一致性
        current_paths = []
        for i in range(self.file_list.count()):
            current_paths.append(self.file_list.item(i).data(Qt.UserRole))
        
        if not current_paths:
            self.canvas.draw()
            return

        # 参数获取
        overlay = self.check_overlay.isChecked()
        smooth_type = self.combo_smooth.currentIndex() # 0: None, 1: MA, 2: TB
        raw_alpha = self.slider_alpha.value() / 100.0
        
        # 准备 Axes
        if overlay:
            axes = [self.figure.add_subplot(111)]
            axes[0].set_title("Episode Returns (Overlay)")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Return")
        else:
            axes = self.figure.subplots(len(current_paths), 1, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            # 调整子图间距
            self.figure.subplots_adjust(hspace=0.4)

        for i, path in enumerate(current_paths):
            data_info = self.loaded_data.get(path)
            if not data_info: continue
            
            ax = axes[0] if overlay else axes[i]
            color = self.colors[i % len(self.colors)]
            
            df = data_info['df']
            y_col = data_info['y']
            x_col = data_info['x']
            
            y_data = df[y_col].values
            if x_col:
                x_data = df[x_col].values
            else:
                x_data = np.arange(1, len(y_data) + 1)

            label_base = f"{data_info['parent']}/{data_info['filename']}" if overlay else "Return"

            # 1. 绘制 Raw Data
            if smooth_type == 0:
                # 不平滑：只画原始线，alpha=1
                ax.plot(x_data, y_data, color=color, linewidth=1.5, alpha=1.0, label=label_base)
            else:
                # 开启平滑：先画透明的 Raw
                if raw_alpha > 0:
                    ax.plot(x_data, y_data, color=color, linewidth=1, alpha=raw_alpha, label=f"{label_base} (raw)" if not overlay else None)
                
                # 计算平滑数据
                smooth_y = y_data
                if smooth_type == 1: # Moving Average
                    win = self.slider_window.value()
                    smooth_y = SmoothingUtils.moving_average(y_data, win)
                elif smooth_type == 2: # TensorBoard
                    w = self.slider_weight.value() / 100.0
                    smooth_y = SmoothingUtils.tensorboard_smooth(y_data, w)
                
                # 绘制平滑线
                smooth_label = label_base if overlay else "Smoothed"
                ax.plot(x_data, smooth_y, color=color, linewidth=2, alpha=1.0, label=smooth_label)

            if not overlay:
                ax.set_title(label_base, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                # 仅最后一个子图显示 x 轴标签
                if i == len(current_paths) - 1:
                    ax.set_xlabel(x_col if x_col else "Episode")
            else:
                ax.grid(True, linestyle='--', alpha=0.5)

        if overlay and current_paths:
            axes[0].legend(fontsize='small')

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "无数据", "请先加载数据文件。")
            return

        # 确定保存目录：优先使用第一个文件的目录
        first_path = self.file_list.item(0).data(Qt.UserRole)
        save_dir = os.path.dirname(first_path)
        
        # 生成默认文件名
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        smooth_str = ["None", "MA", "TB"][self.combo_smooth.currentIndex()]
        default_name = f"plot_{smooth_str}_{ts}.png"
        default_path = os.path.join(save_dir, default_name)

        # 弹出保存对话框（用户可修改）
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_path, 
            "PNG Images (*.png);;PDF Documents (*.pdf);;SVG Images (*.svg)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图片已保存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体大小，避免在高分屏上太小
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = RLPlotterApp()
    window.show()
    sys.exit(app.exec_())