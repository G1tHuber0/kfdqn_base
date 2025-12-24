#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RL Episode Returns Plotter App (v2.4)
-------------------------------------------
更新说明：
1. 图注命名逻辑优化：只保留 'KFDQN_seed69' 字样。
2. 图注位置优化：置于图表内部上方，横向排布。
3. 新增图注控制：可调节字号大小、背景透明度。
4. 坐标轴缩放时图注保持相对位置不变。
"""

import sys
import os
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
        print("错误：缺少必要的依赖库。")
        print(f"请运行: pip install {' '.join(missing)}")
        print("="*60)
        sys.exit(1)

check_dependencies()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QPushButton, QCheckBox, QSlider, 
                             QComboBox, QFileDialog, QFrame, QSplitter, QListWidget, 
                             QListWidgetItem, QMessageBox, QGroupBox, QStackedWidget,
                             QFormLayout, QDoubleSpinBox, QGridLayout, QAbstractItemView,
                             QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont

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
        if window_size <= 1:
            return data
        series = pd.Series(data)
        return series.rolling(window=int(window_size), min_periods=1).mean().values

    @staticmethod
    def tensorboard_smooth(data, weight=0.6):
        if weight <= 0: return data
        if weight >= 1.0: weight = 0.99
        scalar = np.array(data)
        last = 0
        smoothed = []
        debias_weight = 1.0
        for point in scalar:
            last = last * weight + (1 - weight) * point
            debias_weight = debias_weight * weight
            debias = 1.0 - debias_weight
            smoothed_val = point if debias == 0 else last / debias
            smoothed.append(smoothed_val)
        return np.array(smoothed)

# ==========================================
# UI 组件：文件列表
# ==========================================
class FileListWidget(QListWidget):
    def __init__(self, parent_callback, item_changed_callback):
        super().__init__()
        self.parent_callback = parent_callback
        self.itemChanged.connect(item_changed_callback)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setStyleSheet("""
            QListWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        self.placeholder = QLabel("拖入 CSV 文件\n(自动勾选并绘制)", self)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; background: transparent;")
        self.placeholder.setAttribute(Qt.WA_TransparentForMouseEvents)

    def resizeEvent(self, event):
        self.placeholder.resize(self.size())
        super().resizeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

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
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == path:
                return
        
        # 列表显示逻辑：仍然显示完整父目录以便区分，但Tooltips显示完整路径
        parent_dir = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        display_text = f"{parent_dir}"
        
        item = QListWidgetItem(display_text)
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked) 
        
        self.addItem(item)
        self.placeholder.hide()

# ==========================================
# 主窗口 App
# ==========================================
class RLPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Training Plotter v2.4")
        self.resize(1380, 850)
        
        self.loaded_data = {}
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === 左侧控制面板 ===
        control_panel = QFrame()
        control_panel.setFixedWidth(340)
        control_panel.setFrameShape(QFrame.StyledPanel)
        ctrl_layout = QVBoxLayout(control_panel)
        ctrl_layout.setContentsMargins(10, 10, 10, 10)

        # 1. 顶部功能
        top_group = QGroupBox("窗口设置")
        top_h = QHBoxLayout()
        self.check_top = QCheckBox("置顶")
        self.check_top.stateChanged.connect(self.toggle_always_on_top)
        top_h.addWidget(self.check_top)
        top_group.setLayout(top_h)
        ctrl_layout.addWidget(top_group)

        # 2. 文件列表
        file_group = QGroupBox("数据文件")
        file_layout = QVBoxLayout()
        self.file_list = FileListWidget(self.load_files, lambda item: self.update_plot())
        file_layout.addWidget(self.file_list)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self.open_file_dialog)
        self.btn_remove = QPushButton("移除")
        self.btn_remove.clicked.connect(self.remove_selected_files)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_files)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)
        
        file_group.setLayout(file_layout)
        ctrl_layout.addWidget(file_group, stretch=1)

        # 3. 绘图模式与平滑
        plot_group = QGroupBox("绘图模式与平滑")
        plot_layout = QFormLayout()
        
        self.check_overlay = QCheckBox("合并绘制 (Overlay)")
        self.check_overlay.setChecked(True)
        self.check_overlay.toggled.connect(self.update_plot)
        plot_layout.addRow("模式:", self.check_overlay)

        self.combo_smooth = QComboBox()
        self.combo_smooth.addItems(["None", "Moving Average", "TensorBoard Smooth"])
        self.combo_smooth.currentIndexChanged.connect(self.on_smooth_type_changed)
        plot_layout.addRow("平滑:", self.combo_smooth)
        plot_group.setLayout(plot_layout)
        ctrl_layout.addWidget(plot_group)

        # 3.1 动态平滑参数
        self.smooth_param_stack = QStackedWidget()
        self.smooth_param_stack.addWidget(QWidget()) # Empty
        
        # MA
        page_ma = QWidget()
        l_ma = QFormLayout()
        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setRange(1, 200)
        self.slider_window.setValue(10)
        self.label_window = QLabel("10")
        self.slider_window.valueChanged.connect(lambda v: (self.label_window.setText(str(v)), self.update_plot()))
        l_ma.addRow("Window:", self.label_window)
        l_ma.addRow(self.slider_window)
        page_ma.setLayout(l_ma)
        self.smooth_param_stack.addWidget(page_ma)

        # TB
        page_tb = QWidget()
        l_tb = QFormLayout()
        self.slider_weight = QSlider(Qt.Horizontal)
        self.slider_weight.setRange(0, 99)
        self.slider_weight.setValue(60)
        self.label_weight = QLabel("0.60")
        self.slider_weight.valueChanged.connect(lambda v: (self.label_weight.setText(f"{v/100:.2f}"), self.update_plot()))
        l_tb.addRow("Weight:", self.label_weight)
        l_tb.addRow(self.slider_weight)
        page_tb.setLayout(l_tb)
        self.smooth_param_stack.addWidget(page_tb)
        
        ctrl_layout.addWidget(self.smooth_param_stack)

        # 4. 图注(Legend)设置 [新增]
        legend_group = QGroupBox("图注设置 (Legend)")
        legend_layout = QGridLayout()
        
        legend_layout.addWidget(QLabel("字号:"), 0, 0)
        self.spin_legend_size = QSpinBox()
        self.spin_legend_size.setRange(5, 30)
        self.spin_legend_size.setValue(10)
        self.spin_legend_size.valueChanged.connect(self.update_plot)
        legend_layout.addWidget(self.spin_legend_size, 0, 1)

        legend_layout.addWidget(QLabel("透明度:"), 1, 0)
        self.slider_legend_alpha = QSlider(Qt.Horizontal)
        self.slider_legend_alpha.setRange(0, 100)
        self.slider_legend_alpha.setValue(80) # 默认0.8
        self.slider_legend_alpha.valueChanged.connect(self.update_plot)
        legend_layout.addWidget(self.slider_legend_alpha, 1, 1)
        
        legend_group.setLayout(legend_layout)
        ctrl_layout.addWidget(legend_group)

        # 5. 原始数据透明度
        self.group_alpha = QGroupBox("原始数据透明度")
        alpha_layout = QVBoxLayout()
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(30)
        self.slider_alpha.valueChanged.connect(self.update_plot)
        alpha_layout.addWidget(self.slider_alpha)
        self.group_alpha.setLayout(alpha_layout)
        self.group_alpha.setEnabled(False)
        ctrl_layout.addWidget(self.group_alpha)

        # 6. 坐标轴范围
        axis_group = QGroupBox("坐标轴范围 (Auto / Manual)")
        axis_layout = QGridLayout()
        
        self.chk_auto_x = QCheckBox("X Auto")
        self.chk_auto_x.setChecked(True)
        self.chk_auto_x.toggled.connect(self.update_axis_controls)
        axis_layout.addWidget(self.chk_auto_x, 0, 0, 1, 2)
        
        self.spin_x_min = QDoubleSpinBox()
        self.spin_x_min.setRange(-1e9, 1e9)
        self.spin_x_min.setDecimals(0)
        self.spin_x_min.setPrefix("Min: ")
        self.spin_x_min.editingFinished.connect(self.update_plot)
        
        self.spin_x_max = QDoubleSpinBox()
        self.spin_x_max.setRange(-1e9, 1e9)
        self.spin_x_max.setDecimals(0)
        self.spin_x_max.setPrefix("Max: ")
        self.spin_x_max.editingFinished.connect(self.update_plot)

        axis_layout.addWidget(self.spin_x_min, 1, 0)
        axis_layout.addWidget(self.spin_x_max, 1, 1)

        self.chk_auto_y = QCheckBox("Y Auto")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.toggled.connect(self.update_axis_controls)
        axis_layout.addWidget(self.chk_auto_y, 2, 0, 1, 2)
        
        self.spin_y_min = QDoubleSpinBox()
        self.spin_y_min.setRange(-1e9, 1e9)
        self.spin_y_min.setDecimals(2)
        self.spin_y_min.setPrefix("Min: ")
        self.spin_y_min.editingFinished.connect(self.update_plot)
        
        self.spin_y_max = QDoubleSpinBox()
        self.spin_y_max.setRange(-1e9, 1e9)
        self.spin_y_max.setDecimals(2)
        self.spin_y_max.setPrefix("Max: ")
        self.spin_y_max.editingFinished.connect(self.update_plot)

        axis_layout.addWidget(self.spin_y_min, 3, 0)
        axis_layout.addWidget(self.spin_y_max, 3, 1)
        
        axis_group.setLayout(axis_layout)
        ctrl_layout.addWidget(axis_group)

        # 7. 保存
        self.btn_save = QPushButton("保存图片")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)

        # === 右侧绘图区 ===
        self.canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_panel)
        
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(self.canvas_panel)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.update_axis_controls()

    # ==========================================
    # 逻辑功能
    # ==========================================
    def toggle_always_on_top(self, state):
        flags = self.windowFlags()
        if state == Qt.Checked:
            self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
        self.show()

    def update_axis_controls(self):
        x_auto = self.chk_auto_x.isChecked()
        self.spin_x_min.setEnabled(not x_auto)
        self.spin_x_max.setEnabled(not x_auto)
        
        y_auto = self.chk_auto_y.isChecked()
        self.spin_y_min.setEnabled(not y_auto)
        self.spin_y_max.setEnabled(not y_auto)
        
        self.update_plot()

    def open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择 CSV", "", "CSV Files (*.csv)")
        if files: self.load_files(files)

    def load_files(self, file_paths):
        for path in file_paths:
            if path in self.loaded_data: continue
            try:
                df = pd.read_csv(path)
                y_col = next((c for c in df.columns if any(k in c.lower() for k in ['return','reward','score'])), None)
                if not y_col and not df.empty: y_col = df.columns[-1]
                x_col = next((c for c in df.columns if any(k in c.lower() for k in ['episode','step','epoch'])), None)
                
                if y_col:
                    self.loaded_data[path] = {
                        'df': df, 'y': y_col, 'x': x_col,
                        'filename': os.path.basename(path),
                        'parent': os.path.basename(os.path.dirname(path))
                    }
                    self.file_list.add_item_custom(path)
            except Exception as e:
                print(f"Error: {e}")
        self.update_plot()

    def remove_selected_files(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items: return
        for item in selected_items:
            path = item.data(Qt.UserRole)
            if path in self.loaded_data: del self.loaded_data[path]
            self.file_list.takeItem(self.file_list.row(item))
        if self.file_list.count() == 0: self.file_list.placeholder.show()
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

    def extract_short_name(self, dir_name):
        """
        提取规则：提取直到包含 'seed' 的部分。
        输入示例: KFDQN_seed69_20251221_203132
        输出示例: KFDQN_seed69
        """
        parts = dir_name.split('_')
        res = []
        found_seed = False
        for p in parts:
            res.append(p)
            if 'seed' in p.lower():
                found_seed = True
                break
        
        if found_seed:
            return "_".join(res)
        else:
            # 如果没找到seed，尝试返回前两个字段作为备选
            return "_".join(parts[:2]) if len(parts) >= 2 else dir_name

    def update_plot(self):
        if not hasattr(self, 'figure'): return

        self.figure.clear()
        
        selected_paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.Checked:
                path = item.data(Qt.UserRole)
                selected_paths.append(path)
        
        if not selected_paths:
            self.canvas.draw()
            return

        # 布局调整：现在图注在内部，所以不需要给右侧留太多空间
        # 留一点边距美观即可
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)

        overlay = self.check_overlay.isChecked()
        smooth_idx = self.combo_smooth.currentIndex()
        raw_alpha = self.slider_alpha.value() / 100.0
        
        # Legend 参数
        legend_size = self.spin_legend_size.value()
        legend_alpha = self.slider_legend_alpha.value() / 100.0

        if overlay:
            axes = [self.figure.add_subplot(111)]
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Return")
        else:
            axes = self.figure.subplots(len(selected_paths), 1, sharex=True)
            if not isinstance(axes, np.ndarray): axes = [axes]
            self.figure.subplots_adjust(hspace=0.4)

        for i, path in enumerate(selected_paths):
            data = self.loaded_data[path]
            ax = axes[0] if overlay else axes[i]
            color = self.colors[i % len(self.colors)]
            
            y_raw = data['df'][data['y']].values
            x_raw = data['df'][data['x']].values if data['x'] else np.arange(1, len(y_raw)+1)
            
            # 使用提取的短名称
            short_name = self.extract_short_name(data['parent'])
            label_name = short_name if overlay else "Return"

            # 绘制 Raw
            if smooth_idx == 0:
                ax.plot(x_raw, y_raw, color=color, alpha=1.0, linewidth=1.5, label=label_name)
            else:
                if raw_alpha > 0:
                    ax.plot(x_raw, y_raw, color=color, alpha=raw_alpha, linewidth=1, label=f"{label_name} (raw)" if not overlay else None)
                
                # 平滑
                if smooth_idx == 1:
                    y_smooth = SmoothingUtils.moving_average(y_raw, self.slider_window.value())
                else:
                    y_smooth = SmoothingUtils.tensorboard_smooth(y_raw, self.slider_weight.value()/100.0)
                
                smooth_label = label_name if overlay else "Smoothed"
                ax.plot(x_raw, y_smooth, color=color, alpha=1.0, linewidth=2, label=smooth_label)

            # 非Overlay模式标题设置
            if not overlay:
                ax.set_title(f"Return ({short_name})", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)

        # 坐标轴范围
        target_axes = axes if not overlay else [axes[0]]
        for ax in target_axes:
            if not self.chk_auto_x.isChecked():
                xmin = self.spin_x_min.value()
                xmax = self.spin_x_max.value()
                if xmin < xmax: ax.set_xlim(xmin, xmax)
            
            if not self.chk_auto_y.isChecked():
                ymin = self.spin_y_min.value()
                ymax = self.spin_y_max.value()
                if ymin < ymax: ax.set_ylim(ymin, ymax)

        # === 核心修改：图注设置 (Legend) ===
        if overlay:
            axes[0].grid(True, linestyle='--', alpha=0.5)
            # 1. loc='upper left' + bbox_to_anchor=(0, 1) + borderaxespad=0
            #    确保紧贴左上角 (或者 upper center 视需求，这里用 upper left 横向排布)
            # 2. ncol=len(selected_paths) 实现横向排布
            
            axes[0].legend(
                loc='upper left', 
                bbox_to_anchor=(0, 1.0), # 锚点在坐标系左上角，(0,1)代表左上顶点
                borderaxespad=0.2,       # 略微留一点缝隙美观，设为0则完全紧贴
                ncol=len(selected_paths),# 横向排布
                fontsize=legend_size,
                framealpha=legend_alpha,
                fancybox=True
            )

        self.canvas.draw()

    def save_plot(self):
        if not self.file_list.count(): return
        for i in range(self.file_list.count()):
            if self.file_list.item(i).checkState() == Qt.Checked:
                base_dir = os.path.dirname(self.file_list.item(i).data(Qt.UserRole))
                break
        else:
            base_dir = os.path.expanduser("~")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"plot_result_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存", os.path.join(base_dir, name), "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "完成", f"已保存至: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    window = RLPlotterApp()
    window.show()
    sys.exit(app.exec_())