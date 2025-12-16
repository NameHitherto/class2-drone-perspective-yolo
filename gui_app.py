import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("无人机视角目标检测系统 (YOLO11)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.model = None
        self.current_image_path = None
        self.image_list = []
        self.current_index = 0
        
        # 加载模型
        self.load_model()
        
        # 初始化UI
        self.init_ui()
        
    def load_model(self):
        # 尝试加载训练好的最佳模型
        # 优先加载本次训练的 best.pt
        model_path = 'runs/train/visdrone_yolo11_02/weights/best.pt'
        
        if not os.path.exists(model_path):
            print(f"未找到训练模型: {model_path}")
            # 如果找不到，尝试加载预训练模型作为备选
            model_path = 'weight/yolo11l.pt' 
            if not os.path.exists(model_path):
                 # 最后尝试直接下载
                 model_path = 'yolo11l.pt'
        
        try:
            print(f"正在加载模型: {model_path} ...")
            self.model = YOLO(model_path)
            print(f"模型加载成功!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}\n请确保已运行训练脚本或有网络连接下载模型。")

    def init_ui(self):
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 (垂直)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. 标题栏
        title_label = QLabel("无人机视角目标检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 2. 按钮控制区 (水平)
        btn_layout = QHBoxLayout()
        
        self.btn_open_img = QPushButton("📂 打开图片")
        self.btn_open_img.clicked.connect(self.open_image)
        self.btn_open_img.setMinimumHeight(40)
        
        self.btn_open_folder = QPushButton("📁 打开文件夹")
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_folder.setMinimumHeight(40)
        
        self.btn_prev = QPushButton("⬅ 上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        self.btn_prev.setMinimumHeight(40)
        
        self.btn_next = QPushButton("下一张 ➡")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        self.btn_next.setMinimumHeight(40)
        
        self.btn_detect = QPushButton("🚀 开始检测")
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_detect.setMinimumHeight(40)
        self.btn_detect.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        btn_layout.addWidget(self.btn_open_img)
        btn_layout.addWidget(self.btn_open_folder)
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_detect)
        
        main_layout.addLayout(btn_layout)
        
        # 3. 图片显示区 (水平)
        img_layout = QHBoxLayout()
        
        # 左侧原图
        self.lbl_origin = QLabel("请上传图片")
        self.lbl_origin.setAlignment(Qt.AlignCenter)
        self.lbl_origin.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; font-size: 16px; color: #666;")
        self.lbl_origin.setMinimumSize(400, 400)
        
        # 右侧结果图
        self.lbl_result = QLabel("检测结果将显示在这里")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("border: 2px solid #4CAF50; background-color: #f0f0f0; font-size: 16px; color: #666;")
        self.lbl_result.setMinimumSize(400, 400)
        
        img_layout.addWidget(self.lbl_origin, 1) # 1是拉伸因子
        img_layout.addWidget(self.lbl_result, 1)
        
        main_layout.addLayout(img_layout)
        
        # 4. 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #333; padding: 5px;")
        main_layout.addWidget(self.status_label)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_list = [file_path]
            self.current_index = 0
            self.load_current_image()
            self.update_nav_buttons()

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            # 获取文件夹内所有图片
            exts = ('.png', '.jpg', '.jpeg', '.bmp')
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(exts)]
            
            if self.image_list:
                self.current_index = 0
                self.load_current_image()
                self.status_label.setText(f"已加载文件夹，共 {len(self.image_list)} 张图片")
            else:
                QMessageBox.warning(self, "提示", "文件夹内未找到图片")
            
            self.update_nav_buttons()

    def update_nav_buttons(self):
        has_multiple = len(self.image_list) > 1
        self.btn_prev.setEnabled(has_multiple)
        self.btn_next.setEnabled(has_multiple)

    def prev_image(self):
        if self.image_list:
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.load_current_image()

    def next_image(self):
        if self.image_list:
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.load_current_image()

    def load_current_image(self):
        if not self.image_list:
            return
            
        self.current_image_path = self.image_list[self.current_index]
        self.status_label.setText(f"当前文件: {os.path.basename(self.current_image_path)} ({self.current_index + 1}/{len(self.image_list)})")
        
        # 显示原图
        pixmap = QPixmap(self.current_image_path)
        if not pixmap.isNull():
            self.lbl_origin.setPixmap(pixmap.scaled(self.lbl_origin.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_origin.setText("无法加载图片")
            
        self.lbl_result.setText("等待检测...")
        self.lbl_result.setPixmap(QPixmap()) # 清空结果

    def detect_image(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先打开一张图片")
            return
            
        if not self.model:
            QMessageBox.critical(self, "错误", "模型未加载")
            return
            
        self.status_label.setText("正在推理...")
        self.lbl_result.setText("正在检测中...")
        QApplication.processEvents() # 刷新界面
        
        try:
            # 运行推理
            # imgsz=1024 保持与训练一致，conf=0.25 默认置信度
            results = self.model.predict(self.current_image_path, imgsz=1024, conf=0.25)
            
            # 获取结果图 (numpy array BGR)
            res_plotted = results[0].plot()
            
            # 转换为 RGB 以供 Qt 显示
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # 转换为 QImage
            h, w, ch = res_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(res_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 显示结果
            self.lbl_result.setPixmap(QPixmap.fromImage(q_img).scaled(self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 统计信息
            count = len(results[0].boxes)
            self.status_label.setText(f"检测完成: 发现 {count} 个目标")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理出错: {str(e)}")
            self.status_label.setText("推理失败")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
