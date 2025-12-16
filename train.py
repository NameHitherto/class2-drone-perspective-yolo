from ultralytics import YOLO
import torch
import os

def train_drone_model():
    # 1. 加载模型
    weight_name = 'yolo11l.pt'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(current_dir, 'weight', weight_name)
    if os.path.exists(weight_path):
        # model = YOLO(weight_path)
        model = YOLO('yolo11l.yaml').load(weight_path) 
    else:
        print(f"模型权重文件未找到: {weight_path}. 进行下载...")
        model = YOLO(weight_name) 

    # 2. 训练参数配置 (算法策略核心)
    results = model.train(
        data='visdrone.yaml',   # 数据集配置文件
        epochs=50,             # 训练轮数 后续可以调整到100轮及以上

        # --- 性能优化关键参数 ---
        imgsz=1024,             # [关键] 增大输入分辨率以检测小目标 (默认640)
        batch=4,               # 根据显存调整，1024分辨率下显存占用较大
        cache=False,           # [优化] 开启RAM缓存，大幅减少磁盘IO时间,但是会占用大量内存
        workers=0,              # [优化] 增加数据加载线程 
        # freeze=10,            # [可选] 冻结骨干网络前 10 层，仅训练头部，极大加快速度但可能略微降低精度
        
        device=0,               # 使用 GPU 0
        
        # 优化器与学习率策略
        optimizer='auto',       # 自动选择 (通常是 SGD 或 AdamW)
        lr0=0.01,               # 初始学习率
        lrf=0.01,               # 最终学习率 (lr0 * lrf)
        
        # 数据增强策略 (针对航拍视角)
        degrees=0.0,            # 旋转角度 (航拍通常是正视，不需要太大旋转)
        mosaic=1.0,             # Mosaic 增强，有助于小目标检测
        mixup=0.1,              # Mixup 增强
        copy_paste=0.1,         # Copy-Paste 增强
        
        # 训练策略
        close_mosaic=10,        # 最后 10 个 epoch 关闭 Mosaic 以精细化定位
        patience=20,            # 早停机制，20轮无提升则停止
        
        project='runs/train',   # 保存路径
        name='visdrone_yolo11_02', # 实验名称
        exist_ok=True           # 覆盖同名实验
    )

    # 3. 验证模型
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # 4. 导出模型 (可选 ONNX 用于部署)
    # print("正在导出 ONNX 模型...")
    # dynamic=True 允许输入不同尺寸的图片，适合 GUI 中用户上传任意图片
    # model.export(format='onnx', dynamic=True) 

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # 开启 cudnn benchmark 可以加速固定尺寸输入的训练
        # torch.backends.cudnn.benchmark = True
        train_drone_model()
    else:
        print("Warning: CUDA not available. Training will be very slow.")