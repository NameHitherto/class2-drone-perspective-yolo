import time
import os
import glob
from ultralytics import YOLO
import pandas as pd

def evaluate_model(model_path='runs/train/visdrone_yolo11_02/weights/best.pt', data_yaml='visdrone.yaml'):
    """
    评估模型性能：
    1. 定量评估：计算各类别 AP 和 mAP
    2. 速度评估：计算 FPS
    3. 定性评估：保存检测结果可视化图
    """
    
    # 检查模型是否存在，不存在则使用预训练模型演示
    if not os.path.exists(model_path):
        print(f"Warning: Trained model not found at {model_path}.")
        print("Using 'yolo11s.pt' for demonstration purposes.")
        model = YOLO('yolo11s.pt')
    else:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)

    print("\n" + "="*50)
    print("1. 定量结果评估 (Quantitative Results)")
    print("="*50)
    
    # 运行验证
    # 注意: 这里使用 'val' 集进行评估。如果需要提交测试集结果，请更改 split='test'
    metrics = model.val(data=data_yaml, split='val', verbose=False, imgsz=1024, batch=4)
    
    # 获取类别名称和对应的 AP50-95
    class_names = metrics.names
    ap50_95 = metrics.box.maps # 这是一个数组，包含每个类别的 mAP50-95
    
    # 构建结果表格
    results_data = []
    for i, name in class_names.items():
        # 确保索引在范围内
        if i < len(ap50_95):
            results_data.append({
                "ID": i,
                "Class": name,
                "AP(50-95)": f"{ap50_95[i]:.4f}"
            })
    
    # 添加总体 mAP
    results_data.append({
        "ID": "-",
        "Class": "ALL (mAP)",
        "AP(50-95)": f"{metrics.box.map:.4f}"
    })
    
    # 打印表格
    df = pd.DataFrame(results_data)
    print("\n各类别性能表:")
    print(df.to_string(index=False))

    print("\n" + "="*50)
    print("2. 推理速度评估 (Inference Speed / FPS)")
    print("="*50)
    
    # 获取验证集图片用于测试速度
    # 假设数据集结构符合 visdrone.yaml
    val_img_dir = os.path.join('datasets', 'VisDrone2019-DET-val', 'images')
    test_images = glob.glob(os.path.join(val_img_dir, '*.jpg'))[:100] # 测试100张图片
    
    if not test_images:
        print(f"Error: No images found in {val_img_dir}. Cannot calculate FPS.")
    else:
        print(f"Running inference on {len(test_images)} images to calculate FPS...")
        
        # 预热
        model.predict(test_images[0], verbose=False, imgsz=1024)
        
        start_time = time.time()
        for img in test_images:
            model.predict(img, verbose=False, imgsz=1024)
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = len(test_images) / total_time
        avg_latency = (total_time / len(test_images)) * 1000
        
        print(f"Total Time: {total_time:.2f} s")
        print(f"Average Latency: {avg_latency:.2f} ms/img")
        print(f"FPS: {fps:.2f}")

    print("\n" + "="*50)
    print("3. 定性结果可视化 (Qualitative Results)")
    print("="*50)
    
    if test_images:
        vis_images = test_images[:5] # 可视化前5张
        save_dir = 'runs/detect/visdrone_eval'
        
        print(f"Visualizing {len(vis_images)} images...")
        model.predict(vis_images, save=True, project='runs/detect', name='visdrone_eval', exist_ok=True, imgsz=1024)
        
        print(f"可视化结果已保存至: {os.path.abspath(save_dir)}")
    else:
        print("No images to visualize.")

if __name__ == "__main__":
    evaluate_model()
