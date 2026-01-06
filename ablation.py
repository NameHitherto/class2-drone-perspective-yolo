from ultralytics import YOLO
import pandas as pd
import os
import torch

def run_ablation_studies():
    """
    运行消融实验 (Ablation Studies)
    对比 Baseline 与 改进模型及其变体
    
    实验设计:
    1. Baseline: YOLO11l (Standard)
    2. Ours (Full): YOLO11l + P2 + SPD-Conv + EMA
    3. Ablation 1 (No P2): YOLO11l + SPD-Conv + EMA
    4. Ablation 2 (No SPD): YOLO11l + P2 + EMA
    5. Ablation 3 (No EMA): YOLO11l + P2 + SPD-Conv
    """
    
    # 定义实验配置
    experiments = [
        {
            "name": "baseline_yolo11l",
            "yaml": "yolo11l.yaml",
            "description": "Baseline (YOLO11l)"
        },
        {
            "name": "ours_full",
            "yaml": "yolo11l+P2+SPDConv+EMA.yaml",
            "description": "Ours (P2 + SPD + EMA)"
        },
        {
            "name": "ablation_no_p2",
            "yaml": "yolo11l+SPDConv+EMA.yaml",
            "description": "Ablation (w/o P2)"
        },
        {
            "name": "ablation_no_spd",
            "yaml": "yolo11l+P2+EMA.yaml",
            "description": "Ablation (w/o SPD)"
        },
        {
            "name": "ablation_no_ema",
            "yaml": "yolo11l+P2+SPDConv.yaml",
            "description": "Ablation (w/o EMA)"
        }
    ]
    
    results = []
    
    print("开始运行消融实验... (这需要较长时间，请确保有足够的算力)")
    
    # 检查权重文件是否存在
    base_weight = 'yolo11l.pt'
    weight_dir = 'weight'
    if os.path.exists(os.path.join(weight_dir, base_weight)):
        weight_path = os.path.join(weight_dir, base_weight)
    elif os.path.exists(base_weight):
        weight_path = base_weight
    else:
        print(f"Warning: {base_weight} not found in 'weight/' or current dir. Will attempt download.")
        weight_path = base_weight

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp['description']}")
        print(f"YAML: {exp['yaml']}")
        print(f"{'='*60}")
        
        if not os.path.exists(exp['yaml']):
            print(f"Error: YAML file {exp['yaml']} not found. Skipping.")
            continue

        # 1. 构建模型
        # 使用 YAML 构建网络结构，并加载预训练权重 (transfer learning)
        try:
            # 即使是 baseline，也建议显式指定 yaml 以确保配置一致
            model = YOLO(exp['yaml'])
            
            # 加载权重
            # 注意：对于修改了架构的模型，加载权重时会出现不匹配的警告，这是正常的
            # 我们只需要加载 backbone 等匹配的部分
            model.load(weight_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
            continue
        
        # 2. 训练
        # 保持与 train.py 一致的训练参数，确保公平对比
        train_args = {
            'data': 'visdrone.yaml',
            'epochs': 50,           # 统一训练轮数
            'imgsz': 1024,          # 统一输入分辨率 (针对 VisDrone 小目标)
            'batch': 4,             # 统一 Batch Size (根据显存调整)
            'workers': 8,           # Windows 优化
            'project': 'runs/ablation',
            'name': exp['name'],
            'exist_ok': True,
            'verbose': False,       # 减少输出
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'cache': True,
            # 其他增强参数保持默认或与 train.py 一致
            'patience': 20,
            'close_mosaic': 10,
        }
        
        print(f"开始训练 {exp['name']}...")
        try:
            model.train(**train_args)
        except Exception as e:
            print(f"训练过程中出错: {e}")
            continue
        
        # 3. 验证与指标获取
        print(f"正在验证 {exp['name']}...")
        try:
            # 验证时使用 batch=1 来更准确地模拟单张图片推理的 FPS (Latency)
            # 或者使用默认 batch 来测试吞吐量。这里为了 FPS 计算方便，使用 batch=1
            metrics = model.val(split='val', imgsz=1024, batch=1, verbose=False)
            
            # 获取 mAP
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            
            # 获取 FPS
            # metrics.speed 是一个字典: {'preprocess': 0.4, 'inference': 2.3, 'loss': 0.0, 'postprocess': 0.6} (单位 ms)
            # FPS = 1000 / (preprocess + inference + postprocess)
            speed = metrics.speed
            total_time_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
            fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0
            
            print(f"Result: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, FPS={fps:.2f}")
            
            # 记录结果
            results.append({
                "Experiment": exp['description'],
                "YAML": exp['yaml'],
                "mAP50": f"{map50:.4f}",
                "mAP50-95": f"{map50_95:.4f}",
                "FPS": f"{fps:.2f}",
                "Inference Time (ms)": f"{speed['inference']:.2f}"
            })
            
        except Exception as e:
            print(f"验证过程中出错: {e}")
            results.append({
                "Experiment": exp['description'],
                "YAML": exp['yaml'],
                "mAP50": "Error",
                "mAP50-95": "Error",
                "FPS": "Error",
                "Inference Time (ms)": "Error"
            })
        
    # 4. 输出结果表
    print("\n" + "="*80)
    print("消融实验结果汇总 (Ablation Study Results)")
    print("="*80)
    
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        df = df[["Experiment", "YAML", "mAP50", "mAP50-95", "FPS", "Inference Time (ms)"]]
        print(df.to_string(index=False))
        
        # 保存
        save_path = 'runs/ablation/ablation_results_final.csv'
        # 确保目录存在
        os.makedirs('runs/ablation', exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n结果已保存至 {save_path}")
    else:
        print("没有产生任何结果。")

if __name__ == "__main__":
    run_ablation_studies()
