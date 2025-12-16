from ultralytics import YOLO
import pandas as pd
import os

def run_ablation_studies():
    """
    运行消融实验 (Ablation Studies)
    目的: 验证不同模块或策略对模型性能的影响
    
    实验设计:
    1. Baseline (Ours): YOLO11l + P2 Head, imgsz=1024 (当前最佳配置)
    2. Exp 1 (结构影响): Standard YOLO11l (无 P2), imgsz=1024 (验证 P2 头的有效性)
    3. Exp 2 (分辨率影响): YOLO11l + P2 Head, imgsz=640 (验证高分辨率的必要性)
    """
    
    # 定义实验配置
    experiments = [
        {
            "name": "exp_custom_p2_1024",
            "yaml": "yolo11l.yaml",
            "weights": "weight/yolo11l.pt",
            "imgsz": 1024,
            "description": "Ours (YOLO11l + P2 Head + 1024sz)"
        },
        {
            "name": "exp_standard_l_1024",
            "yaml": None, # 标准结构不需要 yaml
            "weights": "weight/yolo11l.pt",
            "imgsz": 1024,
            "description": "Standard YOLO11l (No P2 Head)"
        },
        {
            "name": "exp_custom_p2_640",
            "yaml": "yolo11l.yaml",
            "weights": "weight/yolo11l.pt",
            "imgsz": 640,
            "description": "Ours Low Res (640sz)"
        }
    ]
    
    results = []
    
    print("开始运行消融实验... (注意: 这可能需要很长时间)")
    
    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['description']}")
        
        # 加载模型
        if exp['yaml']:
            # 如果指定了 YAML，则构建自定义结构并加载权重
            if os.path.exists(exp['yaml']):
                model = YOLO(exp['yaml']).load(exp['weights'])
            else:
                print(f"Error: YAML file {exp['yaml']} not found. Skipping.")
                continue
        else:
            # 否则直接加载标准模型
            model = YOLO(exp['weights'])
        
        # 训练
        train_args = {
            'data': 'visdrone.yaml',
            'epochs': 50, # 保持 50 或根据需要调整
            'imgsz': exp['imgsz'],
            'project': 'runs/ablation',
            'name': exp['name'],
            'exist_ok': True,
            'batch': 4, # 保持 batch=4 防止 OOM
            'workers': 0, # Windows 优化
            'verbose': False
        }
        
        model.train(**train_args)
        
        # 验证
        print(f"Validating {exp['name']}...")
        metrics = model.val(split='val', imgsz=exp['imgsz'], batch=4)
        
        # 记录结果
        results.append({
            "Experiment": exp['description'],
            "Input Size": exp['imgsz'],
            "Structure": "Custom P2" if exp['yaml'] else "Standard",
            "mAP50": f"{metrics.box.map50:.4f}",
            "mAP50-95": f"{metrics.box.map:.4f}"
        })
        
    # 输出消融实验对比表
    print("\n" + "="*60)
    print("消融实验结果汇总 (Ablation Study Results)")
    print("="*60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 保存结果到 CSV
    df.to_csv('runs/ablation/ablation_results.csv', index=False)
    print("\n结果已保存至 runs/ablation/ablation_results.csv")

if __name__ == "__main__":
    # 检查是否安装了 ultralytics
    try:
        import ultralytics
        run_ablation_studies()
    except ImportError:
        print("请先安装 ultralytics: pip install ultralytics")
