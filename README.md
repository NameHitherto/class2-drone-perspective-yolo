# Drone Perspective Object Detection (VisDrone) with YOLO11

基于 Ultralytics YOLO11 的无人机视角（VisDrone2019-DET）目标检测工程，包含：
- VisDrone 标注转 YOLO 格式脚本（生成 `labels/`）
- 训练脚本（可切换不同改进结构：P2 / SPDConv / EMA）
- 评估脚本（mAP、每类 AP、FPS、可视化结果保存）
- 消融实验脚本（统一训练/验证配置自动对比）
- 简易 PyQt5 GUI 图片推理 Demo

## 1. 项目结构

关键文件：
- 训练：[`train.py`](train.py)（入口函数：[`train_drone_model`](train.py)）
- 评估：[`evaluate.py`](evaluate.py)（入口函数：[`evaluate_model`](evaluate.py)）
- 消融：[`ablation.py`](ablation.py)（入口函数：[`run_ablation_studies`](ablation.py)）
- 数据转换：[`convert_visdrone.py`](convert_visdrone.py)（入口函数：[`convert_visdrone_to_yolo`](convert_visdrone.py)）
- GUI 推理：[`gui_app.py`](gui_app.py)（主类：[`DetectionApp`](gui_app.py)）
- 数据集配置：[`visdrone.yaml`](visdrone.yaml)
- 模型结构（消融用）：
  - Full：[`yolo11l+P2+SPDConv+EMA.yaml`](yolo11l+P2+SPDConv+EMA.yaml)
  - w/o P2：[`yolo11l+SPDConv+EMA.yaml`](yolo11l+SPDConv+EMA.yaml)
  - w/o SPD：[`yolo11l+P2+EMA.yaml`](yolo11l+P2+EMA.yaml)
  - w/o EMA：[`yolo11l+P2+SPDConv.yaml`](yolo11l+P2+SPDConv.yaml)
  - Base： [`yolo11l.yaml`](yolo11l.yaml)

数据目录（与 [`visdrone.yaml`](visdrone.yaml) 保持一致）：
- `datasets/VisDrone2019-DET-train/images`
- `datasets/VisDrone2019-DET-train/annotations`
- `datasets/VisDrone2019-DET-val/images`
- `datasets/VisDrone2019-DET-val/annotations`
- `datasets/VisDrone2019-DET-test-dev/images`
- `datasets/VisDrone2019-DET-test-dev/annotations`

> 说明：Ultralytics/YOLO 默认会在 `images/` 同级目录寻找 `labels/`，本项目转换脚本会生成：`datasets/<split>/labels/*.txt`。

---

## 2. 环境准备（Python + pip）

推荐：
- Python：3.9 / 3.10 / 3.11
- 有 CUDA 的 GPU（训练/高速推理建议）

### 2.1 创建虚拟环境（示例）
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2.2 安装依赖
```bash
pip install -U pip
pip install -r requirements.txt
```

> 注意：PyTorch（`torch/torchvision`）建议按你的 CUDA 版本从官方命令安装：
> https://pytorch.org/get-started/locally/

---

## 3. VisDrone 标注转换为 YOLO

确保 `datasets/VisDrone2019-DET-*/annotations` 存在（VisDrone 原始 txt 标注目录），然后运行：

```bash
python convert_visdrone.py
```

转换逻辑见：[`convert_visdrone_to_yolo`](convert_visdrone.py)，类别映射与 [`visdrone.yaml`](visdrone.yaml) 的 `names` 保持一致（VisDrone 1~10 -> YOLO 0~9）。

---

## 4. 训练（Train）

训练入口：[`train_drone_model`](train.py)

当前默认逻辑：
- 优先从 `weight/yolo11l.pt` 加载权重(*需要自行下载*)
- 并使用结构 YAML `yolo11l+P2+EMA.yaml` 构建网络后加载权重（transfer learning）

运行：
```bash
python train.py
```

训练关键参数（在 [`train.py`](train.py) 内）：
- `imgsz=1024`（针对小目标）
- `batch=4`
- `epochs=80`
- `project='runs/train'`
- `name='yolo11l+P2+EMA'`

> 若要切换到 Full（P2+SPDConv+EMA）或其它消融结构，修改 [`train.py`](train.py) 中 `YOLO('...yaml')` 指向对应 YAML，例如：
> - Full：[`yolo11l+P2+SPDConv+EMA.yaml`](yolo11l+P2+SPDConv+EMA.yaml)

---

## 5. 评估（Evaluate）

评估入口：[`evaluate_model`](evaluate.py)

默认行为：
- 若找不到 `best.pt`，会用 `yolo11s.pt` 演示
- 定量：在 `split='val'` 上计算每类 AP 与总体 mAP
- 速度：对验证集前 100 张图片测 FPS
- 可视化：保存前 5 张预测结果到 `runs/detect/visdrone_eval`

运行：
```bash
python evaluate.py
```

如需指定权重文件：
```bash
python -c "from evaluate import evaluate_model; evaluate_model(model_path='runs/train/<exp>/weights/best.pt', data_yaml='visdrone.yaml')"
```

---

## 6. 消融实验（Ablation）

消融入口：[`run_ablation_studies`](ablation.py)

包含 5 组实验（Baseline / Full / 去 P2 / 去 SPD / 去 EMA），并统一训练配置以便公平对比。

运行：
```bash
python ablation.py
```

输出：
- 控制台汇总表
- 保存到：`runs/ablation/ablation_results_final.csv`

> 提示：消融训练开销大；可先把 `epochs` 调小验证流程（在 [`ablation.py`](ablation.py) 的 `train_args['epochs']`）。

---

## 7. GUI 图片推理 Demo（PyQt5）

入口：[`gui_app.py`](gui_app.py)，主类：[`DetectionApp`](gui_app.py)

运行：
```bash
python gui_app.py
```

默认加载权重优先级（见 `DetectionApp.load_model()`）：
1. `runs/train/visdrone_yolo11_02/weights/best.pt`
2. `weight/yolo11l.pt`
3. `yolo11l.pt`（自动下载）

推理参数：
- `imgsz=1024`
- `conf=0.25`

---

## 8. 常见问题（FAQ）

### 8.1 训练时找不到 labels
先运行转换脚本：[`convert_visdrone.py`](convert_visdrone.py)，确保生成：
- `datasets/VisDrone2019-DET-train/labels`
- `datasets/VisDrone2019-DET-val/labels`

### 8.2 `best.pt` 在哪里？
Ultralytics 默认在：
- `runs/train/<name>/weights/best.pt`

你也可以在 VS Code 的文件树里查看 `runs/train/`。

### 8.3 FPS 的计算口径
- [`evaluate.py`](evaluate.py)：用 `model.predict()` 循环跑图片，按总耗时计算吞吐
- [`ablation.py`](ablation.py)：用 `metrics.speed`（ms）估算 $FPS = 1000 / (t_{pre}+t_{inf}+t_{post})$

---

## License
本仓库内 `ultralytics/` 目录遵循其原项目许可证声明（见文件头部注释）。其余自编脚本默认建议仅用于学习/研究用途；如需商用请自行核对依赖与许可证。