import os
from PIL import Image
from tqdm import tqdm

def convert_visdrone_to_yolo(data_dir, output_dir=None):
    """
    将 VisDrone 标注转换为 YOLO 格式
    VisDrone 类别: 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van, 
                   6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor
    """
    # 映射 VisDrone 类别 (1-10) 到 YOLO 索引 (0-9)
    # 忽略 0 (ignored regions) 和 11 (others)
    class_map = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 
        6: 5, 7: 6, 8: 7, 9: 8, 10: 9
    }

    sets = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    
    for split in sets:
        img_dir = os.path.join(data_dir, split, 'images')
        ann_dir = os.path.join(data_dir, split, 'annotations')
        
        # YOLO 默认会在 images 同级目录下寻找 labels 文件夹
        label_dir = os.path.join(data_dir, split, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        
        print(f"正在转换 {split} ...")
        
        ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
        
        for ann_file in tqdm(ann_files):
            img_name = ann_file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_name)
            
            # 检查图片是否存在以获取尺寸
            if not os.path.exists(img_path):
                continue
                
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except:
                continue

            txt_path = os.path.join(ann_dir, ann_file)
            out_path = os.path.join(label_dir, ann_file)
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            yolo_lines = []
            for line in lines:
                data = line.strip().split(',')
                if len(data) < 8: continue
                
                # VisDrone 格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                x, y, w, h = float(data[0]), float(data[1]), float(data[2]), float(data[3])
                score = float(data[4])
                category = int(data[5])
                
                # 过滤掉忽略区域(0)和其他(11)，通常也过滤掉低分目标
                if category not in class_map:
                    continue
                
                cls_id = class_map[category]
                
                # 转换为 YOLO 中心点格式并归一化
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # 边界检查
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            if yolo_lines:
                with open(out_path, 'w') as f_out:
                    f_out.write('\n'.join(yolo_lines))

if __name__ == "__main__":
    # 假设 datasets 文件夹在当前目录下
    dataset_path = os.path.join(os.getcwd(), "datasets")
    convert_visdrone_to_yolo(dataset_path)