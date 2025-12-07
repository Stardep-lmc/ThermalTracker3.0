# datasets/mot_rgbt.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import datasets.transforms_rgbt as T

class GTOTDataset(Dataset):
    def __init__(self, root_path, transforms=None):
        self.root = Path(root_path)
        self.transforms = transforms
        
        # 扫描所有视频序列目录
        target_seq = self.root / 'Tricycle'
        if target_seq.exists():
            self.sequences = [target_seq]
            print(f"🔥 DEBUG MODE: Only training on {target_seq.name}")
        else:
            self.sequences = [x for x in self.root.iterdir() if x.is_dir()]
        
        self.samples = []
        
        valid_seq_count = 0
        
        for seq in self.sequences:
            # --- 1. 定位 RGB 和 Thermal 文件夹 ---
            if (seq / 'v').exists() and (seq / 'i').exists():
                rgb_dir = seq / 'v'
                thermal_dir = seq / 'i'
            elif (seq / 'visible').exists() and (seq / 'infrared').exists():
                rgb_dir = seq / 'visible'
                thermal_dir = seq / 'infrared'
            else:
                continue

            # --- 2. 定位 GroundTruth 文件 ---
            gt_path = seq / 'groundTruth_v.txt'
            if not gt_path.exists():
                gt_path = seq / 'groundTruth_i.txt'
            if not gt_path.exists():
                gt_path = seq / 'groundtruth.txt'
            
            if not gt_path.exists():
                continue

            # --- 3. 读取 GT 数据 ---
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            
            # --- 4. 获取图片列表 ---
            exts = ['*.png', '*.jpg', '*.bmp', '*.jpeg']
            rgb_files = sorted([f for ext in exts for f in rgb_dir.glob(ext)])
            thermal_files = sorted([f for ext in exts for f in thermal_dir.glob(ext)])
            
            min_len = min(len(lines), len(rgb_files), len(thermal_files))
            if min_len == 0: continue
            
            valid_seq_count += 1
            
            for i in range(min_len):
                line = lines[i].strip().replace(',', ' ').split()
                try:
                    raw_box = list(map(float, line)) 
                except ValueError:
                    continue 
                
                # [核心修复] 坐标格式转换
                # GTOT 的 txt 可能是 [x1, y1, x2, y2] 也可能是 [x, y, w, h]
                # 我们根据数值特征判断：如果第3个数(w/x2) 很大且接近 x1，那它大概率是 x2
                # Tricycle 数据: 181 9 206 26 -> 显然是 x1, y1, x2, y2
                
                x1, y1, v3, v4 = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
                
                # 简单判断逻辑：如果 v3 (假设是宽) + x1 并没有超出图片太大，但 v4 (高) 非常小...
                # 更稳健的方法：计算两种假设的宽高比。
                # 假设 A (xywh): w=206, h=26 -> ratio 8:1 (太扁了)
                # 假设 B (xyxy): w=206-181=25, h=26-9=17 -> ratio 1.5:1 (正常)
                
                # 这里我们强制针对你下载的 GTOT 版本使用 xyxy -> xywh 转换
                # w = x2 - x1
                # h = y2 - y1
                w_box = v3 - x1
                h_box = v4 - y1
                
                # 如果算出来 w 或 h 是负数，说明原数据可能是 xywh，回退
                if w_box <= 0 or h_box <= 0:
                    box = [x1, y1, v3, v4] # 保持 xywh
                else:
                    box = [x1, y1, w_box, h_box] # 转换为 xywh
                
                self.samples.append({
                    "rgb_path": str(rgb_files[i]),
                    "thermal_path": str(thermal_files[i]),
                    "box": box, # 绝对坐标 xywh
                    "seq_name": seq.name,
                    "frame_idx": i
                })
        
        print(f"Dataset Loaded: Found {len(self.samples)} frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 打开图片
        img_rgb = Image.open(item['rgb_path']).convert('RGB')
        img_thermal = Image.open(item['thermal_path']).convert('L') 
        
        # 强制 Resize Thermal (重要修复)
        if img_rgb.size != img_thermal.size:
            img_thermal = img_thermal.resize(img_rgb.size, Image.BILINEAR)
        
        w, h = img_rgb.size
        
        # 2. 处理 BBox (XYWH -> CX CY W H 归一化)
        box = item['box'] # x, y, w, h (绝对坐标)
        
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        bw = box[2]
        bh = box[3]
        
        target = {}
        # 归一化
        target['boxes'] = torch.tensor([[cx / w, cy / h, bw / w, bh / h]], dtype=torch.float32)
        target['labels'] = torch.tensor([0], dtype=torch.int64) 
        target['ids'] = torch.tensor([0], dtype=torch.int64) 
        target['orig_size'] = torch.tensor([h, w])
        target['size'] = torch.tensor([h, w])
        
        # 3. 数据增强
        if self.transforms is not None:
            img_rgb, img_thermal, target = self.transforms(img_rgb, img_thermal, target)
            
        return img_rgb, img_thermal, target

def build(image_set, args):
    root = Path(args.data_path)
    dataset = GTOTDataset(root, transforms=T.make_rgbt_transforms(image_set))
    return dataset