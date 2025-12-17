# datasets/mot_rgbt.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import datasets.transforms_rgbt as T
import numpy as np

class GTOTDataset(Dataset):
    def __init__(self, root_path, transforms=None):
        self.root = Path(root_path)
        self.transforms = transforms
        
        # --- 调试模式：只训练 Tricycle ---
        # ⚠️ 正式训练时，请注释掉下面这几行，或者改为所有文件夹
        target_seq = self.root / 'Tricycle'
        if target_seq.exists():
            self.sequences = [target_seq]
            print(f"🔥 DEBUG MODE: Only loading sequence: {target_seq.name}")
        else:
            # 正式模式：扫描所有子文件夹
            self.sequences = [x for x in self.root.iterdir() if x.is_dir()]
        
        self.samples = []
        valid_seq_count = 0
        
        for seq in self.sequences:
            # --- 1. 定位 RGB 和 Thermal 文件夹 ---
            # 兼容 GTOT 的各种命名习惯
            if (seq / 'v').exists() and (seq / 'i').exists():
                rgb_dir = seq / 'v'
                thermal_dir = seq / 'i'
            elif (seq / 'visible').exists() and (seq / 'infrared').exists():
                rgb_dir = seq / 'visible'
                thermal_dir = seq / 'infrared'
            else:
                # 找不到图片文件夹，跳过
                continue

            # --- 2. 智能查找 GT 文件 (修复) ---
            # 不再硬编码文件名，而是找目录下的 .txt
            txt_files = sorted(list(seq.glob('*.txt')))
            gt_path = None
            
            if len(txt_files) > 0:
                # 优先找名字里带 'ground' 的
                for t in txt_files:
                    if 'ground' in t.name.lower():
                        gt_path = t
                        break
                # 如果没找到带 ground 的，就默认取第一个 (比如 Tricycle.txt)
                if gt_path is None:
                    gt_path = txt_files[0]
            
            if gt_path is None or not gt_path.exists():
                print(f"⚠️ Warning: No GT file found in {seq.name}, skipping.")
                continue

            # --- 3. 读取 GT 数据 ---
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            
            # --- 4. 获取图片列表 ---
            exts = ['*.png', '*.jpg', '*.bmp', '*.jpeg']
            rgb_files = sorted([f for ext in exts for f in rgb_dir.glob(ext)])
            thermal_files = sorted([f for ext in exts for f in thermal_dir.glob(ext)])
            
            # 确保对齐
            min_len = min(len(lines), len(rgb_files), len(thermal_files))
            if min_len == 0: continue
            
            valid_seq_count += 1
            
            # 读取第一张图，获取图像尺寸 (用于归一化检查)
            # 假设一个序列里的图片尺寸是一样的
            try:
                with Image.open(rgb_files[0]) as tmp_img:
                    seq_w, seq_h = tmp_img.size
            except:
                seq_w, seq_h = 640, 480 # Fallback
            
            for i in range(min_len):
                line = lines[i].strip().replace(',', ' ').replace('\t', ' ').split()
                try:
                    raw_box = list(map(float, line))
                    if len(raw_box) < 4: continue
                except ValueError:
                    continue 
                
                # [核心修复] 坐标处理逻辑
                # 基于 Debug 结果，我们优先假设是 [x1, y1, x2, y2]
                x1, y1, x2, y2 = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
                
                # 计算宽高
                w_box = x2 - x1
                h_box = y2 - y1
                
                # 鲁棒性检查：
                # 如果算出来的宽或高是负数，或者宽大得离谱(超过图像宽度的90%且起点不是0)，
                # 那么原数据可能本身就是 xywh 格式 (有些序列可能是混杂的)
                if w_box <= 0 or h_box <= 0:
                    # 回退到 xywh 假设
                    w_box = x2 # 这里 x2 位置其实是 w
                    h_box = y2 # 这里 y2 位置其实是 h
                
                # 存储为绝对坐标 xywh
                box = [x1, y1, w_box, h_box]
                
                self.samples.append({
                    "rgb_path": str(rgb_files[i]),
                    "thermal_path": str(thermal_files[i]),
                    "box": box, 
                    "seq_name": seq.name,
                    "frame_idx": i
                })
        
        print(f"✅ Dataset Loaded: {valid_seq_count} sequences, {len(self.samples)} frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 打开图片
        img_rgb = Image.open(item['rgb_path']).convert('RGB')
        # Thermal 转为单通道 'L' 灰度图
        img_thermal = Image.open(item['thermal_path']).convert('L') 
        
        # 强制 Resize Thermal (重要修复: 防止尺寸不匹配报错)
        if img_rgb.size != img_thermal.size:
            img_thermal = img_thermal.resize(img_rgb.size, Image.BILINEAR)
        
        w, h = img_rgb.size
        
        # 2. 处理 BBox (XYWH -> CX CY W H 归一化)
        box = item['box'] # x, y, w, h (绝对坐标)
        
        # 转换为 Center 格式
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        bw = box[2]
        bh = box[3]
        
        # 构建 Target 字典 (DETR 需要的格式)
        target = {}
        # 归一化到 0-1
        # 增加 clamp 防止坐标轻微越界 (如 1.0001)
        target['boxes'] = torch.tensor([[
            np.clip(cx / w, 0, 1),
            np.clip(cy / h, 0, 1),
            np.clip(bw / w, 0, 1),
            np.clip(bh / h, 0, 1)
        ]], dtype=torch.float32)
        
        target['labels'] = torch.tensor([0], dtype=torch.int64) # 只有一类：目标
        
        # 3. 数据增强
        if self.transforms is not None:
            img_rgb, img_thermal, target = self.transforms(img_rgb, img_thermal, target)
            
        return img_rgb, img_thermal, target

def build(image_set, args):
    root = Path(args.data_path)
    # 确保传入正确的 transforms
    dataset = GTOTDataset(root, transforms=T.make_rgbt_transforms(image_set))
    return dataset