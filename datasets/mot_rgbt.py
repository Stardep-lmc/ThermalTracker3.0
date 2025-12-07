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
        #self.sequences = [x for x in self.root.iterdir() if x.is_dir()]
        target_seq = self.root / 'Tricycle'
        if target_seq.exists():
            self.sequences = [target_seq]
        else:
            # 如果没有 Tricycle，就随便取第一个
            all_seqs = [x for x in self.root.iterdir() if x.is_dir()]
            self.sequences = [all_seqs[0]]
            
        print(f"🔥 DEBUG MODE: Only training on {self.sequences[0].name}")
        
        self.samples = []
        
        print(f"Loading GTOT from {self.root}...")
        
        valid_seq_count = 0
        
        for seq in self.sequences:
            # --- 1. 定位 RGB 和 Thermal 文件夹 ---
            # 优先匹配 v/i，其次 visible/infrared
            if (seq / 'v').exists() and (seq / 'i').exists():
                rgb_dir = seq / 'v'
                thermal_dir = seq / 'i'
            elif (seq / 'visible').exists() and (seq / 'infrared').exists():
                rgb_dir = seq / 'visible'
                thermal_dir = seq / 'infrared'
            else:
                # 找不到图片文件夹，跳过
                continue

            # --- 2. 定位 GroundTruth 文件 ---
            # 你的情况：groundTruth_v.txt
            gt_path = seq / 'groundTruth_v.txt'
            if not gt_path.exists():
                # 备选方案
                gt_path = seq / 'groundTruth_i.txt'
            if not gt_path.exists():
                gt_path = seq / 'groundtruth.txt'
            
            if not gt_path.exists():
                # print(f"Skipping {seq.name}: No GT found.")
                continue

            # --- 3. 读取 GT 数据 ---
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            
            # --- 4. 获取图片列表 (支持 png/jpg) ---
            exts = ['*.png', '*.jpg', '*.bmp', '*.jpeg']
            rgb_files = sorted([f for ext in exts for f in rgb_dir.glob(ext)])
            thermal_files = sorted([f for ext in exts for f in thermal_dir.glob(ext)])
            
            # --- 5. 对齐长度 ---
            # 取三者最小长度，确保一一对应
            min_len = min(len(lines), len(rgb_files), len(thermal_files))
            
            if min_len == 0:
                continue
            
            valid_seq_count += 1
            
            # --- 6. 构建样本索引 ---
            for i in range(min_len):
                line = lines[i].strip().replace(',', ' ').split()
                try:
                    # GTOT 格式: x_min, y_min, w, h
                    box = list(map(float, line)) 
                except ValueError:
                    continue 
                
                self.samples.append({
                    "rgb_path": str(rgb_files[i]),
                    "thermal_path": str(thermal_files[i]),
                    "box": box, # 绝对坐标 xywh
                    "seq_name": seq.name,
                    "frame_idx": i
                })
        
        print(f"Dataset Loaded: Found {len(self.samples)} aligned frames from {valid_seq_count} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 打开图片
        # RGB 转为 3通道
        img_rgb = Image.open(item['rgb_path']).convert('RGB')
        # Thermal 转为 单通道 (灰度)
        img_thermal = Image.open(item['thermal_path']).convert('L') 
        if img_rgb.size != img_thermal.size:
            img_thermal = img_thermal.resize(img_rgb.size, Image.BILINEAR)
        w, h = img_rgb.size
        
        # 2. 处理 BBox (XYWH -> CX CY W H 归一化)
        box = item['box'] # x, y, w, h (绝对坐标)
        
        # 转换为中心点坐标 cx, cy
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        bw = box[2]
        bh = box[3]
        
        # 归一化 (0~1)
        target = {}
        target['boxes'] = torch.tensor([[cx / w, cy / h, bw / w, bh / h]], dtype=torch.float32)
        
        # 3. 构造其他 Target 信息
        target['labels'] = torch.tensor([0], dtype=torch.int64) 
        # 追踪 ID，因为是单目标，每一帧里只有一个对象，我们暂且给它 ID=0
        # 如果是多目标数据集，这里需要解析真实的 track_id
        target['ids'] = torch.tensor([0], dtype=torch.int64) 
        
        target['orig_size'] = torch.tensor([h, w])
        target['size'] = torch.tensor([h, w])
        
        # 4. 数据增强 (同步变换)
        if self.transforms is not None:
            img_rgb, img_thermal, target = self.transforms(img_rgb, img_thermal, target)
            
        return img_rgb, img_thermal, target

def build(image_set, args):
    # 这里假设 args.data_path 是 GTOT 的根目录
    root = Path(args.data_path)
    
    # 真实训练时，需要区分 train/val
    # 这里简单处理：都返回完整数据集，后续可以在 main.py 里做 subset split
    dataset = GTOTDataset(root, transforms=T.make_rgbt_transforms(image_set))
    return dataset