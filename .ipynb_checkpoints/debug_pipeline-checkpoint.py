import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# =================配置区域=================
DATA_ROOT = './data/GTOT' 
SEQ_NAME = 'Tricycle'      
IMG_SIZE = (640, 512)      
# ==========================================

class SimpleGTOTLoader:
    def __init__(self, root, seq):
        self.seq_path = os.path.join(root, seq)
        self.rgb_dir = os.path.join(self.seq_path, 'v')
        self.th_dir = os.path.join(self.seq_path, 'i')
        
        # 1. 检查图片
        if not os.path.exists(self.rgb_dir):
            print(f"❌ 错误: 找不到 RGB 文件夹")
            return
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, '*.png')))
        self.th_files = sorted(glob.glob(os.path.join(self.th_dir, '*.png')))
        print(f"✅ 序列 [{seq}]: {len(self.rgb_files)} 图片")

        # 2. 查找 GT
        txt_files = glob.glob(os.path.join(self.seq_path, '*.txt'))
        target_txt_path = ""
        if len(txt_files) > 0:
            target_txt_path = txt_files[0]
            for f in txt_files:
                if 'ground' in f.lower(): target_txt_path = f; break
            
            print(f"✅ 锁定 GT 文件: {target_txt_path}")
            
            self.gts = []
            with open(target_txt_path, 'r') as f:
                for line in f:
                    line = line.replace(',', ' ').replace('\t', ' ').strip()
                    parts = [float(x) for x in line.split()]
                    if len(parts) >= 4:
                        self.gts.append(parts[:4]) 
            self.gts = np.array(self.gts)
        else:
            print(f"❌ 找不到 txt 文件")

    def get_item(self, idx):
        rgb_path = self.rgb_files[idx]
        th_path = self.th_files[idx]
        
        img_rgb = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_th = cv2.imread(th_path, cv2.IMREAD_GRAYSCALE)
        
        h_raw, w_raw = img_rgb.shape[:2]
        img_rgb_re = cv2.resize(img_rgb, IMG_SIZE)
        img_th_re = cv2.resize(img_th, IMG_SIZE)
        
        # GT 处理 [关键修正]
        if idx < len(self.gts):
            # 假设原始格式是 x1, y1, x2, y2
            raw = self.gts[idx]
            x1_raw, y1_raw, x2_raw, y2_raw = raw[0], raw[1], raw[2], raw[3]
            
            # 【核心修复】：如果你发现框还是不对，这里可能要改回 xywh，但目前看 90% 是 x1y1x2y2
            # 1. 计算真实的宽高
            w_real = x2_raw - x1_raw
            h_real = y2_raw - y1_raw
            
            # 2. 缩放比例
            scale_x = IMG_SIZE[0] / w_raw
            scale_y = IMG_SIZE[1] / h_raw
            
            # 3. 变换到新尺寸 (xywh 依然是左上角)
            x_new = x1_raw * scale_x
            y_new = y1_raw * scale_y
            w_new = w_real * scale_x
            h_new = h_real * scale_y
            
            # 4. 转为 Center (cx, cy) 并归一化 (0-1)
            cx_norm = (x_new + w_new / 2) / IMG_SIZE[0]
            cy_norm = (y_new + h_new / 2) / IMG_SIZE[1]
            w_norm = w_new / IMG_SIZE[0]
            h_norm = h_new / IMG_SIZE[1]
            
            target = np.array([cx_norm, cy_norm, w_norm, h_norm])
        else:
            target = np.array([0,0,0,0])

        return img_rgb_re, img_th_re, target

def visualize_and_check():
    loader = SimpleGTOTLoader(DATA_ROOT, SEQ_NAME)
    if not hasattr(loader, 'gts'): return

    idx = 50 
    rgb, th, target = loader.get_item(idx)
    
    # 反解坐标
    H, W = IMG_SIZE[1], IMG_SIZE[0]
    cx, cy, w, h = target
    
    # 还原回左上角坐标 (x1, y1, x2, y2)
    x_center = cx * W
    y_center = cy * H
    box_w = w * W
    box_h = h * H
    
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)
    
    print(f"\n🔍 检查第 {idx} 帧 (修正版 V3):")
    print(f"   - 原始GT: {loader.gts[idx]}")
    print(f"   - 归一化GT: [{cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f}]")
    print(f"   - 画图坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"   - 框的宽度: {x2-x1} 像素 (之前是几百)")
    
    # 画框
    vis_img = rgb.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 保存
    save_path = "debug_data_vis_v3.png"
    th_vis = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    combined = np.hstack((vis_img, th_vis))
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"\n✅ 图片已保存: {save_path}")

if __name__ == "__main__":
    visualize_and_check()