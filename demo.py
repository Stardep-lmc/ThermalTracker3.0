# demo.py
import argparse
import time
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

# 引入项目模块
from models.motr import build as build_model
from util.misc import nested_tensor_from_tensor_list

def get_args_parser():
    parser = argparse.ArgumentParser('ThermalTracker Inference', add_help=False)
    
    # --- 模型架构参数 (必须与训练时完全一致) ---
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # Loss 参数 (推理不需要，但构建模型时 build函数 需要这些占位符)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)

    # --- 推理配置 ---
    parser.add_argument('--checkpoint', default='output/overfit_test/checkpoint0049.pth', help='Path to model weights')
    parser.add_argument('--data_path', default='./data/GTOT', type=str)
    parser.add_argument('--seq_name', default='Tricycle', type=str, help='Sequence to visualize')
    parser.add_argument('--output_dir', default='output/vis', help='Folder to save results')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    
    return parser

def preprocess(img_rgb, img_t):
    # 1. 获取原始尺寸
    w, h = img_rgb.size
    
    # 2. [核心修复] 模拟训练时的 Resize 逻辑 (Min size 800, Max size 1333)
    # 这是 Deformable DETR / MOTR 的标准 val_transform
    target_size = 800
    max_size = 1333
    
    scale = float(target_size) / float(min(h, w))
    
    # 防止长边超标
    if max(h, w) * scale > max_size:
        scale = float(max_size) / float(max(h, w))
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 执行 Resize
    img_rgb = img_rgb.resize((new_w, new_h), Image.BILINEAR)
    img_t = img_t.resize((new_w, new_h), Image.BILINEAR)

    # 3. ToTensor
    img_rgb = TF.to_tensor(img_rgb)
    img_t = TF.to_tensor(img_t)

    # 4. Normalize (必须有!)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_rgb = TF.normalize(img_rgb, mean=mean, std=std)
    
    # Thermal Normalize (假设是单通道)
    if img_t.shape[0] == 1:
        img_t = (img_t - 0.5) / 0.5
    else:
        img_t = TF.normalize(img_t, mean=mean, std=std)
        
    return img_rgb, img_t

def main(args):
    print(f"🚀 Running Inference on sequence: {args.seq_name}")
    device = torch.device(args.device)

    # 1. 构建模型
    model, _ = build_model(args)
    model.to(device)
    model.eval()

    # 2. 加载权重
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理 keys 不匹配
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v     
    model.load_state_dict(new_state_dict)

    # 3. 准备数据路径
    root = Path(args.data_path) / args.seq_name
    rgb_dir = root / 'v' if (root / 'v').exists() else root / 'visible'
    thermal_dir = root / 'i' if (root / 'i').exists() else root / 'infrared'
    
    if not rgb_dir.exists():
        print(f"❌ Error: Image folders not found in {root}")
        return

    # 获取图片列表
    img_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
    thermal_files = sorted(list(thermal_dir.glob('*.png')) + list(thermal_dir.glob('*.jpg')))
    
    min_len = min(len(img_files), len(thermal_files))
    img_files = img_files[:min_len]
    thermal_files = thermal_files[:min_len]

    # 4. 创建输出目录
    save_dir = Path(args.output_dir) / args.seq_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {min_len} frames. Saving visualizations to: {save_dir}")

    # 5. 推理循环
    for i, (rgb_p, t_p) in enumerate(zip(img_files, thermal_files)):
        raw_img = Image.open(rgb_p).convert('RGB')
        raw_t = Image.open(t_p).convert('L')
        w, h = raw_img.size

        img_rgb, img_t = preprocess(raw_img, raw_t)
        
        samples_rgb = nested_tensor_from_tensor_list([img_rgb]).to(device)
        samples_thermal = nested_tensor_from_tensor_list([img_t]).to(device)

        with torch.no_grad():
            outputs = model(samples_rgb, samples_thermal)

        # 后处理：只取 Top-1
        pred_logits = outputs['pred_logits'][0] # [300, 1]
        pred_boxes = outputs['pred_boxes'][0]   # [300, 4]
        
        probas = pred_logits.sigmoid().squeeze() 
        if probas.dim() == 0: probas = probas.unsqueeze(0)

        # [核心] Top-1
        top_score, top_idx = torch.topk(probas, 1)
        
        box = pred_boxes[top_idx]
        score = top_score.item()
        
        # 画图
        draw = ImageDraw.Draw(raw_img)
        cx, cy, bw, bh = box[0].tolist()
        
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        
        color = 'red'
        if score < 0.1: color = 'yellow' 
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, y1), f"{score:.2f}", fill='white')

        raw_img.save(save_dir / f"{i:04d}.jpg")
        
        if i % 50 == 0:
            print(f"Processed {i}/{min_len} frames... (Score: {score:.4f})")

    print("\n✅ Done!")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)