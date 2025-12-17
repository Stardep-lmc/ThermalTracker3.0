# demo_residual.py
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import argparse
from pathlib import Path
from util.misc import nested_tensor_from_tensor_list
from models.motr import build as build_model # 直接用 build

def preprocess(img_rgb, img_t):
    w, h = img_rgb.size
    # 简单的 Resize 和 Norm，不搞复杂的
    target_size = 800
    scale = float(target_size) / float(min(h, w))
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_rgb = img_rgb.resize((new_w, new_h), Image.BILINEAR)
    img_t = img_t.resize((new_w, new_h), Image.BILINEAR)
    
    img_rgb = TF.to_tensor(img_rgb)
    img_t = TF.to_tensor(img_t)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_rgb = TF.normalize(img_rgb, mean=mean, std=std)
    if img_t.shape[0] == 1: img_t = TF.normalize(img_t, [0.5], [0.5]) # 简单归一化
    
    return img_rgb, img_t

def main():
    parser = argparse.ArgumentParser()
    # 对应训练时的参数
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=10, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # Loss 占位
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    
    parser.add_argument('--checkpoint', default='output/residual_adapter_run/checkpoint0099.pth')
    parser.add_argument('--data_path', default='./data/GTOT')
    parser.add_argument('--seq_name', default='BlackCar')
    parser.add_argument('--output_dir', default='output/vis_1111')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 1. 构建模型 (直接调用项目里的 build)
    print(">>> 构建模型...")
    model, _ = build_model(args)
    model.to(device)
    model.eval()
    
    # 2. 加载权重
    print(f">>> 加载权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    
    # 清洗 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."): k = k[7:]
        new_state_dict[k] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"⚠️ 加载警告: {e}")
        # 如果是因为 Adapter 名字变了，可能需要 strict=False，但尽量保证 True

    # 3. 推理
    root = Path(args.data_path) / args.seq_name
    rgb_dir = root / 'v' if (root / 'v').exists() else root / 'visible'
    thermal_dir = root / 'i' if (root / 'i').exists() else root / 'infrared'
    
    img_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
    thermal_files = sorted(list(thermal_dir.glob('*.png')) + list(thermal_dir.glob('*.jpg')))
    min_len = min(len(img_files), len(thermal_files))
    
    save_dir = Path(args.output_dir) / args.seq_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> 开始推理 {min_len} 帧...")

    for i in range(min_len):
        raw_img = Image.open(img_files[i]).convert('RGB')
        raw_t = Image.open(thermal_files[i]).convert('L')
        w, h = raw_img.size
        img_rgb, img_t = preprocess(raw_img, raw_t)
        
        samples_rgb = nested_tensor_from_tensor_list([img_rgb]).to(device)
        samples_thermal = nested_tensor_from_tensor_list([img_t]).to(device)
        
        with torch.no_grad():
            outputs = model(samples_rgb, samples_thermal)
            
        prob = outputs['pred_logits'][0].sigmoid().squeeze()
        top_score, top_idx = prob.max(0)
        box = outputs['pred_boxes'][0, top_idx]
        
        cx, cy, bw, bh = box.cpu().unbind(-1)
        x1 = (cx - 0.5 * bw) * w
        y1 = (cy - 0.5 * bh) * h
        x2 = (cx + 0.5 * bw) * w
        y2 = (cy + 0.5 * bh) * h
        
        draw = ImageDraw.Draw(raw_img)
        # 根据分数变颜色，方便观察
        color = 'red' if top_score > 0.5 else 'blue'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        raw_img.save(save_dir / f"{i:04d}.jpg")

    print(f"\n✅ 完成！查看 {save_dir}")

if __name__ == "__main__":
    main()