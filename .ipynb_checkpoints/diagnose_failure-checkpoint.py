# diagnose_failure.py
import torch
import datasets
import util.misc as utils
from models.motr import build as build_model
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    # 必须与训练时的参数完全一致！
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=10, type=int) 
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # Loss 参数 (为了计算 validation loss)
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    
    parser.add_argument('--data_path', default='./data/GTOT')
    # 指向你跑完 400 epoch 的那个权重
    parser.add_argument('--checkpoint', default='output/final_sleep_run/checkpoint0399.pth') 
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def unnormalize_box(box, h, w):
    # 将 cx,cy,w,h (0-1) 转换为 x1,y1,x2,y2 (绝对像素)
    cx, cy, bw, bh = box.unbind(-1)
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs("debug_failure_vis", exist_ok=True)
    
    print(f">>> 1. 加载权重: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"❌ 文件不存在: {args.checkpoint}")
        return

    model, criterion = build_model(args)
    model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(">>> 2. 构建验证集 (Val Mode - No Random Crop)...")
    # 必须用 'val' 模式，保证没有随机裁剪，我们要看模型在“干净”数据上的表现
    dataset = datasets.build_dataset(image_set='val', args=args)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    
    print(">>> 3. 开始诊断前 10 帧...")
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, (samples, samples_thermal, targets) in enumerate(data_loader):
        if i >= 10: break 
        
        samples = samples.to(device)
        samples_thermal = samples_thermal.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(samples, samples_thermal)
        
        # 计算 Loss
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 准备画图
        img_tensor = samples.tensors[0].cpu() 
        # 反归一化
        img_vis = img_tensor * std + mean
        img_vis = img_vis.clamp(0, 1)
        pil_img = TF.to_pil_image(img_vis)
        W, H = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        
        # 画 Ground Truth (绿色)
        gt_box = targets[0]['boxes'][0].cpu()
        gt_rect = unnormalize_box(gt_box, H, W)
        draw.rectangle(gt_rect.tolist(), outline='green', width=4)
        
        # 画 Predictions (Top-1 红色，其他黄色)
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        probs = pred_logits.sigmoid().squeeze()
        top_scores, top_idxs = torch.sort(probs, descending=True)
        
        best_score = top_scores[0].item()
        
        # 只画前 3 个预测
        for k in range(min(3, len(top_scores))):
            idx = top_idxs[k]
            score = top_scores[k].item()
            box = pred_boxes[idx].detach().cpu()
            
            rect = unnormalize_box(box, H, W)
            
            color = 'red' if k == 0 else 'yellow'
            width = 3 if k == 0 else 1
            
            draw.rectangle(rect.tolist(), outline=color, width=width)
            # draw.text((rect[0], rect[1]), f"{score:.2f}", fill=color) # 如果没字体库可能会报错，简单起见先不写字
        
        save_path = f"debug_failure_vis/frame_{i:03d}_loss_{total_loss.item():.2f}_score_{best_score:.2f}.jpg"
        pil_img.save(save_path)
        print(f"Frame {i:03d} | Loss: {total_loss.item():.4f} | Top Score: {best_score:.4f} -> Saved {save_path}")

    print("\n>>> 诊断完毕! 请查看 debug_failure_vis 文件夹")
    print("1. 如果 Loss < 2.0 且 红框准 -> 训练成功，是 demo.py 的预处理有问题。")
    print("2. 如果 Loss > 5.0 且 红框乱 -> 训练失败 (没收敛)。")
    print("3. 如果 Loss < 2.0 但 红框歪 -> 坐标系定义不一致 (Padding vs No-Padding)。")

if __name__ == "__main__":
    main()