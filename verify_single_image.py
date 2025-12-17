# verify_single_image.py
import torch
import torch.optim as optim
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from models.motr import build as build_model
from util.misc import nested_tensor_from_tensor_list
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    # 核心模型参数
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=10, type=int) 
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # Loss 系数
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def preprocess(img):
    # 模拟标准的 DETR 输入流程 (Resize -> Tensor -> Normalize)
    img = img.resize((800, 600)) 
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor

def main():
    args = get_args()
    device = torch.device(args.device)
    
    print(">>> 1. 初始化模型 (随机权重)...")
    model, criterion = build_model(args)
    model.to(device)
    model.train()
    
    # 保护 Backbone，快炒 Transformer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
    
    print(">>> 2. 加载单张图片...")
    # 这里用你之前提到的第一帧文件名
    img_path = "data/GTOT/BlackCar/v/00001v.png" 
    
    # [关键] 这是我们之前通过 check_gt 确认过的正确 GT (cx, cy, w, h)
    # 请确认这个数值是你之前看到的正确数值！
    gt_box = [0.496, 0.099, 0.065, 0.096] 
    
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return
    
    raw_img = Image.open(img_path).convert('RGB')
    t_img = preprocess(raw_img).to(device)
    
    # 构造 Input
    samples = nested_tensor_from_tensor_list([t_img])
    # 伪造 Thermal (全0)，为了排除干扰，只验证 RGB 通路
    samples_thermal = nested_tensor_from_tensor_list([torch.zeros_like(t_img)])
    
    # 构造 Target
    targets = [{
        'boxes': torch.tensor([gt_box], dtype=torch.float32).to(device),
        'labels': torch.tensor([0], dtype=torch.int64).to(device)
    }]
    
    print(">>> 3. 开始单图死循环训练 (200 Steps)...")
    print("    目标: Loss 应该迅速下降，Pred Box 应该逼近 GT Box")
    
    for i in range(201):
        optimizer.zero_grad()
        outputs = model(samples, samples_thermal)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
    
        losses.backward()
        optimizer.step()
    
        if i % 20 == 0:
            # 取出置信度最高的预测框
            prob = outputs['pred_logits'].sigmoid()[0] # [10, 1]
            top_idx = prob.argmax()
            pred_box = outputs['pred_boxes'][0, top_idx].detach().cpu().tolist()
            
            # 简单的格式化输出
            p_str = f"[{pred_box[0]:.3f}, {pred_box[1]:.3f}, {pred_box[2]:.3f}, {pred_box[3]:.3f}]"
            g_str = f"[{gt_box[0]:.3f}, {gt_box[1]:.3f}, {gt_box[2]:.3f}, {gt_box[3]:.3f}]"
            
            print(f"Step {i:03d} | Loss: {losses.item():.4f} | Pred: {p_str} | GT: {g_str}")
    
    print("\n>>> 结论判断:")
    final_loss = losses.item()
    
if __name__ == "__main__":
    main()

    
