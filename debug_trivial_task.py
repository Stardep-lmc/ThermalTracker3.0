# debug_trivial_task.py
import torch
import torch.optim as optim
from models.motr import build as build_model
from util.misc import nested_tensor_from_tensor_list, NestedTensor
import argparse

# ... (复制上面的 get_args 函数) ...
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_queries', default=10, type=int) 
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # 这里其实用不到
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    print(">>> 1. 构建模型...")
    model, criterion = build_model(args)
    model.to(device)
    model.train()
    
    # 2. 构造“傻瓜”数据
    # 一张全黑图，中间有个白块
    fake_img = torch.zeros(1, 3, 128, 128).to(device)
    # 在 (64, 64) 附近画个白块 (0.5, 0.5)
    fake_img[:, :, 50:78, 50:78] = 1.0 
    
    # 构造 NestedTensor (Mask 全 False)
    mask = torch.zeros((1, 128, 128), dtype=torch.bool).to(device)
    samples = NestedTensor(fake_img, mask)
    samples_thermal = NestedTensor(torch.zeros_like(fake_img), mask) # 没用的 thermal
    
    # GT 就在中间
    targets = [{
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32).to(device),
        'labels': torch.tensor([0], dtype=torch.int64).to(device)
    }]
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    print(">>> 3. 开始傻瓜特征测试...")
    for i in range(51):
        optimizer.zero_grad()
        outputs = model(samples, samples_thermal)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses.backward()
        optimizer.step()
        
        if i % 10 == 0:
            pred = outputs['pred_boxes'][0, outputs['pred_logits'].sigmoid()[0].argmax()]
            print(f"Step {i} | Loss: {losses.item():.4f} | Pred: {pred.tolist()}")

    print(">>> 判决：如果这里Loss能降下去，说明Transformer能利用图像特征。如果降不下去，说明算子有问题。")

if __name__ == "__main__":
    main()