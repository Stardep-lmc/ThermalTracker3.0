# models/matcher.py
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 处理预测 Class (防止 [N] -> [N, 1])
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  
        if out_prob.dim() == 1:
            out_prob = out_prob.unsqueeze(-1)
            
        # 2. 处理预测 BBox (防止 [N] -> [N/4, 4] 以外的情况)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # [新增] 强制确保是 2D Tensor
        if out_bbox.dim() == 1:
             out_bbox = out_bbox.view(-1, 4)

        # 3. 处理真实 GT
        if len(targets) > 0:
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            
            # [新增] 核心修复：防止 GT BBox 变成 1维向量 (例如 [4] 变为 [N, 4])
            if tgt_bbox.dim() == 1:
                tgt_bbox = tgt_bbox.view(-1, 4)
        else:
            return []

        # 4. 计算 Cost 矩阵
        # Classification Cost
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        
        # 这里的 indexing 要求 out_prob 是 [N, 1] 或 [N, C]
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # L1 Box Cost (inputs must be 2D: [N, 4], [M, 4])
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU Box Cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        
        # 匈牙利匹配
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)