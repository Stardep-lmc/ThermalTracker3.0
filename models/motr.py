# models/motr.py
# ------------------------------------------------------------------------
# 核心修复版：强制 Mask=False (有效)，防止均值预测
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import box_ops
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.matcher import build_matcher

# 手动实现 Sigmoid Focal Loss
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = loss * alpha_t
    return loss

class MOTRGBT(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        # 1. Projections
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for i in range(num_backbone_outs):
            in_channels = backbone.num_channels[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        # 4th level
        for _ in range(1):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
            
        self.input_proj = nn.ModuleList(input_proj_list)
        
        # 2. Heads
        self.class_embed = nn.Linear(hidden_dim, num_classes) 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        self.aux_loss = aux_loss

    def forward(self, samples_rgb, samples_thermal):
        # 1. Backbone
        features_rgb, features_thermal, pos = self.backbone(samples_rgb, samples_thermal)
        
        src_rgb = []
        src_thermal = []
        masks = []
        pos_embeds = []
        
        # 2. Projection & Preparation
        for l, (feat_rgb, feat_th) in enumerate(zip(features_rgb, features_thermal)):
            src_rgb.append(self.input_proj[l](feat_rgb.tensors))
            src_thermal.append(self.input_proj[l](feat_th.tensors))
            
            # === [核弹级修复] 强制 Mask 全为 False (有效) ===
            b, _, h, w = feat_rgb.tensors.shape
            force_valid_mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat_rgb.tensors.device)
            masks.append(force_valid_mask)
            # ============================================
            
            pos_embeds.append(pos[l])
            
        # 3. 4th Level
        if len(self.input_proj) > len(features_rgb):
            last_feat_rgb = self.input_proj[-1](features_rgb[-1].tensors)
            last_feat_thermal = self.input_proj[-1](features_thermal[-1].tensors)
            src_rgb.append(last_feat_rgb)
            src_thermal.append(last_feat_thermal)
            
            # === [核弹级修复] 第4层也强制有效 ===
            b, _, h, w = last_feat_rgb.shape
            force_valid_mask = torch.zeros((b, h, w), dtype=torch.bool, device=last_feat_rgb.device)
            masks.append(force_valid_mask)
            # ==================================
            
            p = pos[-1]
            pos_l = F.interpolate(p, size=last_feat_rgb.shape[-2:], mode='bilinear', align_corners=False)
            pos_embeds.append(pos_l)

        # 4. Transformer
        hs, _ = self.transformer(src_rgb, masks, pos_embeds, src_thermal, self.query_embed.weight)
        
        # 5. Prediction Heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        if src_logits.dim() == 2:
            src_logits = src_logits.unsqueeze(-1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, 
                                     alpha=self.focal_alpha, gamma=2.0)
        loss_ce = loss_ce.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        return {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'boxes': self.loss_boxes}
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(args):
    num_classes = 1 
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = MOTRGBT(backbone, transformer, num_classes=1, num_queries=args.num_queries, aux_loss=True)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes']
    criterion = SetCriterion(1, matcher=matcher, weight_dict=weight_dict, losses=losses)
    return model, criterion