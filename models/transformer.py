# models/transformer.py
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn

# models/transformer.py (部分代码修改)

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        # [Encoder 定义保持不变...]
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # [Decoder 定义保持不变...]
        decoder_layer = RGBTTDecoderLayer(d_model, dim_feedforward,
                                          dropout, activation,
                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # [修改 1/3] 新增：参考点投影层 (把 256 变成 2)
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        
        # [修改 2/3] 新增：初始化投影层
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias, 0.)
        
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        # [保持不变...]
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_rgb, masks_rgb, pos_embeds, src_thermal, query_embed=None):
            """
            src_rgb: List of [B, C, H, W]
            src_thermal: List of [B, C, H, W]
            """
            # --- 1. 准备 RGB 输入 (Flatten & Prepare) ---
            src_flatten_rgb = []
            mask_flatten_rgb = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            
            for lvl, (src, mask, pos_embed) in enumerate(zip(src_rgb, masks_rgb, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                
                src = src.flatten(2).transpose(1, 2) # [B, L, C]
                mask = mask.flatten(1)               # [B, L]
                pos_embed = pos_embed.flatten(2).transpose(1, 2) # [B, L, C]
                
                # 加上 Level Embedding
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                
                src_flatten_rgb.append(src)
                mask_flatten_rgb.append(mask)
                lvl_pos_embed_flatten.append(lvl_pos_embed)

            src_flatten_rgb = torch.cat(src_flatten_rgb, 1)
            mask_flatten_rgb = torch.cat(mask_flatten_rgb, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_rgb.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks_rgb], 1)

            # --- 2. 准备 Thermal 输入 ---
            src_flatten_thermal = []
            for lvl, src in enumerate(src_thermal):
                src = src.flatten(2).transpose(1, 2)
                src_flatten_thermal.append(src)
            src_flatten_thermal = torch.cat(src_flatten_thermal, 1)

            # --- 3. Encoder (Shared or Dual) ---
            memory_rgb = self.encoder(src_flatten_rgb, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten_rgb)
            memory_thermal = self.encoder(src_flatten_thermal, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten_rgb)

            # --- 4. Decoder (Cross-Modality Fusion) ---
            bs, _, c = memory_rgb.shape
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            
            # [关键修复 1] 使用投影层生成初始参考点 [B, Q, 2]
            reference_points = self.reference_points(query_embed).sigmoid()
            
            # [关键修复 2] 扩展维度以匹配 Deformable Attn [B, Q, L, 2]
            reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(self.level_embed), 1)

            hs, inter_references = self.decoder(tgt, reference_points, 
                                                memory_rgb, memory_thermal, 
                                                spatial_shapes, level_start_index, valid_ratios, 
                                                query_embed, mask_flatten_rgb)

            return hs, inter_references


class RGBTTDecoderLayer(nn.Module):
    """
    [创新核心] RGB-Thermal 融合解码层
    """
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross Attention RGB
        self.cross_attn_rgb = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_rgb = nn.Dropout(dropout)
        self.norm_rgb = nn.LayerNorm(d_model)

        # Cross Attention Thermal
        self.cross_attn_thermal = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_thermal = nn.Dropout(dropout)
        self.norm_thermal = nn.LayerNorm(d_model)

        # Fusion Gate
        self.fusion_gate = nn.Linear(d_model, 1) # 输出一个标量权重

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, 
                src_rgb, src_thermal, 
                src_spatial_shapes, level_start_index, src_padding_mask=None):
        
        # 1. Self Attention (Query-Query)
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross Attention RGB
        tgt_rgb = self.cross_attn_rgb(tgt + query_pos, reference_points, src_rgb, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_rgb = self.norm_rgb(tgt + self.dropout_rgb(tgt_rgb))

        # 3. Cross Attention Thermal
        tgt_thermal = self.cross_attn_thermal(tgt + query_pos, reference_points, src_thermal, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_thermal = self.norm_thermal(tgt + self.dropout_thermal(tgt_thermal))

        # 4. Gated Fusion (自适应融合)
        # alpha shape: [Batch, Num_Query, 1]
        alpha = torch.sigmoid(self.fusion_gate(tgt)) 
        
        tgt_fused = alpha * tgt_rgb + (1 - alpha) * tgt_thermal

        # 5. FFN
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt_fused))))
        tgt = self.norm2(tgt_fused + self.dropout3(tgt2))

        return tgt


# --- Boilerplate Code (标准 Transformer 组件，无需修改但必须有) ---

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(src, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # 这里的bbox_embed通常在外面定义，这里简单处理不包含回归头

    def forward(self, tgt, reference_points, src_rgb, src_thermal, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, query_pos, reference_points, src_rgb, src_thermal, src_spatial_shapes, src_level_start_index, src_padding_mask)
            if self.return_intermediate:
                intermediate.append(output)
                
            # 注意：在标准 Deformable DETR 中，每层后会更新 reference_points
            # 这里简化逻辑，你可以后续加上 iterative bounding box refinement
            
        if self.return_intermediate:
            return torch.stack(intermediate), reference_points
        return output, reference_points

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def build_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True, # [核心修复] 必须为 True，否则输出维度会少一维
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4
    )