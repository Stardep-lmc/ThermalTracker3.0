# models/transformer.py
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from models.ops.modules import MSDeformAttn

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=True,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = RGBTTDecoderLayer(d_model, dim_feedforward,
                                          dropout, activation,
                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight)
        constant_(self.reference_points.bias, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src_rgb, masks_rgb, pos_embeds, src_thermal, query_embed=None):
        src_flatten_rgb = []
        mask_flatten_rgb = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(src_rgb, masks_rgb, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
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

        src_flatten_thermal = []
        for lvl, src in enumerate(src_thermal):
            src = src.flatten(2).transpose(1, 2)
            src_flatten_thermal.append(src)
        src_flatten_thermal = torch.cat(src_flatten_thermal, 1)

        memory_rgb = self.encoder(src_flatten_rgb, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten_rgb)
        memory_thermal = self.encoder(src_flatten_thermal, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten_rgb)

        bs, _, c = memory_rgb.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(self.level_embed), 1)

        hs, inter_references = self.decoder(tgt, reference_points, 
                                            memory_rgb, memory_thermal, 
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_embed, mask_flatten_rgb)

        return hs, inter_references

class RGBTTDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn_rgb = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_rgb = nn.Dropout(dropout)
        self.norm_rgb = nn.LayerNorm(d_model)

        self.cross_attn_thermal = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_thermal = nn.Dropout(dropout)
        self.norm_thermal = nn.LayerNorm(d_model)

        self.fusion_gate = nn.Linear(d_model, 1) 

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_points, 
                src_rgb, src_thermal, 
                src_spatial_shapes, level_start_index, src_padding_mask=None):
        
        # ... (前面的代码不变) ...

        # 2. Cross Attention RGB
        tgt_rgb = self.cross_attn_rgb(tgt + query_pos, reference_points, src_rgb, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_rgb = self.norm_rgb(tgt + self.dropout_rgb(tgt_rgb))

        # 3. Cross Attention Thermal (虽然计算了，但我们不用它)
        tgt_thermal = self.cross_attn_thermal(tgt + query_pos, reference_points, src_thermal, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_thermal = self.norm_thermal(tgt + self.dropout_thermal(tgt_thermal))

        # === [修改这里：强制只用 RGB] ===
        # 原代码：
        # alpha = torch.sigmoid(self.fusion_gate(tgt)) 
        # tgt_fused = alpha * tgt_rgb + (1 - alpha) * tgt_thermal
        
        # 修改后：
        tgt_fused = tgt_rgb 
        # ==============================

        # ... (后面的代码不变) ...
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt_fused))))
        tgt = self.norm2(tgt_fused + self.dropout3(tgt2))

        return tgt

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

    def forward(self, tgt, reference_points, src_rgb, src_thermal, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, query_pos, reference_points, src_rgb, src_thermal, src_spatial_shapes, src_level_start_index, src_padding_mask)
            if self.return_intermediate:
                intermediate.append(output)
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
        return_intermediate_dec=True,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4
    )