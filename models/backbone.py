# models/backbone.py
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from .position_encoding import build_position_encoding

# 定义一个辅助类，用于包装 Tensor 和 Mask
class NestedTensor(object):
    def __init__(self, tensors, mask: torch.Tensor):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    Transformer 训练时通常冻结 Backbone 的 BatchNorm
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # 提取 C3, C4, C5 层 (对应 strides 8, 16, 32)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = 512 if name == 'resnet18' else 2048 # ResNet50 C5 channel is 2048

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将 mask 插值到特征图大小
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class DualBackbone(nn.Module):
    """
    [创新点] 双流骨干网络：同时处理 RGB 和 Thermal
    """
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        super().__init__()
        
        # --- 构建 RGB Backbone ---
        backbone_rgb = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        
        # --- 构建 Thermal Backbone ---
        # 同样使用 ImageNet 预训练权重，虽然领域不同，但纹理特征有共通性
        backbone_thermal = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
            
        # 提取层
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        
        self.body_rgb = IntermediateLayerGetter(backbone_rgb, return_layers=return_layers)
        self.body_thermal = IntermediateLayerGetter(backbone_thermal, return_layers=return_layers)
        
        # 设置通道数 (ResNet50: layer2=512, layer3=1024, layer4=2048)
        self.num_channels = [512, 1024, 2048] 

    def forward(self, tensor_list_rgb: NestedTensor, tensor_list_thermal: NestedTensor):
        # 1. RGB 前向传播
        xs_rgb = self.body_rgb(tensor_list_rgb.tensors)
        
        # 2. Thermal 前向传播
        # Thermal 输入是 [B, 1, H, W]，但 ResNet 需要 [B, 3, H, W]
        # 简单的做法是 repeat
        t_img = tensor_list_thermal.tensors
        if t_img.shape[1] == 1:
            t_img = t_img.repeat(1, 3, 1, 1)
        xs_thermal = self.body_thermal(t_img)
        
        out_rgb: Dict[str, NestedTensor] = {}
        out_thermal: Dict[str, NestedTensor] = {}
        
        # 3. 处理 Mask (RGB 和 Thermal 空间对齐，所以 Mask 是一样的，用 RGB 的即可)
        m = tensor_list_rgb.mask
        assert m is not None
        
        for name, x in xs_rgb.items():
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out_rgb[name] = NestedTensor(x, mask)
            
            # 对应的 Thermal 特征
            x_t = xs_thermal[name]
            out_thermal[name] = NestedTensor(x_t, mask)
            
        return out_rgb, out_thermal

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list_rgb: NestedTensor, tensor_list_thermal: NestedTensor):
        # 调用 DualBackbone
        xs_rgb, xs_thermal = self[0](tensor_list_rgb, tensor_list_thermal)
        
        out_rgb = []
        out_thermal = []
        pos = []
        
        # 遍历每一层 (C3, C4, C5)
        for name, x in xs_rgb.items():
            out_rgb.append(x)
            out_thermal.append(xs_thermal[name])
            
            # 生成位置编码 (共享)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out_rgb, out_thermal, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = True # MOTR/Deformable DETR 需要多尺度特征
    
    # 实例化 DualBackbone
    backbone = DualBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    
    # 组合 Backbone 和 Position Encoding
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model