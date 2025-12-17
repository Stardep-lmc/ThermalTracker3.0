# models/backbone.py
# ------------------------------------------------------------------------
# 论文创新版：Siamese + Residual Thermal Adapter
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from models.position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict: del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

# [创新点] 残差适配器
class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 先压缩特征，再放大，类似 Bottleneck 结构，能提取更深层语义
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # 这里可以用普通 BN，因为 BatchSize=4 勉强够用，或者用 GroupNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
        # 初始化为 0，保证初始状态下 I_out = Repeat(I_in)，不破坏原始信息
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        # x: [B, 1, H, W]
        identity = x.repeat(1, 3, 1, 1) # 基础路径: 直接复制为3通道
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # 修正路径
        
        return identity + out # 残差连接

class Backbone(nn.Module):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        super().__init__()
        
        norm_layer = FrozenBatchNorm2d
        backbone = resnet50(replace_stride_with_dilation=[False, False, dilation],
                            pretrained=is_main_process(), norm_layer=norm_layer)
        
        # 使用新的残差适配器
        self.thermal_adapter = ResidualAdapter()

        if name in ('resnet18', 'resnet34'):
            self.num_channels = [128, 256, 512]
        else:
            self.num_channels = [512, 1024, 2048]
        
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor, mode='rgb'):
        x = tensor_list.tensors
        
        if mode == 'thermal':
            # 确保单通道
            if x.shape[1] == 3: x = x.mean(dim=1, keepdim=True)
            # 过残差适配器
            x = self.thermal_adapter(x)
        
        # RGB 或 适配后的 Thermal 进入共享骨干
        xs = self.body(x)
        
        out: Dict[str, NestedTensor] = {}
        for name, val in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=val.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(val, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, tensor_list_thermal: NestedTensor):
        xs_rgb = self[0](tensor_list, mode='rgb')
        xs_thermal = self[0](tensor_list_thermal, mode='thermal')
        
        out_rgb: List[NestedTensor] = []
        out_thermal: List[NestedTensor] = []
        pos = []
        
        for name, x in xs_rgb.items():
            out_rgb.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
            
        for name, x in xs_thermal.items():
            out_thermal.append(x)

        return out_rgb, out_thermal, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks if hasattr(args, 'masks') else False
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model