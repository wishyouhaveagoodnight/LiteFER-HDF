import torch
import torch.nn as nn
import math
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def _build_mobilenet_backbone(alpha):
    """构建并缩放MobileNetV2主干"""
    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
    backbone = nn.Sequential(*list(backbone.children())[:10])
    
    def scale_channels(channels):
        return max(8, int(math.ceil(channels * alpha / 8)) * 8)
    
    for name, layer in backbone.named_children():
        if isinstance(layer, nn.Conv2d):
            original_out_channels = layer.out_channels
            layer.out_channels = scale_channels(original_out_channels)
        elif hasattr(layer, 'conv'):
            for sub_layer in layer.conv:
                if isinstance(sub_layer, nn.Conv2d):
                    original_in_channels = sub_layer.in_channels
                    original_out_channels = sub_layer.out_channels
                    sub_layer.in_channels = scale_channels(original_in_channels)
                    sub_layer.out_channels = scale_channels(original_out_channels)
    return backbone

def _get_output_channels(layer):
    if hasattr(layer, 'out_channels'):
        return layer.out_channels
    elif hasattr(layer, 'conv') and isinstance(layer.conv[-1], nn.Conv2d):
        return layer.conv[-1].out_channels
    else:
        raise ValueError("无法确定层的输出通道数")
