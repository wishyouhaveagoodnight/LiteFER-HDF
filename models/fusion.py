import torch
import torch.nn as nn
import math
from torchvision.ops import DeformConv2d as TVDeformConv2d
# --------------------------- 辅助模块：位置编码 ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        assert D == self.pe.size(2), "特征维度不匹配"
        pe = self.pe[:, :L, :]
        return x + pe.expand(B, -1, -1)

# --------------------------- 可变形卷积模块 ---------------------------
class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.offset_channels,
            kernel_size=3,
            padding=padding,
            bias=True
        )
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

        self.dcn = TVDeformConv2d(
            in_channels, out_channels, kernel_size, padding=padding
        )

    def forward(self, x):
        offset = self.offset_conv(x)  # [B, 18, H, W]
        return self.dcn(x, offset)

# --------------------------- Transformer 单元 ---------------------------
class TransformerUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.pos_enc = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        is_image = (x.dim() == 4)
        if is_image:
            B, C, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        x = self.pos_enc(x)

        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))

        if is_image:
            x = x.permute(0, 2, 1).view(B, C, H, W)

        return x

# --------------------------- 层级融合模块 ---------------------------

# --------------------------- 新增模块 ---------------------------
class AdaptiveFusion(nn.Module):
     def __init__(self, channels):
         super().__init__()
         self.channel_att = ChannelAttention(channels)
         self.spatial_att = SpatialAttention()
         self.global_weight = nn.Parameter(torch.tensor(0.5))
         self.local_weight = nn.Parameter(torch.tensor(0.5))
         self.dw_conv = nn.Conv2d(channels, channels, 3, 
                               padding=1, groups=channels)
 
     def forward(self, local, global_feat):
         # 通道注意力增强局部特征
         local_att = self.channel_att(local) * local
         # 空间注意力增强全局特征
         global_att = self.spatial_att(global_feat) * global_feat
         # 动态权重融合
         w_g = torch.sigmoid(self.global_weight)
         w_l = torch.sigmoid(self.local_weight)
         fused = self.dw_conv(w_g*global_att + w_l*local_att)
         return fused

# --------------------------- 修改层级融合模块 ---------------------------
class HierarchicalFusion(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.local_branch = nn.Sequential(
            DeformConv2d(in_channels, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(in_channels, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            TransformerUnit(d_model)
        )

        self.adaptive_fusion = AdaptiveFusion(d_model)

    def forward(self, x):
        local = self.local_branch(x)
        global_feat = self.global_branch(x)

        return self.adaptive_fusion(local, global_feat)
