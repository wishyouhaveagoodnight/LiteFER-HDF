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

# --------------------------- 局部与全局分支 ---------------------------

class LocalBranch(nn.Module):
    """可变形卷积+通道注意力分支"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 可变形卷积
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 标准化与ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 通道注意力
        self.channel_att = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.channel_att(x) * x  # 加权

class GlobalBranch(nn.Module):
    """轻量Transformer+空间注意力分支"""
    def __init__(self, in_channels, d_model=128):
        super().__init__()
        # 1x1卷积
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.bn = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU(inplace=True)
        # 轻量Transformer单元
        self.transformer = TransformerUnit(d_model)
        # 空间注意力
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.transformer(x)
        return self.spatial_att(x) * x

# --------------------------- 自适应融合模块 ---------------------------
class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 动态权重参数：用于平衡局部与全局特征
        self.global_weight = nn.Parameter(torch.tensor(0.5))
        self.local_weight = nn.Parameter(torch.tensor(0.5))
        # 深度可分离卷积，用于融合后处理
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, local, global_feat):
        """
        local: 局部分支输出，已包含通道注意力
        global_feat: 全局分支输出，已包含空间注意力
        """
        # 使用 sigmoid 将权重限制在 [0, 1]
        w_g = torch.sigmoid(self.global_weight)
        w_l = torch.sigmoid(self.local_weight)

        # 加权融合
        fused = w_g * global_feat + w_l * local

        # 后处理：卷积 + BN + ReLU
        fused = self.dw_conv(fused)
        fused = self.norm(fused)
        fused = self.act(fused)

        return fused


# --------------------------- 层级融合模块 ---------------------------
class HierarchicalFusion(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.local_branch = LocalBranch(in_channels, d_model)
        self.global_branch = GlobalBranch(in_channels, d_model)
        self.adaptive_fusion = AdaptiveFusion(d_model)

    def forward(self, x):
        local = self.local_branch(x)
        global_feat = self.global_branch(x)
        return self.adaptive_fusion(local, global_feat)
