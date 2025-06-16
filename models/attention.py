import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(8, channels//reduction)),
            nn.ReLU(),
            nn.Linear(max(8, channels//reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        avg = self.avg_pool(x).view(b, c)
        max_val = self.max_pool(x).view(b, c)
        weight = self.fc(avg + max_val).view(b, c, 1, 1)
        return x * weight

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))
