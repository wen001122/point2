# ptv3_model.py (简化版 PointTransformerV3)
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import segment_csr

# ----------------------------- 简化版 Point 对象 -----------------------------
class Point:
    def __init__(self, data):
        self.coord = data["coord"]          # (N, 3)
        self.feat = data["feat"]            # (N, 6)
        self.label = data.get("label", None)
        self.batch = data["batch"]          # (N,)
        self.grid_size = data["grid_size"]  # float
        self.grid_coord = (self.coord / self.grid_size).floor().int()
        self.offset = torch.cumsum(torch.bincount(self.batch.cpu()), dim=0).long()
        self.sparse_conv_feat = None

    def sparsify(self):
        indices = torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1)
        spatial_shape = self.grid_coord.max(dim=0).values + 1
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=indices,
            spatial_shape=spatial_shape.tolist(),
            batch_size=int(self.batch.max().item()) + 1
        )
        return self

# ----------------------------- Attention 模块 -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, 3, H, C // H).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (H, B, C//H)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(out)

# ----------------------------- Transformer Block -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------------------- 主模型 -----------------------------
class PointTransformerV3(nn.Module):
    def __init__(self, input_dim=6, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)

        self.encoder = nn.Sequential(
            TransformerBlock(64, num_heads=4),
            TransformerBlock(64, num_heads=4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, data_dict):
        point = Point(data_dict).sparsify()
        feat = point.sparse_conv_feat.features  # (N, C)
        x = self.embedding(feat)                # (N, 64)
        x = self.encoder(x)                     # (N, 64)
        out = self.classifier(x)                # (N, num_classes)
        return out
