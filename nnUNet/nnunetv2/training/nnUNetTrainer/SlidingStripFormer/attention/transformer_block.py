import torch.nn as nn

from .attention_native import SlidingWindowAttention3d, GlobalAttention3d

class StripAttention(nn.Module):
    """
    Strip注意力机制：根据stage选择使用SlidingWindowAttention3d或GlobalAttention3d
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size = None,  # None表示使用全局注意力
        input_resolution = None,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by number of heads!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.input_resolution = input_resolution
        
        if window_size is None:
            self.attention = GlobalAttention3d(
                dim=embed_dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_dropout,
                proj_drop=proj_dropout,
            )
        else:
            # print("using native attention")
            self.attention = SlidingWindowAttention3d(
                dim=embed_dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_dropout,
                proj_drop=proj_dropout,
            )

    def forward(self, x):
        B, N, C = x.shape
        D, H, W = self.input_resolution
        assert N == D * H * W, f"Input size {N} doesn't match resolution {D}x{H}x{W}"
        return self.attention(x, D, H, W)



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        assert N == D * H * W, f"Input feature has wrong number of tokens ({N}), but expected ({D}*{H}*{W})"
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = int(mlp_ratio * in_feature)
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GatedConvFFN(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        # 遵循 ConvolutionalGLU 的设计模式
        out_features = int(mlp_ratio * in_feature)
        hidden_feature = int(2 * out_features / 3)  # 减少隐藏层大小
        
        self.fc1 = nn.Linear(in_feature, hidden_feature * 2)  # 输出两倍用于门控
        self.dwconv = DWConv(dim=hidden_feature)  # 卷积处理一半特征
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(hidden_feature, in_feature)  # 投影回原始维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, D, H, W):
        x, gate = self.fc1(x).chunk(2, dim=-1)  # 分割为特征和门控
        x = self.dwconv(x, D, H, W)   # 卷积
        x = self.act_fn(x) * gate
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class StripTransformerBlockUNETR(nn.Module):
    """
    一个混合Transformer块，采用 (Norm -> Attention -> Add -> Norm -> MLP -> Add) 结构
    Attention: 使用您的 StripAttention (Token Mix)
    MLP: 使用您的 _MLP + DWConv (Channel Mix)
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            window_size,
            input_resolution,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            FFN_type: str = "GatedConvFFN",
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = StripAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            input_resolution=input_resolution,
            attn_dropout=dropout_rate,
            proj_dropout=dropout_rate,
        )
        
        self.norm2 = nn.LayerNorm(hidden_size)
        if FFN_type == "ConvFFN":
            self.mlp = ConvFFN(
                in_feature=hidden_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate
            )
        elif FFN_type == "GatedConvFFN":
            self.mlp = GatedConvFFN(
                in_feature=hidden_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate
            )
        else:
            raise ValueError(f"Invalid FFN type: {FFN_type}")

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)

        # Attention block
        x_seq = x_seq + self.attention(self.norm1(x_seq))
        
        # MLP block (Channel Mix)
        x_seq = x_seq + self.mlp(self.norm2(x_seq), D, H, W)

        # Reshape back to image format
        x = x_seq.transpose(1, 2).reshape(B, C, D, H, W)
        return x

