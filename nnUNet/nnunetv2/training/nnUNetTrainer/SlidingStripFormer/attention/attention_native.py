from .unfold_operate import Unfold3d, unfold3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    # 确保window_size是元组格式
    if isinstance(window_size, int):
        window_size = (window_size, window_size, window_size)
    
    attn_map = unfold3d(
        torch.ones([1, 1, input_resolution[0], input_resolution[1], input_resolution[2]]),
        kernel_size=window_size,
        dilation=1,
        padding=tuple(ws // 2 for ws in window_size),
        stride=1
    )
    attn_local_length = attn_map.sum(1).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

class SlidingWindowAttention3d(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 处理window_size，支持int和tuple两种输入格式
        # 重要：window_size的维度顺序是 (D, H, W)，对应深度、高度、宽度
        # 例如：
        # - window_size = 3 -> (3, 3, 3) 立方体窗口
        # - window_size = (11, 3, 3) -> 深度方向长条状窗口
        # - window_size = (3, 11, 3) -> 高度方向长条状窗口  
        # - window_size = (3, 3, 11) -> 宽度方向长条状窗口
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        elif isinstance(window_size, (list, tuple)):
            if len(window_size) == 1:
                window_size = (window_size[0], window_size[0], window_size[0])
            elif len(window_size) == 2:
                window_size = (window_size[0], window_size[1], window_size[1])
            elif len(window_size) == 3:
                window_size = tuple(window_size)
            else:
                raise ValueError(f"window_size should have 1, 2, or 3 elements, got {len(window_size)}")
        else:
            raise ValueError(f"window_size should be int or tuple/list, got {type(window_size)}")
            
        # 确保所有维度的窗口大小都是奇数
        assert all(ws % 2 == 1 for ws in window_size), f"All window sizes must be odd, got {window_size}"

        self.window_size = window_size
        self.local_len = np.prod(window_size)  # wd * wh * ww

        self.unfold = Unfold3d(kernel_size=window_size, padding=tuple(ws // 2 for ws in window_size), stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # Generate padding_mask && sequence length scale
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy())),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        assert N == D * H * W, f"Input tensor size {N} does not match D*H*W ({D}*{H}*{W})"

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale
        
        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, D, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))

        # Calculate attention weights
        attn = attn_local.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values
        x = (attn.unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class GlobalAttention3d(nn.Module):
    """
    全局3D注意力机制，适用于最后stage的小体块
    保持与SlidingWindowAttention3d的设计一致性
    """
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.input_resolution = input_resolution
        
        # 计算总的空间位置数
        self.total_positions = np.prod(input_resolution)  # D * H * W
        
        # 可学习温度参数（与原始代码一致）
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))
        
        # 查询投影和查询嵌入（与原始代码一致）
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        
        # KV投影
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 简单的全局位置偏置
        self.global_pos_bias = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.total_positions), mean=0, std=0.0004))

        # 生成序列长度缩放（与原始代码保持一致）
        # 对于全局注意力，所有位置的序列长度都是总位置数
        seq_length = torch.full((self.total_positions, 1), self.total_positions, dtype=torch.float32)
        self.register_buffer("seq_length_scale", torch.log(seq_length), persistent=False)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        assert N == D * H * W, f"Input tensor size {N} does not match D*H*W ({D}*{H}*{W})"
        assert N == self.total_positions, f"Input size {N} does not match expected {self.total_positions}"

        # 生成查询，归一化，添加查询嵌入，并用温度和序列长度缩放（与原始代码一致）
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        # 生成键和值，对键进行L2归一化
        k_global, v_global = self.kv(x).chunk(2, dim=-1)
        k_global = F.normalize(k_global.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        v_global = v_global.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算全局相似度
        attn_global = (q_norm_scaled @ k_global.transpose(-2, -1)) + self.global_pos_bias.unsqueeze(1)

        # 计算注意力权重
        attn = attn_global.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 聚合值
        x = (attn @ v_global).transpose(1, 2).reshape(B, N, C)

        # 线性投影和输出
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}'


if __name__ == "__main__":
    import time
    
    # 测试SlidingWindowAttention3d
    print("Testing SlidingWindowAttention3d...")
    D, H, W = 48, 28, 28
    input_tensor = torch.randn(2, D * H * W, 128).cuda()
    window_size = [11, 3, 3]
    
    time_start = time.time()
    model = SlidingWindowAttention3d(
        dim=128, 
        input_resolution=[D, H, W],
        num_heads=8, 
        window_size=window_size
    ).cuda()
    output = model(input_tensor, D, H, W)
    print(f"SlidingWindow output shape: {output.shape}")
    time_end = time.time()
    print(f"SlidingWindow time: {time_end - time_start:.4f} seconds")
    
    # 测试GlobalAttention3d（使用更小的分辨率）
    print("\nTesting GlobalAttention3d...")
    D_small, H_small, W_small = 4, 4, 4  # 小分辨率适合全局注意力
    input_tensor_small = torch.randn(2, D_small * H_small * W_small, 128).cuda()
    
    time_start = time.time()
    global_model = GlobalAttention3d(
        dim=128,
        input_resolution=[D_small, H_small, W_small],
        num_heads=8
    ).cuda()
    output_global = global_model(input_tensor_small, D_small, H_small, W_small)
    print(f"Global output shape: {output_global.shape}")
    time_end = time.time()
    print(f"Global time: {time_end - time_start:.4f} seconds")
    