import torch
import torch.nn as nn
from typing import Tuple, Union, List, Sequence
import numpy as np
import torch.nn.functional as F

from .dynunet_block import UnetResBlock, UnetOutBlock, get_conv_layer, get_norm_layer
from .attention.transformer_block import StripTransformerBlockUNETR


class StripUnetrPPEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: List[int],
        dims: List[int],
        depths: List[int],
        num_heads: List[int],
        window_sizes: List,
        mlp_ratios: List[float],
        dropout: float = 0.0,
        FFN_type: str = "ConvFFN",
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        # Stem - Z轴下采样2倍，H和W下采样4倍
        stem_layer = nn.Sequential(
            get_conv_layer(3, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4), dropout=dropout, conv_only=True),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        
        # 定义每个阶段的下采样策略
        downsample_configs = [
            (2, 2, 2),  # Stage 1->2: Z:96->48, H:28->14, W:28->14
            (2, 2, 2),  # Stage 2->3: Z:48->24, H:14->7, W:14->7  
            (2, 2, 2),  # Stage 3->4: Z:24->12, H:7->4, W:7->4 (修改为全维度下采样)
        ]
        
        # Downsample convs with different strategies for each stage
        for i in range(3):
            kernel_size = downsample_configs[i]
            stride = downsample_configs[i]
            downsample_layer = nn.Sequential(
                get_conv_layer(3, dims[i], dims[i+1], kernel_size=kernel_size, stride=stride, dropout=dropout, conv_only=True),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        # 计算每个阶段的分辨率
        self.input_resolutions = self._compute_resolutions(input_size)
        print("编码器各阶段分辨率:", self.input_resolutions)
        
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    StripTransformerBlockUNETR(
                        hidden_size=dims[i],
                        num_heads=num_heads[i],
                        window_size=window_sizes[i],
                        input_resolution=self.input_resolutions[i],
                        mlp_ratio=mlp_ratios[i],
                        dropout_rate=dropout,
                        FFN_type=FFN_type,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))

    def _compute_resolutions(self, input_size):
        """
        计算各阶段的分辨率
        输入: [192, 112, 112]
        期望输出: 
        Stage 1: [96, 28, 28]   (stem: /2, /4, /4)
        Stage 2: [48, 14, 14]   (/2, /2, /2)
        Stage 3: [24, 7, 7]     (/2, /2, /2)
        Stage 4: [12, 4, 4]     (/2, /2, /2)
        """
        resolutions = []
        
        # Stage 1 (stem)
        d = input_size[0] // 2   # 192 -> 96
        h = input_size[1] // 4   # 112 -> 28
        w = input_size[2] // 4   # 112 -> 28
        current_size = [d, h, w]
        resolutions.append(current_size)
        
        # Stage 2
        current_size = [current_size[0] // 2, current_size[1] // 2, current_size[2] // 2]  # [48, 14, 14]
        resolutions.append(current_size)
        
        # Stage 3
        current_size = [current_size[0] // 2, current_size[1] // 2, current_size[2] // 2]  # [24, 7, 7]
        resolutions.append(current_size)
        
        # Stage 4 - 全维度下采样
        current_size = [current_size[0] // 2, current_size[1] // 2, current_size[2] // 2]  # [12, 4, 4]
        resolutions.append(current_size)
        
        return resolutions

    def forward(self, x):
        hidden_states = []
        # Stage 1
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        hidden_states.append(x)
        
        # Stages 2, 3, 4
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            hidden_states.append(x)
        return hidden_states

class StripUnetrUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_heads: int,
            input_resolution: List[int],
            window_size,
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str] = "instance",
            depth: int = 1, # 使用1层ConvLSTM就足够了
            mlp_ratio: float = 4.0,
            conv_decoder: bool = False,
            FFN_type: str = "GatedConvFFN",
    ) -> None:
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size, conv_only=True, is_transposed=True,
        )

        self.decoder_block = nn.ModuleList()
        if conv_decoder:
            # 将UnetResBlock替换为ConvLSTM
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1, norm_name=norm_name)
            )
        else:
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    StripTransformerBlockUNETR(
                        hidden_size=out_channels,
                        num_heads=num_heads,
                        window_size=window_size,
                        input_resolution=input_resolution,
                        mlp_ratio=mlp_ratio,
                        FFN_type=FFN_type,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        # Spatially resize `out` to match `skip` in case of dimension mismatch from strided convolutions
        if out.shape[2:] != skip.shape[2:]:
            out = F.interpolate(out, size=skip.shape[2:], mode="trilinear", align_corners=False)
        out = out + skip
        out = self.decoder_block[0](out)
        return out

class StripFormer3D_UNETR_PP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            input_size: List[int],
            window_sizes: list = [[21, 3, 3], [11, 3, 3], [7, 3, 3], None],
            dims: list = [32, 64, 128, 256],
            depths: list = [2, 2, 2, 2],
            num_heads: list = [2, 4, 8, 16],
            mlp_ratios: list = [4, 4, 4, 4],
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            do_ds: bool = True,
            FFN_type: str = "GatedConvFFN",
    ) -> None:
        super().__init__()
        self.do_ds = do_ds
        
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=dims[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        
        # --- Encoder ---
        self.encoder = StripUnetrPPEncoder(
            in_channels=in_channels,
            input_size=input_size,
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            window_sizes=window_sizes,
            mlp_ratios=mlp_ratios,
            dropout=dropout_rate,
            FFN_type=FFN_type,
        )
        
        # --- Decoder ---
        encoder_resolutions = self.encoder.input_resolutions
        
        # 定义解码器的上采样策略，与编码器的下采样相对应
        self.decoder4 = StripUnetrUpBlock(
            spatial_dims=3, in_channels=dims[3], out_channels=dims[2],
            num_heads=num_heads[2], upsample_kernel_size=(2, 2, 2),  # 全维度上采样: 12->24, 4->7, 4->7
            input_resolution=encoder_resolutions[2], window_size=window_sizes[2], norm_name=norm_name,
            mlp_ratio=mlp_ratios[2], conv_decoder=True,
        )
        self.decoder3 = StripUnetrUpBlock(
            spatial_dims=3, in_channels=dims[2], out_channels=dims[1],
            num_heads=num_heads[1], upsample_kernel_size=(2, 2, 2),  # 24->48, 7->14, 7->14
            input_resolution=encoder_resolutions[1], window_size=window_sizes[1], norm_name=norm_name,
            mlp_ratio=mlp_ratios[1], conv_decoder=True,
        )
        self.decoder2 = StripUnetrUpBlock(
            spatial_dims=3, in_channels=dims[1], out_channels=dims[0],
            num_heads=num_heads[0], upsample_kernel_size=(2, 2, 2),  # 48->96, 14->28, 14->28
            input_resolution=encoder_resolutions[0], window_size=window_sizes[0], norm_name=norm_name,
            mlp_ratio=mlp_ratios[0], conv_decoder=True,
        )
        self.decoder1 = StripUnetrUpBlock(
            spatial_dims=3, in_channels=dims[0], out_channels=dims[0], # Final block has same channels
            num_heads=num_heads[0], upsample_kernel_size=(2, 4, 4),  # 96->192, 28->112, 28->112
            input_resolution=input_size, window_size=window_sizes[0],
            conv_decoder=True, norm_name=norm_name,
        )

        # --- Output heads ---
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=dims[1], out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=dims[2], out_channels=out_channels)

    def forward(self, x_in):
        # Generate high-resolution skip connection from input
        convBlock = self.encoder1(x_in)

        # Encoder forward pass
        hidden_states = self.encoder(x_in)
        
        enc1, enc2, enc3, enc4 = hidden_states

        # Decoder forward pass with skip connections
        dec3 = self.decoder4(enc4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        
        # The final decoder block uses the high-resolution skip connection from convBlock
        out = self.decoder1(dec1, convBlock)
        
        if self.do_ds:
            # Deep supervision outputs are taken from the outputs of decoder stages
            logits = [self.out1(out), self.out2(dec2), self.out3(dec3)]
        else:
            logits = self.out1(out)

        return logits


if __name__ == '__main__':
    # A simple test case
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ptflops import get_model_complexity_info

    model = StripFormer3D_UNETR_PP(
        in_channels=1,
        out_channels=3,
        input_size=[192, 128, 128],
        window_sizes=[[3, 3, 3], [17, 3, 3], [21, 3, 3], None],  # 适应新分辨率的窗口大小
        dims=[32, 64, 128, 256],
        depths=[3, 3, 5, 2],
        num_heads=[4, 4, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        do_ds=False,
    ).to(device)

    print("Model created successfully.")
    macs, params = get_model_complexity_info(model, (1, 192, 128, 128), as_strings=True, print_per_layer_stat=False)
    print(f"MACs: {macs}")
    print(f"Params: {params}")

    input_tensor = torch.randn(1, 1, 192, 128, 128).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    if isinstance(output, list):
        for i, o in enumerate(output):
            print(f"Output {i} shape: {o.shape}")
    else:
        print(f"Output shape: {output.shape}")

    print("Test passed!") 