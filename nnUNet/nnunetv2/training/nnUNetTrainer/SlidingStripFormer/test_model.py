import torch
from nnunetv2.training.nnUNetTrainer.SlidingStripFormer.StripFormer3d import StripFormer3D_UNETR_PP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = (96, 96, 96)
input_tensor = torch.randn(1, 1, *input_size).to(device)

model = StripFormer3D_UNETR_PP(
    in_channels=1,
    out_channels=3,
    input_size=input_size,
    window_sizes=[[3, 3, 3], [17, 3, 3], [21, 3, 3], None],
    dims=[32, 64, 128, 256],
    depths=[3, 3, 5, 2],
    num_heads=[4, 4, 8, 8],
    mlp_ratios=[4, 4, 4, 4],
    do_ds=False,
).to(device)

model.eval()

print("=== FLOPs & Params Comparison (96³ input) ===\n")

# ------------------- 1. fvcore (最准确，医学影像领域标准) -------------------
from fvcore.nn import FlopCountAnalysis
flops_fvcore = FlopCountAnalysis(model, input_tensor).total()
params = sum(p.numel() for p in model.parameters())
print(f"fvcore      : {flops_fvcore/1e9:.3f} G FLOPs   |  {params/1e6:.3f} M Params")

# ------------------- 2. ptflops (你原来用的，通常偏低) -------------------
try:
    from ptflops import get_model_complexity_info
    macs, params2 = get_model_complexity_info(model, (1, *input_size), as_strings=False, print_per_layer_stat=False)
    print(f"ptflops     : {macs/1e9:.3f} G FLOPs   |  {params2/1e6:.3f} M Params")
except Exception as e:
    print(f"ptflops     : Failed → {e}")


# ------------------- 前向验证 -------------------
with torch.no_grad():
    _ = model(input_tensor)
print("\nForward pass successful!")