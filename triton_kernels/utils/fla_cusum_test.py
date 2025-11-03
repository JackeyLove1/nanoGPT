import torch
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar

# 输入数据：批次大小=1，序列长度=8，头数=2
B, S, H = 1, 8, 2
input_tensor = torch.randn(B, S, H, dtype=torch.float32).cuda()

# 分块大小=4，将 8 分成 2 个块
output = chunk_local_cumsum_scalar(
    g=input_tensor,
    chunk_size=4,
    reverse=False,
    scale=None,
    head_first=False
)

# 输出形状：(2, 128, 8)
print(output.shape)  # torch.Size([2, 128, 8])
print("input: ", input_tensor)
print("output: ", output)