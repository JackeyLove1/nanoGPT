"""Sinusoidal Positional Encoding"""
import torch
import math
import torch.nn as nn

def sinusoidal_positional_encoding(seq_len: int,
                                   d_model: int,
                                   base: float = 10000.0,
                                   device=None,
                                   dtype=torch.float32) -> torch.Tensor:
    """
    返回形状 [seq_len, d_model] 的正弦位置编码矩阵（不含 batch 维）。
    与 Vaswani et al. (2017) 一致：偶数维用 sin，奇数维用 cos。
    """
    device = device or torch.device('cpu')
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)          # [seq_len, 1]
    i = torch.arange(d_model // 2, device=device, dtype=dtype).unsqueeze(0)       # [1, d_model//2]

    # 计算频率：1 / base^(2i/d_model)
    div_term = torch.exp(-(2 * i) * math.log(base) / d_model)                     # [1, d_model//2]
    angles = pos * div_term                                                       # [seq_len, d_model//2]

    pe = torch.empty(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles)                                               # 偶数维
    pe[:, 1::2] = torch.cos(angles)                                               # 奇数维

    if d_model % 2 == 1:
        # 若 d_model 为奇数，最后一维再补一个 sin 频率（常见做法之一）
        extra = torch.sin(pos * torch.exp(-(2 * (d_model//2)) * math.log(base) / d_model))
        pe[:, -1] = extra.squeeze(1)

    return pe


class SPE(nn.Module):
    """
    标准 Transformer 风格的位置编码层：
    - 构造时预生成最大长度的 PE，并以 buffer 形式存放（不参与梯度）
    - 前向时切片到需要的长度并相加，再做一次 dropout
    - 支持 offset（用于分批缓存解码）
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000, base: float = 10000.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = sinusoidal_positional_encoding(max_len, d_model, base=base)          # [max_len, d_model]
        pe = pe.unsqueeze(0)                                                      # [1, max_len, d_model]
        self.register_buffer('pe', pe, persistent=False)  # 不保存梯度/优化器状态

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        x: [batch, seq_len, d_model]
        offset: 位置起点（解码时可传入累计长度）
        """
        seq_len = x.size(1)
        x = x + self.pe[:, offset:offset + seq_len, :]
        return self.dropout(x)