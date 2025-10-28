""""""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SwiGLU_FFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: Optional[int] = None,
            ffn_dim_multiplier: Optional[int] = 4,
            multiple_of: Optional[int] = 64,
            dropout: float = 0.0,
            bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model

        self.d_ff = d_ff
        if self.d_ff is None:
            base = int(2 * d_model / 3)
            if ffn_dim_multiplier is not None:
                base = int(base * ffn_dim_multiplier)
            self.d_ff = base
        if multiple_of is not None and multiple_of > 0:
            self.d_ff = multiple_of * ((self.d_ff + multiple_of - 1) // multiple_of)

        # merge up_proj and gate
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_ff, bias=bias)
        self.out_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        u, v = self.in_proj(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.dropout(h)
        return self.out_proj(h)

