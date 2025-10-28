"""Gaussian Error Linear Unit"""
import torch
import torch.nn as nn
from torch import Tensor
import math

def gelu_extract(x: Tensor) -> Tensor:
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh((
        math.sqrt(2.0/ math.pi) * (x + 0.044715 * (x ** 3))
    )))

def gelu_torch(x: Tensor) -> Tensor:
    return nn.GELU()(x)

class GELU_FFN(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            dropout: float = 0.0,
            bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        self.up_proj = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.activate = nn.GELU()
        self.down_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.up_proj(x)
        h = self.activate(h)
        h = self.dropout(h)
        return self.down_proj(h)
