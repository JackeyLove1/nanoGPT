import torch
import torch.nn as nn

"""

"""
class Gamma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        # in common rms norm self.weight = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x:torch.Tensor):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1) + self.eps)
        x_norm = x * rms
        return x_norm * (self.weight + 1.0)
