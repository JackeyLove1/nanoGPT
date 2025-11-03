import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x:torch.Tensor, gate: Optional[torch.Tensor] = None):
        hidden_state = x * torch.rsqrt(x.pow(2).mean(dim=-1) + self.eps)
        hidden_state = self.weight * hidden_state
        hidden_state = hidden_state * F.silu(gate.to(torch.float32))
        return hidden_state