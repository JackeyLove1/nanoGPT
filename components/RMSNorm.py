import torch
import torch.nn as nn

'''RMS Norm torch version, more light than LayerNorm'''
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_f32 = x.to(torch.float32)
        """(batch_size, seq_len, dim) * (b, s, 1) = (b, s, d)"""
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        """(b, s, d) * (dim) -> (b, s, d)"""
        return (x_f32 * rms * self.weight).type_as(x)


class RMSNormTorh(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.rms_norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor):
        return self.rms_norm(x)
