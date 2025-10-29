'''
QK Norm Attention
References:
    1. https://ar5iv.labs.arxiv.org/html/2010.04245
    2. https://arxiv.org/abs/2505.09388
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# class RMSNorm(nn.Module):
#     """norm in the last dimension"""
#     def __init__(self, dim: int, eps:float=1e-6):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
#
#     def forward(self, x: torch.Tensor):
#         inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#         return self.weight * x * inv_rms


class QKNormMHA(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            max_seq_len: int = 4096,
            dropout: float = 0.0,
            bias:bool = False
    ):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd is not the multiple of n_heads"
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = self.n_embd // self.n_heads

        # q,k,v do once
        self.in_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.scale = self.head_dim ** -0.5

        mask = torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)


    def forward(self, x:torch.Tensor):
        """x: (batch, seq, Dim)"""
        assert len(x.shape) == 3, "x.shape is not [B, S, D]"
        assert x.shape[-1] == self.n_embd, f"x embedding size:{x.shape[-1]} not equal to {self.n_embd}"
        B, S, D = x.shape

        # (B, S, 3 * n_embd)
        x: torch.Tensor = self.in_proj(x)
        # (B, S, hidden_dim)
        q, k, v = x.chunk(3, dim=-1)
        # (B, n_head, S, head_dim)
        q = q.contiguous().view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_head, S, head_dim)
        q_norm: torch.Tensor = self.q_norm(q)
        k_norm: torch.Tensor = self.k_norm(k)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            scores = F.scaled_dot_product_attention(
                q_norm, k_norm, v, attn_mask=None, is_causal=True
            )
        else:
            # (B, n_head, S, S)
            scores = torch.matmul(q_norm, k_norm.transpose(-1, -2)) * self.scale
            scores = F.softmax(scores + self.mask[:,:,:S,:S] , dim=-1)
            scores = self.attn_drop(scores)
            # (B, n_head, S, head_dim)
            scores = scores @ v
        # (B, S, n_head, head_dim)
        out = scores.transpose(1, 2)
        # (B, S, D)
        out = out.contiguous().view(B, S, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, D, H = 2, 16, 64, 8
    x = torch.randn(B, S, D)
    attention = QKNormMHA(D, H)
    y = attention(x)
    print(y.shape)
    assert x.shape == y.shape