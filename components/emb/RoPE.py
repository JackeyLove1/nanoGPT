"""Rotary Position Embedding"""
import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_len: int,
            base: int = 10000,
            device:torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert dim % 2 == 0, "dimension must be multiple of 2"
        self.dim = dim
        self.max_seq_len = max_seq_len
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dtype is None:
            dtype = torch.float32

        rope_cache = self.build_rope_cache(dim, max_seq_len, base, device, dtype)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    @torch.no_grad()
    def build_rope_cache(
            self,
            dim: int,
            max_seq_len: int,
            base: int,
            device: torch.device,
            dtype: torch.dtype
    ) -> torch.Tensor:
        # (dim // 2) - 计算频率: theta_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        # (seq)
        t = torch.arange(0, max_seq_len, device=device, dtype=dtype)
        # (seq, dim // 2)
        freq = torch.einsum("i,j->ij", t, inv_freq)
        cos, sin = torch.cos(freq), torch.sin(freq)
        # (seq, dim // 2, 2)
        rope_cache = torch.stack([cos, sin], dim=-1)
        return rope_cache

    def forward(self, x: torch.Tensor):
        """x: (batch, seq, n_head, dim)"""
        # (seq, dim // 2)
        cos, sin = self.rope_cache[..., 0], self.rope_cache[..., 1]
        B, T, H, D = x.shape
        cos = cos[:T].to(device=x.device, dtype=x.dtype)
        sin = sin[:T].to(device=x.device, dtype=x.dtype)
        # (1, seq, 1, dim // 2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # (batch, seq, n_head, dim // 2, 2)
        x_reshaped = x.contiguous().reshape(B, T, H, self.dim // 2, 2)
        # (batch, seq, n_head, dim // 2)
        x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]

        # (batch, seq, n_head, dim // 2)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        x_out = torch.stack([out_real, out_imag], dim=-1)

        # (batch, seq, n_head, dim)
        x_out = x_out.reshape(B, T, H, D)
        return x_out

