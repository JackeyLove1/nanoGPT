"""
DeepSeek MLA Implement(using decoupled rope)
References:
    1. https://arxiv.org/html/2405.04434
"""

"""
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      spda_o shape [Sq, N, V]
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return spda_o @ W_O

deepseek v2
rope_dim = 64
head_dim = 128
latent_dim = 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def apply_rope(x: Tensor):
    #TODO
    return x

class MultiHeadLatentAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_latent: int = 512,
            n_rope: int = 64,
            max_seq_len: int = 4096,
            bias: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model % n_heads != 0"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_rope = n_rope
        self.d_latent = d_latent
        self.scale = self.head_dim ** -0.5

        self.W_DQ = nn.Linear(self.head_dim, self.d_latent, bias=bias)
        self.W_UQ = nn.Linear(self.d_latent, self.head_dim, bias=bias)
        self.W_DKV = nn.Linear(self.head_dim, 2 * self.d_latent, bias=bias)
        self.W_UKV = nn.Linear(2 * self.d_latent, 2 * self.head_dim, bias=bias)
        self.W_QR = nn.Linear(self.head_dim, self.n_rope, bias=bias)
        self.W_KR = nn.Linear(self.head_dim, self.n_rope, bias=bias)
        self.W_O = nn.Linear(self.head_dim, self.head_dim, bias=bias)

        self.Q_Norm = nn.RMSNorm(self.head_dim)
        self.K_Norm = nn.RMSNorm(self.head_dim)

        mask = torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x:torch.Tensor):
        """x : (batch_size, seq_len, d_model)"""
        assert len(x.shape) == 3, "x shape is not [B, T, D]"
        assert x.shape[-1] == self.d_model, "x.shape[-1] != d_model "
        B, T, D = x.shape
        # (B, T, H, D)
        x = x.contiguous().view(B, T, self.n_heads, self.head_dim)
        # (B, H, T, D)
        x = x.transpose(1, 2)

        # (B, H, T, d_latent)
        c_q = self.W_DQ(x)
        c_kv: Tensor  = self.W_DKV(x)

        # (B, H, T, head_dim)
        q_c = self.W_UQ(c_q)
        q_c_norm = self.Q_Norm(q_c)
        # (B, H, T, n_rope)
        q_r = apply_rope(self.W_QR(q_c_norm))
        # (B, H, T, head_dim + n_rope)
        q = torch.concatenate([q_c_norm, q_r], dim=-1)


        # (B, H, T, head_dim)
        k_c, v_c = self.W_UKV(c_kv).chunk(2, dim=-1)
        k_c_norm = self.K_Norm(k_c)
        # (B, H, T, n_rope)
        k_r = apply_rope(self.W_KR(x))
        # (B, H, T, d_latent + n_rope)
        k = torch.concatenate([k_c_norm, k_r], dim=-1)

        #TODO: use flash attention
        #(B, H, T, T)
        scores = q @ k.transpose(-1,-2) * self.scale
        # (B, H, T, T)
        scores = F.softmax(scores + self.mask[:, :, :T, :T], dim=-1)
        # (B, H, T, head_dim)
        scores = scores @ v_c
        # (B, H, T, head_dim)
        out: Tensor = self.W_O(scores)
        # (B, T, H, head_dim)
        out = out.transpose(1, 2)
        return out.flatten(2) # (B, T, D)

if __name__ == "__main__":
    B, T, D, H = 2, 1024, 4096, 32
    mla = MultiHeadLatentAttention(D, H)
    x = torch.randn(B, T, D)
    y = mla(x)
    print(y.shape)
    assert x.shape == y.shape