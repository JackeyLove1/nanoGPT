import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SwiGLU(nn.Module):
    def __init__(self, n_embed: int, n_moe_intermediate_size: int, bias:bool=False):
        super().__init__()
        self.in_proj = nn.Linear(n_embed, 2 * n_moe_intermediate_size, bias=bias)
        self.out_proj = nn.Linear(n_moe_intermediate_size, n_embed, bias=bias)
    def forward(self, x:torch.Tensor):
        u, v = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(u * F.silu(v))

class Qwen3MOEMLP(nn.Module):
    def __init__(
            self,
            n_embed: int,
            n_moe_intermediate_size: int, 
            num_experts: int,
            num_experts_per_token: int,
            bias: bool = False
    ):
        super().__init__()
        self.n_embed = n_embed
        self.n_moe_intermediate_size = n_moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        self.gate = nn.Linear(self.n_embed, self.num_experts, bias=bias)
        
        self.experts = nn.ModuleList([
            SwiGLU(self.n_embed, self.n_moe_intermediate_size,bias) for _ in range(self.num_experts)
        ])
        
    def forward(self, x : torch.Tensor):
        """x: (batch, seq, n_embed)"""
        assert len(x.shape) == 3, "x shape != 3"
        B, T, D = x.shape
        scores = self.gate(x)
        # (B, T, topk)
        topk_scores, topk_indices = torch.topk(scores, k=self.num_experts_per_token, dim=-1)
        # (B, T, topk)
        topk_probs = topk_scores.softmax(dim=-1)

        # (B * T, D)
        x_flat = x.contiguous().view(-1, D)
        # (B * T, topk)
        topk_indices_flat = topk_indices.contiguous().view(-1, self.num_experts_per_token)
        topk_probs_flat = topk_probs.contiguous().view(-1, self.num_experts_per_token)

        # (B * T, D)
        output = torch.zeros_like(x_flat)

        for expert_index in range(self.num_experts):
            # (B * T, topk)
            expert_mask = (expert_index == topk_indices_flat)
            # (B * T, topk)
            token_indices, k_indices = torch.where(expert_mask)
            if topk_indices.numel() == 0:
                continue

            expert_block = self.experts[expert_index]
            # (N, D)
            expert_input = x_flat[topk_indices]
            # (N, D)
            expert_output = expert_block(expert_input)

            # (N, 1)
            weights = topk_probs_flat[token_indices, k_indices].unsqueeze(-1)

            # (N, D)
            output[token_indices] += weights * expert_output

        return output.contiguous().view(B, T, D)
