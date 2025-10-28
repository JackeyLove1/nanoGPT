import torch
import torch.nn as nn

# count parameters number
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, seq, n_kv_heads, dim) -> (batch, seq, n_kv_heads * n_rep, dim)"""
    batch, seq, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(batch, seq, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq, n_kv_heads * n_rep, head_dim)
    )
