import torch
import torch.nn as nn
import torch.nn.functional as F

# SwiGLU
class Qwen3DenseMLP(nn.Module):
    def __init__(
            self,
            n_embd: int,
            n_mlp: int
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_mlp = n_mlp
        # TODO: merge gate operation and up operation to an operation
        self.gate = nn.Linear(self.n_embd, self.n_mlp)
        self.up = nn.Linear(self.n_embd, self.n_mlp)
        self.down = nn.Linear(self.n_mlp, self.n_embd)

    def forward(self, x: torch.Tensor):
        h = self.up(x) * F.silu(self.gate(x))
        return self.down(h)