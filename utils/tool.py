import torch
import torch.nn as nn

# count parameters number
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)