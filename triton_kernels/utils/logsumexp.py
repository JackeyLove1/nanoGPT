import torch
import numpy as np

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from jaxtyping import Float

exp = tldevice.exp
log = tldevice.log

@triton.heuristics({
    "HAS_SCALE": lambda args: args["scale"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_wraps)
        for num_wraps in [1,2,4,8,16,32]
    ],
    key=["D"],
    cache_results=True
)
@triton.jit
def logsumexp_fwd_kernel(
    input_ptr, # (B, D)
    output_ptr,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    pid_n, pid_d = tl.program_id(0), tl.program_id(1)
    row_offs = pid_n * D
    col_offs = pid_d * B + tl.arange(0, B)
    mask = col_offs < D

    block_input = tl.load(input_ptr + row_offs + col_offs, mask=mask, other=-float('inf'))
    if HAS_SCALE:
        block_input *= scale

    block_max = tl.max(block_input, axis=0)
    block_exp = exp(block_input - block_max)
    block_sum = tl.sum(block_exp, axis=0)
    block_output = log(block_sum) + block_max

    tl.store(output_ptr + pid_n * tl.cdiv(D, B) + pid_d, block_output, mask=mask)

def logsumexp_fwd(
    x: torch.Tensor,
    scale: float | None = None,
    dtype: torch.dtype | None = None,
):
    r"""
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """
    shape = x.shape
    x = x.view(-1, x.shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        input_ptr=x,
        output_ptr=z,
        scale=scale,
        D=D,
        B=B
    )
    z = z.logsumexp(dim=-1).view(*shape[-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z