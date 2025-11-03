import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

# use fast exp implement
exp = tldevice.exp

NUM_WARPS_AUTOTUNE = [2,4,8,16,32]
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_wraps) for num_wraps in NUM_WARPS_AUTOTUNE],
    key=["D"],
    cache_results=True
)
@triton.jit
def softmax_fwd_kernel(
        x, y,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    """x: [B * S, D]"""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    block_x = tl.load(x + row * BLOCK_SIZE + offsets, mask=mask, other=-int('float'))
    block_max = tl.max(block_x, 0)
    block_x = exp(block_x - block_max)
    block_prob = block_x / tl.sum(block_x, 0)

    tl.store(y + row * BLOCK_SIZE + offsets, block_prob.to(x.dtype.element_ty),  mask=mask)

@triton.autotune(
    configs=[triton.Config({}, num_warps=num_wraps) for num_wraps in NUM_WARPS_AUTOTUNE],
    key=["D"],
    cache_results=True
)
@triton.jit
def softmax_bwd_kernel(
        y_ptr,  # 前向的 softmax 输出 y
        dy_ptr,  # 来自上一层的梯度 dL/dy
        dx_ptr,  # 要写回的梯度 dL/dz
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < BLOCK_SIZE

    y = tl.load(y_ptr + row * n_cols + offs, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row * n_cols + offs, mask=mask, other=0.0)

    # 1. s = sum_i (y_i, dy_i)
    s = tl.sum(y * dy, axis=0)
    # 2. dx_i = y_i * (dy_i - s)
    dx = y * (dy - s)

    tl.store(dx_ptr + row * n_cols + offs, dx,mask=mask)


def softmax_fwd(
        x: torch.Tensor,
        dtype: torch.Type | None = torch.float32
) -> torch.Tensor:
    shape = x.shape
    x = x.view(-1, x.shape[-1])

    N, D = x.shape
    BLOCK_SIZE = triton.next_power_of_2(D)

    out = torch.empty_like(x, dtype=dtype)
    softmax_fwd_kernel[(N, )](
        x=x,
        y=out,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE
    )



def softmax_bwd(
    y: torch.Tensor,
    dy: torch.Tensor,
    dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    shape = y.shape
    y = y.view(-1, y.shape[-1])
    dx = torch.empty_like(y, dtype=dtype)

    N, D = y.shape
    BLOCK_SIZE = triton.next_power_of_2(D)
    softmax_bwd_kernel[(N,)](
        y=y,
        dy=dy,
        dx=dx,
        n_cols=D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dx.view(*shape)