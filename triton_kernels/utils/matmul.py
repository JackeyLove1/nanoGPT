import torch

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


exp = tldevice.exp

@triton.jit
def leaky_relu(x):
    return tl.where(x > 0, x, 0.01 * x)

@triton.jit
def sigmoid(x):
    return 1.0 / (1 + exp(-x))

@triton.jit
def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

@triton.jit
def relu(x):
    return tl.maximum(x, 0.0)


@triton.heuristics({
    "HAS_ALPHA": lambda args: args['alpha'] is not None,
    "HAS_BETA": lambda args: args['beta']  is not None
})
@triton.autotune(
    configs=[
triton.Config({'BM': 128, 'BK': 64, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BM': 128, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8),
        # triton.Config({'BM': 256, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=3, num_warps=8),
        # triton.Config({'BM': 256, 'BK': 128, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 64, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 64, 'BK': 64, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4),
        # triton.Config({'BM': 128, 'BK': 64, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4)
    ],
    key=["M", "N", "K"],
    cache_results=True
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    input,
    alpha,
    beta,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `s_am` is how much to increase `a`
    # by to get the element one row down (A has M rows).
    stride_ab, stride_am, stride_ak,  # a: batch, M, K
    stride_bk, stride_bn,             # b: K, N
    stride_cb, stride_cm, stride_cn,  # c: batch, M, N
    # Meta-parameters
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_INPUT: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BETA: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    X_DIM: tl.constexpr = 1,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    index_b, index_m, index_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    NM, NN = tl.num_programs(0), tl.num_programs(1)
    index_m, index_n = tl.swizzle2d(index_m, index_n, NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details
    a_batch_ptr = a + index_b * stride_ab
    off_am = (index_m * BM + tl.arange(0, BM)) % M
    off_bn = (index_n * BN + tl.arange(0, BN)) % N
    off_k = tl.arange(0, BK)

    p_a = a_batch_ptr + (off_am[:, None] * stride_am + off_k[None, :] * stride_ak) # (BM, BK)
    p_b = b + (off_k[:, None] * stride_bk + off_bn[None, :] * stride_bn) # (BK, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        block_a = tl.load(p_a, mask=(off_k[None, :] < K - k * BK), other=0.0) # (1, BK) -> (BM, BK)
        block_b = tl.load(p_a, mask=(off_k[:, None] < K - k * BK), other=0.0) # (BK, 1) -> (BK, BN)

        acc = tl.dot(block_a, block_b, acc=acc, allow_tf32=ALLOW_TF32) # (BM, BN)

        # Advanced the ptrs to the next K block
        p_a += BK * stride_ak
        p_b += BK * stride_bk

    off_cm = index_m * BM + tl.arange(0, BM)
    off_cn = index_n * BN + tl.arange(0, BN)
    mask = (off_cm[:, None] < M) & (off_cn[None, :] < N)

    b_c = acc
    if ACTIVATION == "leaky_relu":
        b_c = leaky_relu(b_c)
    elif ACTIVATION == "relu":
        b_c = relu(b_c)
    elif ACTIVATION == "sigmoid":
        b_c = sigmoid(b_c)
    elif ACTIVATION == "tanh":
        b_c = tanh(b_c)

    if HAS_ALPHA:
        b_c *= tl.load(alpha)

    # output = matmul(A, B) + β * input
    # output = attention_output + residual
    # output = matmul(W, x) + bias
    if HAS_INPUT:
        p_i = input + (stride_cm * off_cm[:, None] if X_DIM == 2 else 0) + stride_cn * off_cn[None, :]
        mask = (off_cm[:, None] < M) & (off_cn[None, :] < N)
        mask_p = (off_cn[None, :] < N) if X_DIM == 1 else mask
        bias_i = tl.load(p_i, mask=mask_p, other=0.0).to(tl.float32)
        if HAS_BETA:
            bias_i *= tl.load(beta)
        b_c += bias_i

    c_batch_ptr = c + index_b * stride_cb
    p_c = c_batch_ptr + stride_cm * off_cm[:, None] + stride_cn * off_cn[None, :]
    tl.store(p_c, b_c.to(c.dtype.element_ty), mask=mask)

def matmul(a, b, activation=""):
    assert a.dim() in [2, 3], "a must be 2D or 3D"
    assert b.dim() == 2, "b must be 2D"
    assert a.shape[-1] == b.shape[0], f"Incompatible dimensions: A {a.shape}, B {b.shape}"

    if a.dim() == 2:
        a_dim = 2
        a = a.unsqueeze(0).contiguous()  # (1, M, K)
    else:
        a_dim = 3
    allow_tf32 = False if a.dtype == torch.float32 else True

    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    K_b, N = b.shape[0], b.shape[1]
    assert K_b == K, f"Incompatible K dimension: A {K} vs B {K_b}"
    c = a.new_empty(B, M, N)

    def grid(meta): return (N, triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    matmul_kernel[grid](
        a, b, c, None, None, None,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        ACTIVATION=activation,
        ALLOW_TF32=allow_tf32,
        HAS_INPUT=False,
    )
    return c.squeeze(0) if a_dim == 2 else c

def addmm(
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float | None = None,
        beta: float | None = None
) -> torch.Tensor:
    assert a.dim() in [2, 3], "a must be 2D or 3D"
    assert b.dim() == 2, "b must be 2D"
    assert a.shape[-1] == b.shape[0], f"Incompatible dimensions: A {a.shape}, B {b.shape}"

    if a.dim() == 2:
        a_dim = 2
        a = a.unsqueeze(0).contiguous()  # (1, M, K)
    else:
        a_dim = 3
    allow_tf32 = False if a.dtype == torch.float32 else True

    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    K_b, N = b.shape
    assert K_b == K, f"Incompatible K dimension: A {K} vs B {K_b}"
    c = a.new_empty(B, M, N)

    def grid(meta):
        return (B, triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))

    matmul_kernel[grid](
        a, b, c, x, alpha, beta,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),  # stride_ab, stride_am, stride_ak
        b.stride(0), b.stride(1),  # stride_bk, stride_bn (b.dim() == 2)
        c.stride(0), c.stride(1), c.stride(2),  # stride_cb, stride_cm, stride_cn
        ACTIVATION=None,
        ALLOW_TF32=allow_tf32,
        HAS_INPUT=True,
        X_DIM=x.dim(),
    )
    return c.squeeze(0) if a_dim == 2 else c