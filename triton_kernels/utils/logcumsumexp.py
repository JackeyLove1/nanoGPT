import torch
import numpy as np

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from jaxtyping import Float

exp = tldevice.exp
log = tldevice.log

# numpy version
def logcumsumexp_fwd(s_bh: Float[np.ndarray, "T S"],BT: int, z_bh) -> Float[np.ndarray, "T S"]:
    # running state
    T, S = s_bh.shape
    m_prev = np.full(S, -np.inf)
    a_prev = np.zeros(S)

    for block_start in range(0, T, BT):
        block_end = min(block_start + BT, T)

        # [BT, S] block
        b_s = s_bh[block_start:block_end, :]             # pad 到 BT 行可以理解为 masked

        # 当前块局部 max
        m_block = b_s.max(axis=0)                        # [S]
        m_cur   = np.maximum(m_prev, m_block)            # [S]

        # 重新缩放之前块的和到 new max
        a_prev = a_prev * np.exp(m_prev - m_cur)         # [S]

        # 当前块减 max 后 exp
        e = np.exp(b_s - m_cur)                          # [BT, S]

        # 块内前缀和 + 之前块的和
        a_block = np.cumsum(e, axis=0) + a_prev          # [BT, S]

        # 写输出：logcumsumexp
        s_bh_out_block = np.log(np.maximum(a_block, 1e-20)) + m_cur
        z_bh[block_start:block_end, :] = s_bh_out_block

        # 更新状态给下一块（用这一块最后一行的值）
        a_prev = a_block[-1, :]
        m_prev = m_cur

    return z_bh



"""
tl.make_block_ptr(
    base,           # 基地址指针
    shape,          # 完整张量的形状
    strides,        # 每个维度的步幅
    offsets,        # 当前块的起始偏移
    block_shape,    # 要加载的块大小
    order           # 维度顺序
)
"""

@triton.jit
def logcumsumexp_fwd_kernel(
        input_ptr, # (B*H, T, S)
        output_ptr, # (B*H, T, S)
        T, # seq_len
        S: tl.constexpr,
        BT: tl.constexpr,
):
    pass