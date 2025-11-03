import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

# 近似计算 默认使用
if os.environ.get('FLA_USE_FAST_OPS', '1') == '1':
    exp = tldevice.exp
    exp2 = tldevice.exp2
    log = tldevice.log
    log2 = tldevice.log2
else:
    exp = tl.exp
    exp2 = tl.exp2
    log = tl.log
    log2 = tl.log2

@triton.jit
def safe_exp(x):
    return exp(tl.where(x > 0, x, int('-float')))