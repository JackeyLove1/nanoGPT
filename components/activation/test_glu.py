import torch

from SwiGLU import SwiGLU_FFN
from GeGLU import GeGLU_FFN

def test_SwiGLU_shape():
    torch.manual_seed(0)
    B, T, d_model = 32, 128, 1024
    ffn = SwiGLU_FFN(d_model, ffn_dim_multiplier=4, multiple_of=256)
    x = torch.randn(B, T, d_model)
    y = ffn(x)
    print(f"y shape: {y.shape}")
    assert y.shape == x.shape

def test_GeGLU_shape():
    torch.manual_seed(0)
    B, T, d_model = 32, 128, 1024
    ffn = GeGLU_FFN(d_model, ffn_dim_multiplier=4, multiple_of=256)
    x = torch.randn(B, T, d_model)
    y = ffn(x)
    print(f"y shape: {y.shape}")
    assert y.shape == x.shape