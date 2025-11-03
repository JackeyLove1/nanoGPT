import torch
from fla.ops.utils.matmul import matmul, addmm


def matmul_test():
    """测试基本的矩阵乘法功能"""
    print("=" * 50)
    print("测试 matmul 函数")
    print("=" * 50)

    # 示例1: 2D 矩阵乘法
    print("\n1. 2D矩阵乘法 (M, K) x (K, N) = (M, N)")
    a = torch.randn(128, 64, device='cuda', dtype=torch.float16)  # (M=128, K=64)
    b = torch.randn(64, 256, device='cuda', dtype=torch.float16)  # (K=64, N=256)
    c = matmul(a, b)
    print(f"A shape: {a.shape}, B shape: {b.shape}")
    print(f"C shape: {c.shape}")
    print(f"C = A @ B, 验证结果: {torch.allclose(c, a @ b, rtol=1e-2, atol=1e-2)}")

    # 示例2: 3D 批量矩阵乘法
    print("\n2. 3D批量矩阵乘法 (B, M, K) x (K, N) = (B, M, N)")
    a_batch = torch.randn(4, 128, 64, device='cuda', dtype=torch.float16)  # (B=4, M=128, K=64)
    b = torch.randn(64, 256, device='cuda', dtype=torch.float16)  # (K=64, N=256)
    c_batch = matmul(a_batch, b)
    print(f"A shape: {a_batch.shape}, B shape: {b.shape}")
    print(f"C shape: {c_batch.shape}")

    # 验证批量结果
    c_expected = torch.stack([a_batch[i] @ b for i in range(a_batch.shape[0])])
    print(f"批量验证结果: {torch.allclose(c_batch, c_expected, rtol=1e-2, atol=1e-2)}")

    # 示例3: 带激活函数的矩阵乘法
    print("\n3. 带激活函数的矩阵乘法")
    activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    for act in activations:
        c_act = matmul(a, b, activation=act)
        print(f"  - {act}: output shape {c_act.shape}, mean={c_act.mean().item():.4f}")


def addmm_test():
    """测试 addmm 函数 (类似 torch.addmm)"""
    print("\n" + "=" * 50)
    print("测试 addmm 函数")
    print("=" * 50)

    # 示例1: 基本的 addmm (C = A @ B + x)
    print("\n1. 基本 addmm: C = A @ B + x")
    a = torch.randn(128, 64, device='cuda', dtype=torch.float16)  # (M=128, K=64)
    b = torch.randn(64, 256, device='cuda', dtype=torch.float16)  # (K=64, N=256)
    x = torch.randn(128, 256, device='cuda', dtype=torch.float16)  # (M=128, N=256)

    c = addmm(x, a, b)
    c_expected = a @ b + x
    print(f"A shape: {a.shape}, B shape: {b.shape}, x shape: {x.shape}")
    print(f"C shape: {c.shape}")
    print(f"验证结果: {torch.allclose(c, c_expected, rtol=1e-2, atol=1e-2)}")

    # 示例2: 带 alpha 和 beta 参数 (C = alpha * (A @ B) + beta * x)
    print("\n2. 带缩放参数: C = alpha * (A @ B) + beta * x")
    alpha = 2.0
    beta = 0.5

    # 需要将 alpha 和 beta 转换为张量
    alpha_tensor = torch.tensor(alpha, device='cuda', dtype=torch.float32)
    beta_tensor = torch.tensor(beta, device='cuda', dtype=torch.float32)

    c_scaled = addmm(x, a, b, alpha=alpha_tensor, beta=beta_tensor)
    c_expected_scaled = alpha * (a @ b) + beta * x
    print(f"alpha={alpha}, beta={beta}")
    print(f"验证结果: {torch.allclose(c_scaled, c_expected_scaled, rtol=1e-2, atol=1e-2)}")

    # 示例3: 广播的输入 (x 是 1D)
    print("\n3. 广播输入 (x 是 1D 偏置)")
    x_bias = torch.randn(256, device='cuda', dtype=torch.float16)  # (N=256,)
    c_bias = addmm(x_bias, a, b)
    print(f"x_bias shape: {x_bias.shape}")
    print(f"C shape: {c_bias.shape}")
    print(f"C = A @ B + x_bias (广播)")

    # 示例4: 批量 addmm
    print("\n4. 批量 addmm")
    a_batch = torch.randn(4, 128, 64, device='cuda', dtype=torch.float16)
    x_batch = torch.randn(4, 128, 256, device='cuda', dtype=torch.float16)
    c_batch = addmm(x_batch, a_batch, b)
    print(f"A_batch shape: {a_batch.shape}, x_batch shape: {x_batch.shape}")
    print(f"C_batch shape: {c_batch.shape}")


def performance_comparison():
    """性能对比"""
    print("\n" + "=" * 50)
    print("性能对比 (Triton vs PyTorch)")
    print("=" * 50)

    import time

    # 较大的矩阵
    N = 4096
    a = torch.randn(N, N, device='cuda', dtype=torch.float16)
    b = torch.randn(N, N, device='cuda', dtype=torch.float16)

    # 预热
    for _ in range(10):
        _ = matmul(a, b)
        _ = a @ b
    torch.cuda.synchronize()

    # Triton matmul
    start = time.time()
    for _ in range(100):
        c_triton = matmul(a, b)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    # PyTorch matmul
    start = time.time()
    for _ in range(100):
        c_torch = a @ b
    torch.cuda.synchronize()
    torch_time = time.time() - start

    print(f"\n矩阵大小: {a.shape} x {b.shape}")
    print(f"Triton matmul: {triton_time * 10:.2f} ms")
    print(f"PyTorch matmul: {torch_time * 10:.2f} ms")
    print(f"加速比: {torch_time / triton_time:.2f}x")
    print(f"结果一致性: {torch.allclose(c_triton, c_torch, rtol=1e-2, atol=1e-2)}")


if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持才能运行此示例")
        exit(1)

    print("开始测试 matmul.py 功能...\n")

    # 运行测试
    # matmul_test()
    # addmm_test()
    performance_comparison()

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)