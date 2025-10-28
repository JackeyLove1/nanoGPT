"""Test script to verify RoPE.py and RoPELlama.py produce consistent results"""
import torch
import torch.nn as nn
from RoPE import RoPE
from RoPELlama import precompute_freqs_cis, apply_rotary_emb


def test_rope_consistency():
    """
    测试RoPE.py和RoPELlama.py两种实现对同一tensor的embedding效果是否一致
    """
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    
    # 定义参数
    batch_size = 2
    seq_len = 8
    n_head = 4
    head_dim = 64  # 必须是偶数
    base = 10000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    print(f"测试配置:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {n_head}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Base/Theta: {base}")
    print("-" * 60)
    
    # 创建测试数据 (batch, seq, n_head, dim)
    x = torch.randn(batch_size, seq_len, n_head, head_dim, device=device, dtype=dtype)
    print(f"输入tensor shape: {x.shape}")
    
    # ========== 方法1: 使用RoPE.py ==========
    print("\n使用RoPE.py进行embedding...")
    rope_module = RoPE(
        dim=head_dim,
        max_seq_len=seq_len,
        base=base,
        device=device,
        dtype=dtype
    )
    output1 = rope_module(x)
    print(f"RoPE.py输出shape: {output1.shape}")
    print(f"RoPE.py输出统计: mean={output1.mean():.6f}, std={output1.std():.6f}")
    
    # ========== 方法2: 使用RoPELlama.py ==========
    print("\n使用RoPELlama.py进行embedding...")
    # 预计算频率
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=seq_len,
        theta=base
    ).to(device)
    print(f"freqs_cis shape: {freqs_cis.shape}")
    
    # apply_rotary_emb需要xq和xk，我们用同一个x来测试
    # 注意：RoPELlama的实现期望的输入形状也是(batch, seq, n_head, dim)
    output2, _ = apply_rotary_emb(x, x, freqs_cis)
    print(f"RoPELlama.py输出shape: {output2.shape}")
    print(f"RoPELlama.py输出统计: mean={output2.mean():.6f}, std={output2.std():.6f}")
    
    # ========== 比较结果 ==========
    print("\n" + "=" * 60)
    print("结果比较:")
    print("=" * 60)
    
    # 计算差异
    diff = torch.abs(output1 - output2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"最大绝对差异: {max_diff:.10f}")
    print(f"平均绝对差异: {mean_diff:.10f}")
    
    # 计算相对误差
    relative_error = (diff / (torch.abs(output1) + 1e-8)).mean().item()
    print(f"平均相对误差: {relative_error:.10f}")
    
    # 使用allclose进行比较
    # 默认容差: rtol=1e-05, atol=1e-08
    is_close_default = torch.allclose(output1, output2)
    is_close_relaxed = torch.allclose(output1, output2, rtol=1e-4, atol=1e-6)
    
    print(f"\ntorch.allclose (默认容差): {is_close_default}")
    print(f"torch.allclose (放宽容差 rtol=1e-4, atol=1e-6): {is_close_relaxed}")
    
    # 详细分析差异分布
    print(f"\n差异分布:")
    print(f"  差异 < 1e-6 的元素占比: {(diff < 1e-6).float().mean().item() * 100:.2f}%")
    print(f"  差异 < 1e-5 的元素占比: {(diff < 1e-5).float().mean().item() * 100:.2f}%")
    print(f"  差异 < 1e-4 的元素占比: {(diff < 1e-4).float().mean().item() * 100:.2f}%")
    
    # 判断测试是否通过
    print("\n" + "=" * 60)
    if is_close_relaxed:
        print("✓ 测试通过！两种实现的RoPE embedding结果一致！")
    else:
        print("✗ 测试失败！两种实现存在较大差异！")
        print("\n显示部分差异较大的位置:")
        large_diff_mask = diff > 1e-4
        if large_diff_mask.any():
            indices = torch.nonzero(large_diff_mask)[:5]  # 显示前5个
            for idx in indices:
                b, s, h, d = idx.tolist()
                print(f"  位置[{b},{s},{h},{d}]: RoPE={output1[b,s,h,d]:.6f}, "
                      f"RoPELlama={output2[b,s,h,d]:.6f}, diff={diff[b,s,h,d]:.6f}")
    print("=" * 60)
    
    return is_close_relaxed


def test_rope_properties():
    """
    测试RoPE的一些基本性质
    """
    print("\n\n" + "=" * 60)
    print("测试RoPE的基本性质")
    print("=" * 60)
    
    torch.manual_seed(123)
    
    batch_size = 1
    seq_len = 4
    n_head = 2
    head_dim = 32
    base = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建RoPE模块
    rope = RoPE(dim=head_dim, max_seq_len=seq_len, base=base, device=device)
    
    # 测试1: RoPE不改变向量的模长（近似保持）
    x = torch.randn(batch_size, seq_len, n_head, head_dim, device=device)
    x_rope = rope(x)
    
    norm_before = torch.norm(x, dim=-1)
    norm_after = torch.norm(x_rope, dim=-1)
    norm_diff = torch.abs(norm_before - norm_after).max().item()
    
    print(f"\n1. 模长保持性测试:")
    print(f"   最大模长差异: {norm_diff:.10f}")
    print(f"   模长是否保持: {'✓ 是' if norm_diff < 1e-5 else '✗ 否'}")
    
    # 测试2: 不同位置应用不同的旋转
    x_uniform = torch.ones(1, seq_len, 1, head_dim, device=device)
    x_rope_uniform = rope(x_uniform)
    
    # 检查不同位置的输出是否不同
    pos_0 = x_rope_uniform[0, 0, 0, :]
    pos_1 = x_rope_uniform[0, 1, 0, :]
    pos_diff = torch.norm(pos_0 - pos_1).item()
    
    print(f"\n2. 位置依赖性测试:")
    print(f"   位置0和位置1的输出差异: {pos_diff:.6f}")
    print(f"   不同位置输出不同: {'✓ 是' if pos_diff > 0.01 else '✗ 否'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE实现一致性测试")
    print("=" * 60)
    
    # 运行主要的一致性测试
    test_passed = test_rope_consistency()
    
    # 运行性质测试
    test_rope_properties()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

