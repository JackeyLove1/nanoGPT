"""
Pack/Unpack 序列示例演示
展示如何处理变长序列的打包和解包操作
"""

import torch
from fla.ops.utils.pack import pack_sequence, unpack_sequence


def create_sample_data():
    """创建示例数据：3个不同长度的序列"""
    # 假设我们有3个句子，特征维度D=4
    # 句子1: 长度3, 句子2: 长度2, 句子3: 长度4

    batch_size = 3
    max_seq_len = 4  # 最大序列长度
    feature_dim = 4  # 特征维度

    # 真实序列长度
    actual_lengths = torch.tensor([3, 2, 4]).cuda()

    return batch_size, max_seq_len, feature_dim, actual_lengths


def demo_right_padding():
    """演示右侧padding的pack/unpack操作"""
    print("=" * 80)
    print("【示例1: 右侧Padding (padding_side='right')】")
    print("=" * 80)

    B, S, D, lengths = create_sample_data()

    # 创建带padding的输入张量 (B=3, S=4, D=4)
    # 为了便于理解，我们用简单的数字填充
    x = torch.zeros(B, S, D, dtype=torch.float32).cuda()

    # Batch 0: 长度3，有效数据在前3个位置
    x[0, :3, :] = torch.arange(1, 13).reshape(3, 4).float()

    # Batch 1: 长度2，有效数据在前2个位置
    x[1, :2, :] = torch.arange(13, 21).reshape(2, 4).float()

    # Batch 2: 长度4，全是有效数据
    x[2, :4, :] = torch.arange(21, 37).reshape(4, 4).float()

    print(f"\n📥 输入张量形状: {x.shape} (Batch={B}, SeqLen={S}, Features={D})")
    print(f"真实序列长度: {lengths.tolist()}")
    print("\n输入数据 (右侧padding):")
    for i in range(B):
        print(f"Batch {i} (长度={lengths[i]}):")
        print(x[i].cpu().numpy())
        print()

    # 计算累积序列长度 cu_seqlens
    cu_seqlens = torch.cat([
        torch.tensor([0]).cuda(),
        torch.cumsum(lengths, dim=0)
    ])
    print(f"cu_seqlens (累积序列长度): {cu_seqlens.tolist()}")
    print(
        f"  解释: batch0=[0:{cu_seqlens[1]}], batch1=[{cu_seqlens[1]}:{cu_seqlens[2]}], batch2=[{cu_seqlens[2]}:{cu_seqlens[3]}]")

    # ==================== PACK 操作 ====================
    print("\n" + "─" * 80)
    print("🔽 执行 PACK 操作 (移除padding)")
    print("─" * 80)

    packed = pack_sequence(x, cu_seqlens, padding_side='right')

    print(
        f"\n📦 打包后形状: {packed.shape} (从 {B}×{S}×{D}={B * S * D} 压缩到 {packed.shape[0]}×{D}={packed.shape[0] * D})")
    print(f"节省空间: {B * S * D - packed.shape[0] * D} 个元素 ({(1 - packed.shape[0] * D / (B * S * D)) * 100:.1f}%)")
    print("\n打包后数据 (紧凑格式，无padding):")
    print(packed.cpu().numpy())
    print("\n数据来源:")
    for i in range(B):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        print(f"  位置[{start}:{end}] <- Batch {i} 的有效数据")

    # ==================== UNPACK 操作 ====================
    print("\n" + "─" * 80)
    print("🔼 执行 UNPACK 操作 (恢复padding)")
    print("─" * 80)

    unpacked = unpack_sequence(packed, cu_seqlens, padding_side='right', desired_shape=x.shape)

    print(f"\n📤 解包后形状: {unpacked.shape}")
    print("\n解包后数据 (恢复padding):")
    for i in range(B):
        print(f"Batch {i}:")
        print(unpacked[i].cpu().numpy())
        print()

    # 验证往返一致性
    print("✅ 验证结果:")
    if torch.allclose(x, unpacked, atol=1e-5):
        print("   Pack → Unpack 往返验证成功！数据完全一致。")
    else:
        print("   ❌ 数据不一致！")
        print(f"   最大差异: {torch.max(torch.abs(x - unpacked)).item()}")


def demo_left_padding():
    """演示左侧padding的pack/unpack操作"""
    print("\n\n" + "=" * 80)
    print("【示例2: 左侧Padding (padding_side='left')】")
    print("=" * 80)

    B, S, D, lengths = create_sample_data()

    # 创建带左侧padding的输入张量
    x = torch.zeros(B, S, D, dtype=torch.float32).cuda()

    # Batch 0: 长度3，左侧padding 1个位置
    x[0, 1:4, :] = torch.arange(1, 13).reshape(3, 4).float()

    # Batch 1: 长度2，左侧padding 2个位置
    x[1, 2:4, :] = torch.arange(13, 21).reshape(2, 4).float()

    # Batch 2: 长度4，无padding
    x[2, :4, :] = torch.arange(21, 37).reshape(4, 4).float()

    print(f"\n📥 输入张量形状: {x.shape}")
    print("\n输入数据 (左侧padding):")
    for i in range(B):
        padding_count = S - lengths[i].item()
        print(f"Batch {i} (长度={lengths[i]}, 左侧padding={padding_count}):")
        print(x[i].cpu().numpy())
        print()

    cu_seqlens = torch.cat([
        torch.tensor([0]).cuda(),
        torch.cumsum(lengths, dim=0)
    ])

    # PACK
    print("🔽 执行 PACK 操作 (移除左侧padding)")
    packed = pack_sequence(x, cu_seqlens, padding_side='left')

    print(f"\n📦 打包后形状: {packed.shape}")
    print("打包后数据:")
    print(packed.cpu().numpy())

    # UNPACK
    print("\n🔼 执行 UNPACK 操作 (恢复左侧padding)")
    unpacked = unpack_sequence(packed, cu_seqlens, padding_side='left', desired_shape=x.shape)

    print(f"\n📤 解包后形状: {unpacked.shape}")
    print("\n解包后数据:")
    for i in range(B):
        print(f"Batch {i}:")
        print(unpacked[i].cpu().numpy())
        print()

    # 验证
    print("✅ 验证结果:")
    if torch.allclose(x, unpacked, atol=1e-5):
        print("   Pack → Unpack 往返验证成功！")
    else:
        print("   ❌ 数据不一致！")


def demo_real_world_use_case():
    """演示真实应用场景：文本序列处理"""
    print("\n\n" + "=" * 80)
    print("【示例3: 真实应用场景 - 文本序列Embedding】")
    print("=" * 80)

    # 模拟3个句子的token embeddings
    sentences = [
        "Hello world !",  # 3 tokens
        "How are you ?",  # 4 tokens
        "AI is amazing",  # 3 tokens
    ]

    lengths = torch.tensor([3, 4, 3]).cuda()
    max_len = 4
    embed_dim = 8  # embedding维度

    print("\n原始句子:")
    for i, sent in enumerate(sentences):
        print(f"  句子{i}: '{sent}' (长度={lengths[i]})")

    # 创建随机embeddings (模拟真实embedding输出)
    torch.manual_seed(42)
    x = torch.randn(3, max_len, embed_dim)

    # 将padding位置置零
    for i in range(3):
        x[i, lengths[i]:, :] = 0

    print(f"\n带padding的embeddings形状: {x.shape}")
    print(f"总元素数: {x.numel()}")

    cu_seqlens = torch.cat([torch.tensor([0]).cuda(), torch.cumsum(lengths, dim=0)])

    # Pack: 移除padding，节省计算
    packed = pack_sequence(x, cu_seqlens, padding_side='right')

    print(f"\n打包后形状: {packed.shape}")
    print(f"总元素数: {packed.numel()}")
    print(f"💾 节省内存: {(1 - packed.numel() / x.numel()) * 100:.1f}%")

    print("""
    💡 应用场景说明:
    ─────────────────────────────────────────────────────────────
    1. 在Transformer中，pack后的序列可以直接送入attention层
    2. 只计算有效token，跳过padding，节省30%-50%的计算量
    3. 在Flash Attention等优化算法中特别重要
    4. 梯度反向传播时，自动unpack回原始形状
    ─────────────────────────────────────────────────────────────
    """)


def demo_gradient_flow():
    """演示梯度流动"""
    print("\n\n" + "=" * 80)
    print("【示例4: 梯度流动验证】")
    print("=" * 80)

    B, S, D = 2, 3, 4
    lengths = torch.tensor([2, 3]).cuda()
    cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

    # 创建需要梯度的输入
    x = torch.randn(B, S, D, requires_grad=True).cuda()

    print(f"输入形状: {x.shape}, requires_grad={x.requires_grad}")

    # Pack操作
    packed = pack_sequence(x, cu_seqlens, padding_side='right')
    print(f"Pack后形状: {packed.shape}, requires_grad={packed.requires_grad}")

    # 模拟某种计算（例如经过一个线性层）
    loss = (packed ** 2).sum()
    print(f"损失值: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    print(f"\n✅ 梯度成功传播！")
    print(f"输入梯度形状: {x.grad.shape}")
    print(f"输入梯度 (只有有效位置有梯度):")
    print(x.grad)
    print("\n注意: padding位置的梯度为0，因为它们没有参与前向计算")


if __name__ == "__main__":
    # 运行所有示例
    demo_right_padding()
    demo_left_padding()
    demo_real_world_use_case()
    demo_gradient_flow()

    print("\n\n" + "=" * 80)
    print("🎉 所有示例运行完成！")
    print("=" * 80)

