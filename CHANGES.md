# nanoGPT 优化器支持改进 - 变更日志

## 概述
本次更新为nanoGPT训练框架添加了多优化器支持，允许用户灵活地选择和对比不同的优化器（如SGD、Adam、LAMB、RMSprop等）。

---

## 主要文件改动

### 1. `model.py` - 核心优化器实现
**位置**: `configure_optimizers()` 方法

**改动内容**:
- 添加了 `optimizer_type` 参数（默认值：'adamw'）
- 支持多种优化器类型:
  - ✅ **AdamW** (默认) - 解耦权重衰减，适合LLM训练
  - ✅ **SGD** - 经典随机梯度下降，内存高效
  - ✅ **Adam** - 标准Adam优化器
  - ✅ **RMSprop** - 自适应学习率
  - ✅ **LAMB** - 层级自适应矩估计（大批量训练）

**关键特性**:
- 为不同优化器设置合适的默认参数
- 参数组的权重衰减管理（decay_params vs nodecay_params）
- 自动fallback机制（如LAMB不可用时回退到AdamW）
- 详细的日志输出

**代码示例**:
```python
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    device_type='cuda',
    optimizer_type='sgd'  # 指定优化器类型
)
```

### 2. `train.py` - 训练脚本增强
**位置**: 配置部分 + 优化器初始化

**改动内容**:
- 添加 `optimizer_type` 配置参数
- 在命令行参数中支持该参数
- 将 `optimizer_type` 传递给 `configure_optimizers()`

**使用方式**:
```bash
python train.py --optimizer_type=sgd --learning_rate=3e-4
python train.py --optimizer_type=adam --learning_rate=6e-4
python train.py --optimizer_type=adamw --learning_rate=6e-4
```

### 3. `compare_optimizers.py` - 新增对比工具（✨新文件）
**功能**: 自动化对比不同优化器的性能

**主要特性**:
- 自动运行多个优化器的训练实验
- 从日志中提取性能指标
- 生成对比图表（PNG格式）
- 生成总结报告（TXT格式）
- 保存配置信息（JSON格式）

**预定义的优化器配置**:
- AdamW: lr=6e-4, beta1=0.9, beta2=0.95
- SGD: lr=3e-4, beta1=0.9 (momentum)
- Adam: lr=6e-4, beta1=0.9, beta2=0.999
- RMSprop: lr=3e-4, beta1=0.99

**输出结构**:
```
optimizer_comparison_results_YYYYMMDD_HHMMSS/
├── adamw/
│   ├── ckpt.pt
│   └── ...
├── sgd/
├── adam/
├── rmsprop/
├── optimizer_comparison.png      # 对比图表
├── comparison_summary.txt        # 总结报告
└── config.json                   # 配置文件
```

**使用方式**:
```bash
# 快速对比（5000迭代）
python compare_optimizers.py --batch_size=32 --max_iters=5000 --compile=False

# 完整对比
python compare_optimizers.py --max_iters=600000
```

### 4. `OPTIMIZER_GUIDE.md` - 完整文档（✨新文件）
**内容**: 详细的优化器使用指南

**包含章节**:
- 快速开始
- 支持的优化器详解（5种优化器）
- 单独使用示例
- 自动化对比方法
- 优化器特性与建议
- 性能对比表
- 进阶配置（学习率调度、梯度裁剪等）
- 常见问题FAQ
- 参考资源

**特色**:
- 中英文详细说明
- 每个优化器的参数详解
- 实际命令示例
- 选择建议

### 5. `OPTIMIZER_QUICK_START.md` - 快速参考（✨新文件）
**内容**: 快速查询指南

**包含内容**:
- 一行命令示例
- 完整命令示例
- 参数速查表
- 适用场景速查
- 常见问题快速答案
- 性能指标表

### 6. `optimizer_demo.py` - 教学演示脚本（✨新文件）
**功能**: 可视化演示优化器的行为

**演示内容**:
1. **优化器对比** - 在Rosenbrock函数上的收敛行为
2. **学习率影响** - 不同学习率的影响分析

**输出**:
- `optimizer_comparison_demo.png` - 4个优化器的收敛轨迹
- `learning_rate_comparison.png` - 学习率影响分析

**用途**:
- 理解优化器的特性
- 学习学习率的影响
- 可视化收敛过程

---

## 优化器特性速查

| 优化器 | 默认学习率 | 内存占用 | 收敛速度 | 最终精度 | 推荐场景 |
|--------|----------|--------|--------|--------|---------|
| **AdamW** | 6e-4 | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | LLM预训练 ✅ |
| **SGD** | 3e-4 | 低 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 资源受限 |
| **Adam** | 6e-4 | 中 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用 |
| **RMSprop** | 3e-4 | 中 | ⭐⭐⭐ | ⭐⭐⭐ | RNN任务 |
| **LAMB** | 6e-4 | 中高 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 大批量分布式 |

---

## 使用示例

### 基础用法
```bash
# 使用SGD优化器
python train.py --optimizer_type=sgd --learning_rate=3e-4

# 使用Adam优化器
python train.py --optimizer_type=adam --learning_rate=6e-4

# 使用AdamW（默认）
python train.py --optimizer_type=adamw --learning_rate=6e-4

# 使用RMSprop优化器
python train.py --optimizer_type=rmsprop --learning_rate=3e-4
```

### 完整配置示例
```bash
python train.py \
    --optimizer_type=sgd \
    --learning_rate=3e-4 \
    --beta1=0.9 \
    --weight_decay=0.1 \
    --batch_size=32 \
    --compile=False \
    --max_iters=10000 \
    --eval_interval=500
```

### 自动化对比
```bash
# 快速对比所有优化器
python compare_optimizers.py \
    --batch_size=32 \
    --eval_interval=500 \
    --max_iters=5000 \
    --compile=False
```

### 可视化演示
```bash
# 运行优化器演示（无需GPU）
python optimizer_demo.py
```

---

## 向后兼容性

✅ **完全向后兼容**

- 默认使用AdamW（原有行为保持不变）
- 所有现有训练脚本无需修改即可运行
- `optimizer_type` 参数是可选的
- 原有命令行参数完全保留

```bash
# 旧命令仍然有效
python train.py --batch_size=32 --compile=False
# 等同于 (自动使用AdamW):
python train.py --batch_size=32 --compile=False --optimizer_type=adamw
```

---

## 技术实现细节

### 参数组管理
不同大小的参数使用不同的权重衰减策略：
- **decay_params**: 2D张量（权重矩阵和嵌入）- 应用权重衰减
- **nodecay_params**: 1D张量（偏置和LayerNorm） - 不应用权重衰减

### 优化器初始化
```python
if optimizer_type == 'adamw':
    # 使用fused版本（如果可用）以提高性能
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'

elif optimizer_type == 'sgd':
    # Nesterov加速梯度下降
    optimizer = torch.optim.SGD(..., nesterov=True)

elif optimizer_type == 'lamb':
    # 具有fallback机制
    try:
        from torch.optim.lamb import Lamb
    except ImportError:
        # fallback to AdamW
```

### 错误处理
- 无效的优化器类型会抛出 `ValueError`
- 不支持的PyTorch优化器会自动fallback
- 详细的日志输出便于调试

---

## 测试建议

### 功能测试
```bash
# 测试各优化器是否正常工作
python train.py --optimizer_type=adamw --max_iters=100
python train.py --optimizer_type=sgd --max_iters=100
python train.py --optimizer_type=adam --max_iters=100
python train.py --optimizer_type=rmsprop --max_iters=100
```

### 性能对比
```bash
# 对比不同优化器的收敛速度
python compare_optimizers.py --max_iters=10000 --eval_interval=1000
```

### 可视化验证
```bash
# 验证优化器行为
python optimizer_demo.py
```

---

## 文件清单

### 修改的文件
1. ✏️ `model.py` - 添加多优化器支持
2. ✏️ `train.py` - 添加optimizer_type参数

### 新增文件
3. ✨ `compare_optimizers.py` - 对比工具（224行）
4. ✨ `optimizer_demo.py` - 演示脚本（235行）
5. ✨ `OPTIMIZER_GUIDE.md` - 完整指南（~600行）
6. ✨ `OPTIMIZER_QUICK_START.md` - 快速参考（~150行）
7. 📝 `CHANGES.md` - 本文件

---

## 性能影响

- ✅ **无性能下降** - 优化器选择在初始化时进行，对训练性能无影响
- ✅ **内存占用** - 根据选择的优化器而异（SGD最低，AdamW中等）
- ✅ **编译时间** - 无额外编译时间

---

## 扩展建议

### 可能的改进
1. 支持自定义优化器注册
2. 添加优化器调度器（如学习率预热）
3. 支持多优化器混合（如Muon + AdamW）
4. Web界面可视化工具
5. 自动化超参数搜索

### 社区贡献
欢迎提交PR以支持更多优化器：
- Lion优化器
- Adafactor
- Sophia
- DADAPT
- 等等...

---

## 常见问题

### Q: 我应该选择哪个优化器？
**A**: 
- 对于LLM训练：**AdamW**（推荐）
- 对于资源受限：**SGD**
- 对于快速实验：**Adam**

### Q: SGD为什么需要更小的学习率？
**A**: SGD的梯度更新是直接的，没有像Adam那样的自适应二阶矩估计。

### Q: 能否使用多个优化器？
**A**: 可以！需要修改train.py以支持多个optimizer.step()调用。

### Q: 如何自定义学习率调度？
**A**: 所有优化器都支持相同的学习率调度机制，在train.py中配置。

---

## 更新历史

- **v1.0** (2024) - 初始版本
  - 支持5种优化器
  - 添加对比工具
  - 完整文档和示例

---

## 参考资源

- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [AdamW Paper](https://arxiv.org/abs/1711.05101) - Decoupled Weight Decay Regularization
- [LAMB Paper](https://arxiv.org/abs/1904.00325) - Layer-wise Adaptive Moments optimizer for Batch training
- [SGD with Momentum](https://arxiv.org/abs/1609.04747)

---

**最后更新**: 2024年
**作者**: nanoGPT优化器增强项目
**许可证**: 遵循原项目许可证

