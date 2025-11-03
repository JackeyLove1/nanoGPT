import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from triton.testing import do_bench


# 定义快速版本的kernel
@triton.jit
def fast_exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tldevice.fast_expf(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def standard_exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fast_exp2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tldevice.exp2(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def standard_exp2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.exp2(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fast_log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tldevice.fast_logf(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def standard_log_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.log(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fast_log2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tldevice.fast_log2f(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def standard_log2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.log2(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark_ops():
    # 测试参数
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    BLOCK_SIZE = 1024

    print("=" * 80)
    print("Triton Math Operations Benchmark")
    print("=" * 80)

    for size in sizes:
        print(f"\n测试数据大小: {size} 元素")
        print("-" * 80)

        # 准备测试数据
        x = torch.randn(size, device='cuda')
        x_pos = torch.rand(size, device='cuda') + 0.1  # 用于log函数，确保为正数
        output = torch.empty(size, device='cuda')

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

        # 测试 exp
        time_fast_exp = do_bench(
            lambda: fast_exp_kernel[grid](x, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )
        time_std_exp = do_bench(
            lambda: standard_exp_kernel[grid](x, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )

        # 测试 exp2
        time_fast_exp2 = do_bench(
            lambda: fast_exp2_kernel[grid](x, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )
        time_std_exp2 = do_bench(
            lambda: standard_exp2_kernel[grid](x, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )

        # 测试 log
        time_fast_log = do_bench(
            lambda: fast_log_kernel[grid](x_pos, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )
        time_std_log = do_bench(
            lambda: standard_log_kernel[grid](x_pos, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )

        # 测试 log2
        time_fast_log2 = do_bench(
            lambda: fast_log2_kernel[grid](x_pos, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )
        time_std_log2 = do_bench(
            lambda: standard_log2_kernel[grid](x_pos, output, size, BLOCK_SIZE=BLOCK_SIZE)
        )

        # 打印结果
        print(f"{'操作':<20} {'Fast版本(ms)':<15} {'标准版本(ms)':<15} {'加速比':<10}")
        print(f"{'-' * 60}")

        speedup_exp = time_std_exp / time_fast_exp
        print(f"{'exp':<20} {time_fast_exp:<15.4f} {time_std_exp:<15.4f} {speedup_exp:<10.2f}x")

        speedup_exp2 = time_std_exp2 / time_fast_exp2
        print(f"{'exp2':<20} {time_fast_exp2:<15.4f} {time_std_exp2:<15.4f} {speedup_exp2:<10.2f}x")

        speedup_log = time_std_log / time_fast_log
        print(f"{'log':<20} {time_fast_log:<15.4f} {time_std_log:<15.4f} {speedup_log:<10.2f}x")

        speedup_log2 = time_std_log2 / time_fast_log2
        print(f"{'log2':<20} {time_fast_log2:<15.4f} {time_std_log2:<15.4f} {speedup_log2:<10.2f}x")

        # 精度检查
        print(f"\n精度检查 (相对误差):")
        x_test = torch.randn(100, device='cuda')
        x_pos_test = torch.rand(100, device='cuda') + 0.1
        out_fast = torch.empty(100, device='cuda')
        out_std = torch.empty(100, device='cuda')

        fast_exp_kernel[grid](x_test, out_fast, 100, BLOCK_SIZE=BLOCK_SIZE)
        standard_exp_kernel[grid](x_test, out_std, 100, BLOCK_SIZE=BLOCK_SIZE)
        rel_error_exp = torch.mean(torch.abs(out_fast - out_std) / (torch.abs(out_std) + 1e-8)).item()
        print(f"  exp相对误差:  {rel_error_exp:.6e}")

        fast_log_kernel[grid](x_pos_test, out_fast, 100, BLOCK_SIZE=BLOCK_SIZE)
        standard_log_kernel[grid](x_pos_test, out_std, 100, BLOCK_SIZE=BLOCK_SIZE)
        rel_error_log = torch.mean(torch.abs(out_fast - out_std) / (torch.abs(out_std) + 1e-8)).item()
        print(f"  log相对误差:  {rel_error_log:.6e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持才能运行此基准测试")
    else:
        benchmark_ops()