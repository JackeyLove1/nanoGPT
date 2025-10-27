"""
Optimizer Comparison Script
This script runs multiple training experiments with different optimizers and compares their performance.

Example usage:
$ python compare_optimizers.py --batch_size=32 --eval_interval=500 --max_iters=10000
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 要对比的优化器配置
OPTIMIZERS = {
    'adamw': {
        'optimizer_type': 'adamw',
        'learning_rate': 6e-4,
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': 1e-1,
    },
    'sgd': {
        'optimizer_type': 'sgd',
        'learning_rate': 3e-4,  # SGD通常需要更小的学习率
        'beta1': 0.9,  # SGD使用beta1作为momentum
        'beta2': 0.95,
        'weight_decay': 1e-1,
    },
    'adam': {
        'optimizer_type': 'adam',
        'learning_rate': 6e-4,
        'beta1': 0.9,
        'beta2': 0.999,  # Adam默认为0.999
        'weight_decay': 0.0,  # Adam不推荐weight decay（使用AdamW代替）
    },
    'rmsprop': {
        'optimizer_type': 'rmsprop',
        'learning_rate': 3e-4,
        'beta1': 0.99,  # RMSprop的alpha参数
        'beta2': 0.95,
        'weight_decay': 1e-1,
    },
}

def run_training_experiment(optimizer_name, config_args, out_dir):
    """运行单个优化器的训练实验"""
    print(f"\n{'='*60}")
    print(f"开始训练实验: {optimizer_name.upper()}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'train.py',
        f'--out_dir={out_dir}/{optimizer_name}',
        f'--optimizer_type={config_args["optimizer_type"]}',
        f'--learning_rate={config_args["learning_rate"]}',
        f'--beta1={config_args["beta1"]}',
        f'--beta2={config_args["beta2"]}',
        f'--weight_decay={config_args["weight_decay"]}',
        f'config/train_shakespeare_char.py'
    ]
    
    # 添加额外的命令行参数
    cmd.extend(sys.argv[1:])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行训练
    result = subprocess.run(cmd, cwd='/root/autodl-tmp/nanoGPT')
    
    if result.returncode != 0:
        print(f"警告: {optimizer_name}训练失败")
        return False
    
    return True

def extract_metrics_from_log(log_file):
    """从日志文件中提取指标"""
    metrics = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # 解析 "step XXX: train loss X.XXXX, val loss X.XXXX" 格式
                if 'step' in line and 'train loss' in line and 'val loss' in line:
                    try:
                        parts = line.split()
                        step_idx = parts.index('step')
                        train_idx = parts.index('loss') - 2  # 'train' 'loss' 的前一个是数值
                        val_idx = parts.index('loss', train_idx + 3) - 2
                        
                        iter_num = int(parts[step_idx + 1].rstrip(':'))
                        train_loss = float(parts[train_idx + 2].rstrip(','))
                        val_loss = float(parts[val_idx + 2])
                        
                        metrics['iterations'].append(iter_num)
                        metrics['train_loss'].append(train_loss)
                        metrics['val_loss'].append(val_loss)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"警告: 日志文件 {log_file} 未找到")
    
    return metrics

def plot_comparison(results, output_dir):
    """绘制优化器对比图表"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练损失对比
    ax = axes[0]
    for optimizer_name, metrics in results.items():
        if metrics['iterations']:
            ax.plot(metrics['iterations'], metrics['train_loss'], 
                   marker='o', label=optimizer_name.upper(), linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证损失对比
    ax = axes[1]
    for optimizer_name, metrics in results.items():
        if metrics['iterations']:
            ax.plot(metrics['iterations'], metrics['val_loss'], 
                   marker='s', label=optimizer_name.upper(), linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'optimizer_comparison.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"对比图表已保存到: {output_path}")
    plt.close()

def generate_summary_report(results, output_dir):
    """生成总结报告"""
    report_path = os.path.join(output_dir, 'comparison_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("优化器对比总结报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for optimizer_name, metrics in results.items():
            f.write(f"\n{optimizer_name.upper()}:\n")
            f.write("-" * 40 + "\n")
            
            if metrics['iterations']:
                final_train_loss = metrics['train_loss'][-1]
                final_val_loss = metrics['val_loss'][-1]
                min_val_loss = min(metrics['val_loss']) if metrics['val_loss'] else float('inf')
                
                f.write(f"  最终训练损失: {final_train_loss:.4f}\n")
                f.write(f"  最终验证损失: {final_val_loss:.4f}\n")
                f.write(f"  最小验证损失: {min_val_loss:.4f}\n")
                f.write(f"  总迭代次数: {metrics['iterations'][-1]}\n")
            else:
                f.write("  无数据\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"总结报告已保存到: {report_path}")

def main():
    """主函数"""
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'/root/autodl-tmp/nanoGPT/optimizer_comparison_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"结果将保存到: {results_dir}\n")
    
    # 保存配置
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(OPTIMIZERS, f, indent=2)
    
    results = {}
    
    # 运行每个优化器的训练
    for optimizer_name, config_args in OPTIMIZERS.items():
        success = run_training_experiment(optimizer_name, config_args, results_dir)
        
        if success:
            # 提取指标
            log_file = os.path.join(results_dir, optimizer_name, 'train_log.txt')
            results[optimizer_name] = extract_metrics_from_log(log_file)
        else:
            results[optimizer_name] = {
                'iterations': [],
                'train_loss': [],
                'val_loss': [],
                'learning_rates': [],
            }
    
    # 生成报告和图表
    if any(r['iterations'] for r in results.values()):
        plot_comparison(results, results_dir)
        generate_summary_report(results, results_dir)
    else:
        print("警告: 没有成功的训练实验")
    
    print(f"\n对比完成! 所有结果已保存到: {results_dir}")

if __name__ == '__main__':
    main()
