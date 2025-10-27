"""
Prepare the WikiText-103 dataset using GPT-2 BPE tokenizer.
- Uses OpenAI's tiktoken library for fast tokenization
- Save train.bin, val.bin containing the token ids, and meta.pkl containing metadata
"""
import os
import pickle
import numpy as np
from datasets import load_dataset
"""
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
"""
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system("export HF_ENDPOINT=https://hf-mirror.com") # for CN region download
os.system("export HF_HOME=/root/autodl-tmp/hf")
# Import GPT-2 tokenizer
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    os.system("pip install tiktoken")
    import tiktoken

# Get the directory path
data_dir = os.path.dirname(__file__)
# dataset = "Salesforce/wikitext" ~ 200M
dataset = "tiennv/english-wiki-corpus" #  ~1.41G

# Initialize GPT-2 tokenizer
print("Loading GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # GPT-2 vocab size: 50257
print(f"GPT-2 vocab size: {vocab_size}")

# Download the WikiText-103-raw-v1 dataset
print("\nLoading WikiText-103-raw-v1 dataset...")
ds = load_dataset(dataset)

# 检查数据集中可用的分割
print("Available splits:", list(ds.keys()))

# Combine train and validation data
print("Combining train and validation splits...")
print("train: ", ds['train'])

# 检查是否存在validation分割，如果不存在，从train中分割
if 'validation' not in ds:
    print("⚠️ No 'validation' split found in dataset. Splitting train data (90% train, 10% val)...")
    from datasets import DatasetDict, load_dataset
    
    # 从train分割出验证集
    split_dataset = ds['train'].train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    print(f"Train split: {len(ds['train'])} samples")
    print(f"Validation split: {len(ds['validation'])} samples")

# 使用流式处理避免OOM：分批处理、边读边tokenize、直接写入
def stream_and_tokenize(split_name, batch_size=100000):
    """
    流式读取数据集，分批过滤和tokenize，返回token生成器
    batch_size: 每次处理的样本数，调整此参数平衡内存和速度
    """
    dataset_split = ds[split_name]
    total_samples = len(dataset_split)
    
    print(f"\n{split_name.upper()} Split: {total_samples:,} samples, processing in batches of {batch_size}...")
    
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_texts = dataset_split['text'][batch_start:batch_end]
        
        # 过滤并清理文本（逐条处理，避免拼接整个分割）
        cleaned_texts = []
        for text in batch_texts:
            lines = text.split('\n')
            # 移除只有"="的header行
            cleaned_lines = [line for line in lines if line.strip() and not line.strip().startswith('=')]
            if cleaned_lines:
                cleaned_texts.append('\n'.join(cleaned_lines))
        
        # 合并此批次文本并tokenize
        if cleaned_texts:
            batch_text = '\n'.join(cleaned_texts)
            batch_ids = enc.encode_ordinary(batch_text)
            
            for token_id in batch_ids:
                yield token_id
            
            # 进度显示
            if (batch_end - batch_start) % batch_size == 0 or batch_end == total_samples:
                print(f"  Processed: {batch_end}/{total_samples} samples")

# 使用生成器流式处理并直接保存到文件
def save_tokens_streaming(split_name, output_path):
    """
    流式处理token生成器，直接写入numpy二进制文件
    """
    print(f"\nTokenizing and saving {split_name} split...")
    
    # 为了提高写入效率，我们分块缓存tokens
    buffer = []
    buffer_size = 100000  # 缓存10万个token后再写入
    total_tokens = 0
    
    with open(output_path, 'wb') as f:
        for token_id in stream_and_tokenize(split_name):
            buffer.append(token_id)
            total_tokens += 1
            
            # 缓冲区满则写入
            if len(buffer) >= buffer_size:
                token_array = np.array(buffer, dtype=np.uint16)
                token_array.tofile(f)
                buffer = []
        
        # 写入剩余tokens
        if buffer:
            token_array = np.array(buffer, dtype=np.uint16)
            token_array.tofile(f)
    
    print(f"{split_name.upper()} tokens: {total_tokens:,}")
    return total_tokens

# 处理train和val分割
train_bin_path = os.path.join(data_dir, 'train.bin')
val_bin_path = os.path.join(data_dir, 'val.bin')

train_token_count = save_tokens_streaming('train', train_bin_path)
val_token_count = save_tokens_streaming('validation', val_bin_path)

# Save metadata
# meta = {
#     'vocab_size': vocab_size,
#     'tokenizer_name': 'gpt2',
# }
#
# meta_path = os.path.join(data_dir, 'meta.pkl')
# with open(meta_path, 'wb') as f:
#     pickle.dump(meta, f)
# print(f"Metadata file: {meta_path}")

print("\n✓ Data preparation with GPT-2 tokenizer completed successfully!")
print(f"Train file: {train_bin_path} ({os.path.getsize(train_bin_path) / 1024 / 1024:.2f} MB)")
print(f"Val file: {val_bin_path} ({os.path.getsize(val_bin_path) / 1024 / 1024:.2f} MB)")
print(f"Total tokens (train + val): {train_token_count + val_token_count:,}")
print(f"Memory-efficient streaming processing completed! No OOM issues. ✓")