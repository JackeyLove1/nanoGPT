"""
Prepare the WikiText-103 dataset using GPT-2 BPE tokenizer.
- Uses OpenAI's tiktoken library for fast tokenization
- Save train.bin, val.bin containing the token ids, and meta.pkl containing metadata
"""
import os
import pickle
import numpy as np
from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Import GPT-2 tokenizer
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    os.system("pip install tiktoken")
    import tiktoken

# Get the directory path
data_dir = os.path.dirname(__file__)

# Initialize GPT-2 tokenizer
print("Loading GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # GPT-2 vocab size: 50257
print(f"GPT-2 vocab size: {vocab_size}")

# Download the WikiText-103-raw-v1 dataset
print("\nLoading WikiText-103-raw-v1 dataset...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

# Combine train and validation data
print("Combining train and validation splits...")
train_text = '\n'.join(ds['train']['text'])
val_text = '\n'.join(ds['validation']['text'])

# Remove lines with only "= " headers (common in WikiText)
train_text = '\n'.join([line for line in train_text.split('\n') if line.strip() and not line.strip().startswith('=')])
val_text = '\n'.join([line for line in val_text.split('\n') if line.strip() and not line.strip().startswith('=')])

print(f"\nTrain text length: {len(train_text):,} characters")
print(f"Val text length: {len(val_text):,} characters")

# Tokenize using GPT-2 tokenizer
print("\nTokenizing train split...")
train_ids = enc.encode_ordinary(train_text)
print(f"Train tokens: {len(train_ids):,}")

print("Tokenizing val split...")
val_ids = enc.encode_ordinary(val_text)
print(f"Val tokens: {len(val_ids):,}")

# Convert to numpy arrays with uint16 dtype
# (Note: uint16 can handle values up to 65535, GPT-2 vocab is 50257, so safe)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save to binary files
print("\nSaving to binary files...")
train_bin_path = os.path.join(data_dir, 'train.bin')
val_bin_path = os.path.join(data_dir, 'val.bin')

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)
print(f"Train file: {train_bin_path} ({os.path.getsize(train_bin_path) / 1024 / 1024:.2f} MB)")
print(f"Val file: {val_bin_path} ({os.path.getsize(val_bin_path) / 1024 / 1024:.2f} MB)")

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
print(f"Compression ratio: 1 token ≈ {len(train_text) / len(train_ids):.2f} characters")