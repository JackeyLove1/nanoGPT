"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")


from transformers import GPT2TokenizerFast
print("use gpt2 tokenizer")
model_name = 'gpt2'

from transformers import PreTrainedTokenizerFast, AutoTokenizer
tokenizer_json_path = os.path.join(os.path.dirname(__file__), 'tokenizer.json')
print(f"tokenizer json path:{tokenizer_json_path}")

if os.path.exists(tokenizer_json_path):
    print("use local tokenizer json file")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
else:
    from transformers import GPT2TokenizerFast
    model_name = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
train_ids = tokenizer(train_data, return_tensors='np')["input_ids"][0].tolist()
val_ids = tokenizer(val_data, return_tensors='np')["input_ids"][0].tolist()


print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


