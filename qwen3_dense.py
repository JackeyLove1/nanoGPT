"""
Qwen3 Dense
References:
    1. https://ar5iv.labs.arxiv.org/html/2505.09388
    2. https://qwen.ai/blog?id=1e3fa5c2d4662af2855586055ad037ed9e555125

Qwen3-Dense = Llama2 + QK-Norm
"""


import math
import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from components.RMSNorm import RMSNormTorch
from components.LayerNorm import LayerNorm
from components.emb.RoPE import RoPE
from components.activation.GeGLU import GeGLU_FFN
from components.activation.SwiGLU import SwiGLU_FFN
from tool import repeat_kv


@dataclass
class Qwen3Config:
    block_size: int = 256  # max_sequence_length
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 4  # use group query attention, n_group = n_head // n_kv_head
    n_embd: int = 384  # d_model
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class CausalSelfAttention(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "n_embd is not the multiple of n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head is not the multiple of n_head"

        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = config.dropout
        self.bias = config.bias

        # position encoding
        self.rope = RoPE(self.head_dim, config.block_size)

        # TODO: merge 3 matmul to 1 matmul like: self.c_attn = nn.Linear(config.n_embd, self.n_head * self.head_dim + 2 * self.n_kv_head * self.head_dim, bias=config.bias)
        self.wq = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=self.bias)
        self.wk = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=self.bias)
        self.wv = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=self.bias)
        self.wo = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

        # QK_norm
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            mask = torch.full((1, 1, config.block_size, config.block_size), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq, dim)"""
        B, T, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (batch, seq, dim) -> (batch, seq, n_head * head_dim) -> (batch, seq, n_head, head_dim)
        xq: torch.Tensor = self.wq(x)
        xq = xq.contiguous().view(B, T, self.n_head, self.head_dim)

        # (batch, seq, dim) -> (batch, seq, n_kv_head * head_dim)
        xk, xv = self.wk(x), self.wv(x)
        # (batch, seq, n_kv_head * head_dim) -> (batch, seq, n_kv_head, head_dim)
        xk, xv = xk.contiguous().view(B, T, self.n_kv_head, self.head_dim), xv.contiguous().view(B, T, self.n_kv_head,
                                                                                                 self.head_dim)

        # (batch, seq, n_head, head_dim) -> (batch, seq, n_head, head_dim)
        xq, xk = self.rope(xq), self.rope(xk)

        # (batch, seq, n_kv_head, head_dim) -> (batch, seq, n_kv_head * n_rep, head_dim)
        xk, xv = repeat_kv(xk, self.n_rep), repeat_kv(xv, self.n_rep)

        # (batch, seq, n_head, head_dim) -> (batch, n_head, seq, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        # apply qk norm
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            # manual implementation of attention
            # attention: n_head = n_kv_head * n_rep
            # (batch, n_head, seq, head_dim) @ (batch,head_dim, n_head, seq) -> (batch, n_head, seq, seq)
            scores = (xq @ xk.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            assert hasattr(self, "mask"), "attention mask is not exist!"
            scores = scores + self.mask[:, :, :T, :T]  # mask future information
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            y = torch.matmul(scores, xv)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # (batch, n_head, seq, head_dim) -> (batch, seq, n_head, head_dim) -> (batch, seq, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.wo(y)
        y = self.resid_dropout(y)
        return y


class Block(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.norm_1 = RMSNormTorch(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNormTorch(config.n_embd)
        self.ffn = SwiGLU_FFN(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.ffn(self.norm_2(x))
        return x


class Qwen3Dense(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        norm = RMSNormTorch(config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm=norm,
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith("out_weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # absolute position encoding
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        # N 个 pre-norm block（每个 block 里自己先 norm 再 Attn/MLP）
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type='adamw'):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create optimizer based on optimizer_type
        optimizer_type = optimizer_type.lower()
        print(f"using optimizer: {optimizer_type}")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        # Detect GPU type and set peak flops (dense bfloat16 or equivalent)
        gpu_flops = {
            "A100": 312e12,
            "H100": 989e12,
            "V100": 125e12,  # FP16 equivalent
            "4090": 165.2e12,
            "RTX 4090": 165.2e12,
            "RTX 3080": 119e12,
            "RTX 3090": 142e12,
        }
        flops_promised = 119e12  # default to 3080
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).upper()
            for key, val in gpu_flops.items():
                if key in gpu_name:
                    flops_promised = val
                    break
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
