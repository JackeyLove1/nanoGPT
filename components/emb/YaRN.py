import torch
import torch.nn as nn
import math

class YaRNRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_position_embeddings: int = 2048, alpha: float = 1.0, beta: float = 32.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings  # 训练上下文 L
        self.alpha = alpha
        self.beta = beta
        # 预计算 inv_freq = theta_d = base ** (-2i / dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=1024)  # 初始化缓存

    def _compute_gamma(self, r: torch.Tensor) -> torch.Tensor:
        """计算斜坡函数 gamma(r)"""
        gamma = torch.zeros_like(r)
        mask_linear = (r >= self.alpha) & (r < self.beta)
        gamma[mask_linear] = (r[mask_linear] - self.alpha) / (self.beta - self.alpha)
        gamma[r >= self.beta] = 1.0
        return gamma

    def _compute_h_theta(self, theta: torch.Tensor, s: float, d_indices: torch.Tensor) -> torch.Tensor:
        """计算 h(theta_d)"""
        # 计算 lambda_d = pi * base * dim / d (d from 1 to dim/2)
        d = (d_indices + 1).float()  # d starts from 1
        lambda_d = math.pi * self.base * self.dim / d
        r = self.max_position_embeddings / lambda_d
        gamma = self._compute_gamma(r)
        h_theta = (1 - gamma) * theta * s + gamma * theta
        return h_theta

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device = None, dtype: torch.dtype = None):
        """动态计算 cos/sin，应用 YaRN"""
        self.seq_len = seq_len
        if seq_len > self.max_position_embeddings:
            s = seq_len / self.max_position_embeddings
        else:
            s = 1.0
        # 温度缩放
        temp_scale = 0.1 * math.log(s) + 1.0 if s > 1 else 1.0
        # d_indices for dimensions (0 to dim//2 -1)
        d_indices = torch.arange(0, self.dim // 2)
        # 修改 theta
        theta = self.inv_freq[d_indices]  # original theta_d
        h_theta = self._compute_h_theta(theta, s, d_indices)
        # 角度
        t = torch.arange(self.seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = t[:, None] * h_theta[None, :] * temp_scale  # 应用温度缩放
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # 扩展到 dim
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        # x: (bs, seq_len, dim) 或 (seq_len, dim)
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len != self.seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        cos = self.cos_cached[:seq_len].unsqueeze(0).expand(x.shape[0], -1, -1)
        sin = self.sin_cached[:seq_len].unsqueeze(0).expand(x.shape[0], -1, -1)
        # 应用旋转 (简化版，实际中用于 q/k)
        return self.apply_rotary_pos_emb(x, cos, sin)

    @staticmethod
    def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """应用旋转到 q (类似 k)"""
        q_embed = q * cos + torch.roll(q * sin, dims=-1, shifts=1)
        return q_embed  # 实际中需处理偶/奇维度对

# 示例使用
if __name__ == "__main__":
    dim = 128
    seq_len = 4096  # 扩展长度
    embedding = YaRNRotaryEmbedding(dim=dim, max_position_embeddings=2048)  # 训练 L=2048
    x = torch.randn(2, seq_len, dim)  # 模拟 q
    x_rot = embedding(x, seq_len=seq_len)
    print(x_rot.shape)  # torch.Size([2, 4096, 128])