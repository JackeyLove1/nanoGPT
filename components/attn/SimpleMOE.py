"""Top-1 Mix Of Expert"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_hidden: int,
            dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.net = nn.Sequential(
            nn.Linear(self.d_model, self.d_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.d_hidden, self.d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x:torch.Tensor):
        return self.net(x)

class SimpleMoE(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_hidden: int,
            num_experts: int,
            top_k: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            Expert(d_model, d_hidden) for _ in range(num_experts)
        ])


    def forward(self, x:torch.Tensor):
        """x: [batch, d_model]"""
        assert len(x.shape) == 2, "x.shape != 2"
        assert x.shape[-1] == self.d_model, f"x.shape[-1] != {self.d_model}"

        B, D = x.shape

        gate_logits = self.gate(x) # (B, E)
        gate_scores = F.softmax(gate_logits, dim=-1) # (B, E)

        top_expert = gate_scores.argmax(dim=-1) # (B, )
        top_score = gate_scores.gather(1, top_expert.unsqueeze(-1)) # (B, 1)

        outputs = torch.zeros(B, D)
        for e in range(self.num_experts):
            # find the expert' task in batch
            idx = (top_expert == e).nonzero(as_tuple=False).squeeze(0) # (num_experts, )

            # no task
            if idx.numel() == 0:
                continue

            x_e = x[idx] # (num_experts, d_model)
            y_e = self.experts[e](x_e) # (num_experts, d_model)

            outputs[idx] = y_e

        outputs = outputs * top_score # (B, D)

        # 5) 负载均衡辅助信息（optional）
        # 看每个专家被选中的比例，用于构造 load balance loss
        density = torch.bincount(top_expert, minlength=self.num_experts).float() / B  # (E,)

        return outputs, {
            "gate_scores": gate_scores,  # (B, E)
            "top_expert": top_expert,  # (B,)
            "density": density,          # (E,)
        }

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 10
    d_model = 16
    d_hidden = 32
    num_experts = 4

    moe = SimpleMoE(d_model, d_hidden, num_experts)
    x = torch.randn(B, d_model)

    y, info = moe(x)
    print("out:", y.shape)               # (10, 16)
    print("top:", info["top_expert"])  # 每个样本被分到了哪个 expert
    print("density:", info["density"])   # 每个 expert 分到多少比例的样本