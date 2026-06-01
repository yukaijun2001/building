import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_expert=1):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False),
        ])

        self.gate = nn.Linear(input_dim, num_expert)

    def forward(self, x):
        """
        x: [batch, time_block, hidden_dim]  (LLaMA 输出)
        """
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        gates = self.gate(x)
        gates = torch.softmax(gates, dim=-1)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)

        moe_out = (gates.unsqueeze(-1) * expert_outs).sum(1)
        moe_out = moe_out.view(orig_shape)

        return moe_out
