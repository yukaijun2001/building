from __future__ import annotations

import math

import torch
from torch import nn


class GatedAttention(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) / math.sqrt(hidden_size))
        self.score = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.score(states))
        logits = torch.matmul(energy, self.query)
        attn = torch.softmax(logits, dim=1)
        context = torch.sum(states * attn.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.gate(context))
        context = self.dropout(context * gate)
        return context, attn, gate
