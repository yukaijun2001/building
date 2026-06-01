import math

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, feature_size, device):
        super().__init__()
        self.feature_size = feature_size

        self.query_proj = nn.Linear(feature_size, feature_size)
        self.key_proj = nn.Linear(feature_size, feature_size)
        self.value_proj = nn.Linear(feature_size, feature_size)

        self.output_proj = nn.Linear(feature_size, 1)

        self.to(device)

    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_size)
        输出: (batch_size, 1, seq_len)
        """
        # 计算注意力权重
        queries = self.query_proj(x)
        keys = self.key_proj(x)  # (batch, seq_len, num_vars)
        values = self.value_proj(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.feature_size)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        return self.output_proj(output)


if __name__ == '__main__':
    t = torch.randn(256, 672, 7).to(0)
    attention = AttentionFusion(7, 0)
    a = attention(t)
    print("over")
