import torch
import torch.nn as nn


class ProbFusion(nn.Module):
    def __init__(self, seq_len, feature_size, device):
        super().__init__()

        self.prob_proj = nn.Parameter(torch.randn(seq_len, feature_size))

        self.to(device)

    def forward(self, x):
        """
        x: (batch_size, seq_len, feature_size)
        输出: (batch_size, 1, seq_len)
        """
        prob_matrix = torch.softmax(self.prob_proj, dim=-1)
        output = torch.einsum("ijk, jk->ijk", x, prob_matrix).sum(dim=-1, keepdim=True)
        return output


if __name__ == '__main__':
    # t = torch.randn(256, 672, 7).to(0)
    t = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]]).to(0)
    p = ProbFusion(3, 4, 0)
    p(t)
    print(1222)