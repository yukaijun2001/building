import torch
import torch.nn as nn


class MIMO(nn.Module):
    """
    Learnable Queries + Cross-Attention
    Learnable Queries + 上下文加成
    """
    def __init__(self, d_model=2048, token_len=96, time_future=8, mode="context"):
        super().__init__()
        assert mode in ["context", "cross_attention"], "mode must be context or cross_attention"
        self.mode = mode
        self.token_len = token_len
        self.time_future = time_future

        # 未来 queries (learnable)
        self.future_queries = nn.Parameter(torch.randn(time_future, d_model))

        # 解码到时间序列 (96 steps)
        self.decoder = nn.Linear(d_model, token_len)

        if self.mode == "cross_attention":
            # cross-attention 模块
            self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

    def forward(self, encoder_out):
        """
        encoder_out: (B, num_blocks, d_model)  历史块编码后的输出
        return: (B, 1, num_future, token)
        """
        B = encoder_out.size(0)

        # 生成未来 queries
        queries = self.future_queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_future, d_model)

        if self.mode == "context":
            # -------- 上下文加成 --------
            context = encoder_out.mean(dim=1)  # (B, d_model)
            queries = queries + context.unsqueeze(1)  # (B, num_future, d_model)

        elif self.mode == "cross_attention":
            # -------- Cross-Attention --------
            # Q = queries, K=V=encoder_out
            queries, _ = self.cross_attn(query=queries, key=encoder_out, value=encoder_out)
            # (B, num_future, d_model)

        # 解码
        out = self.decoder(queries)   # (B, num_future, block_len)
        # out = out.unsqueeze(1)        # (B, 1, num_future, block_len)
        return out
