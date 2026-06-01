import torch
import torch.nn as nn


class AutoregressiveDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, pred_len, output_dim=96):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        # Decoder 层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出预测层
        self.fc_out = nn.Linear(d_model, output_dim)

        # 预测值 -> embedding
        self.value_to_emb = nn.Linear(output_dim, d_model)

        # 起始 token（可选，替代 memory[:, -1:, :]）
        # self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, memory):
        bs = memory.size(0)
        outputs = []

        # 用 start token（也可以用 memory[:, -1:, :]）
        # tgt = self.start_token.expand(bs, 1, -1)  # [bs, 1, d_model]
        tgt = memory[:, -1:, :]  # [bs, 1, d_model]

        for step in range(self.pred_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

            dec_out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [bs, step+1, d_model]

            last_token = dec_out[:, -1:, :]  # [bs, 1, d_model]
            pred = self.fc_out(last_token)  # [bs, 1, output_dim]
            outputs.append(pred)

            # 将预测转为 embedding，拼回 tgt
            pred_embed = self.value_to_emb(pred)  # [bs, 1, d_model]
            tgt = torch.cat([tgt, pred_embed], dim=1)

        outputs = torch.cat(outputs, dim=1)  # [bs, pred_len, output_dim]
        return outputs
