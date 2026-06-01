import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from ..layers import MLP, MoE, MIMO, AutoregressiveDecoder


class TimesFusionLlama(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.token_len = self.seq_len - configs.label_len
        self.time_future = (configs.label_len + configs.pred_len) // self.token_len
        self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
        )
        self.hidden_dim_of_llama = 2048

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        self.num_experts = configs.num_experts
        self.lambda_reg = configs.lambda_reg
        self.pad_len = (self.token_len - self.seq_len % self.token_len) % self.token_len

        # self.extend_encoder = nn.Linear(self.token_len * 7, self.hidden_dim_of_llama)
        self.fusion_gate = nn.Parameter(torch.zeros([]))
        self.moe = MoE(self.hidden_dim_of_llama, self.hidden_dim_of_llama)

        if configs.mlp_hidden_layers == 0:
            print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len * 7, self.hidden_dim_of_llama,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    """
    多变量Patching
    """
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = x_enc.permute(0, 2, 1)
        if self.pad_len > 0:
            x_enc = F.pad(x_enc, (0, self.pad_len))
        x_enc = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        bs, _, times_block, _ = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3)
        x_enc = x_enc.reshape(bs, times_block, -1)
        x_enc = self.encoder(x_enc)

        alpha = torch.sigmoid(self.fusion_gate)
        times_embeds = alpha * x_enc + (1 - alpha) * x_mark_enc

        outputs = self.llama.model(inputs_embeds=times_embeds)[0]

        # MoE
        outputs = self.moe(outputs)

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, 1, -1)[:, :, :self.seq_len]
        dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
