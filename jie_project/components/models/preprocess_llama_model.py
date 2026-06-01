import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer


class PreprocessLlamaModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs.gpu
        print(self.device)

        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.llama_tokenizer.eos_token:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.llama_tokenizer.add_special_tokens({'pad_token': pad_token})
            self.llama_tokenizer.pad_token = pad_token

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.llama_tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result

    def forecast(self, x_mark_enc):
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        # x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        x_mark_enc = self.tokenizer(x_mark_enc)
        text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)
