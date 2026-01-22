import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTOSSWrapper:
    def __init__(self, model_name="openai/gpt-oss-20b", device="cpu"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )

        self.model.to(self.device)
        self.model.eval()

    def get_hidden_size(self):
        return self.model.config.hidden_size
