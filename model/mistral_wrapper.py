import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MistralWrapper:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cpu"
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )

        self.model.to(self.device)
        self.model.eval()

    def get_hidden_size(self):
        return self.model.config.hidden_size
