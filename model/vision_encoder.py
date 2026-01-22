import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPVisionEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).vision_model
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, images):
        """
        images: PIL Image or list of PIL Images
        returns: vision embeddings (batch_size, num_patches, hidden_dim)
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state
