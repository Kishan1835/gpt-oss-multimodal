import torch
import torch.nn as nn


class VisionToTextProjector(nn.Module):
    """
    Projects vision encoder embeddings into LLM embedding space.
    """

    def __init__(self, vision_dim=1024, text_dim=4096, hidden_dim=4096):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, vision_embeds):
        """
        vision_embeds: (batch_size, num_patches, vision_dim)
        returns: (batch_size, num_patches, text_dim)
        """
        return self.proj(vision_embeds)
