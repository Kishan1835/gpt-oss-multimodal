import torch
from model.projector import VisionToTextProjector

dummy_vision_embeds = torch.randn(1, 257, 1024)

projector = VisionToTextProjector(
    vision_dim=1024,
    text_dim=4096
)

output = projector(dummy_vision_embeds)

print("Projected shape:", output.shape)
