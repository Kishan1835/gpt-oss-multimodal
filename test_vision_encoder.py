from PIL import Image
from model.vision_encoder import CLIPVisionEncoder

encoder = CLIPVisionEncoder(device="cpu")

image = Image.new("RGB", (224, 224), color="white")
embeddings = encoder.encode(image)

print("Embedding shape:", embeddings.shape)
