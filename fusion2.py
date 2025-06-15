from diffusers import StableDiffusionPipeline
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Load Stable Diffusion model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir="./models"
)
pipe = pipe.to("cuda")

# Generate image
prompt = "a dog with a tie, 8k"
image = pipe(prompt).images[0]
image.save("output.png")
print("✅ Image saved as output.png")

# --- Segmentation Part ---

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Load DeepLabV3 segmentation model
seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to("cuda")
with torch.no_grad():
    output = seg_model(input_tensor.to("cuda"))["out"][0]
segmentation = output.argmax(0).byte().cpu().numpy()

# Visualize and save segmentation
plt.imsave("segmentation_mask.png", segmentation, cmap='tab20b')
print("✅ Segmentation mask saved as segmentation_mask.png")