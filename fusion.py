from diffusers import StableDiffusionPipeline
import torch
 

# Model you want to use
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

# Load model with GPU acceleration and caching
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    cache_dir="./models"  # Local model cache
)

# Send to GPU
pipe = pipe.to("cuda")

# Generate image
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

# Save output
image.save("output.png")
print("✅ Image saved as output.png")