import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from diffusers import StableDiffusionPipeline
from segment_anything import sam_model_registry, SamPredictor
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

def run_multimodal_pipeline(prompt, sam_checkpoint_path, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Image via Stable Diffusion
    print("🎨 Generating image with Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir="./models"
    ).to("cuda")

    image = pipe(prompt).images[0]
    image_path = os.path.join(output_dir, "generated_image.png")
    image.save(image_path)
    print(f"✅ Image saved: {image_path}")

    # 2. Region Segmentation with SAM (dummy box for now)
    print("📦 Applying SAM region segmentation...")
    image_np = np.array(image)
    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to("cuda")
    predictor = SamPredictor(sam_model)
    predictor.set_image(image_np)

    # Use a region in the center of the image as a dummy example
    H, W = image_np.shape[:2]
    center_box = np.array([[W//4, H//4, W//2, H//2]], dtype=np.float32)
    input_boxes = center_box.copy()
    input_boxes[:, 2] += input_boxes[:, 0]  # x1 = x0 + w
    input_boxes[:, 3] += input_boxes[:, 1]  # y1 = y0 + h

    masks = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=torch.tensor(input_boxes).to("cuda"),
        multimask_output=False
    )[0].cpu().numpy()

    # Visualize SAM results
    print("🖼️ Drawing SAM mask and box...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    for mask in masks:
        ax.imshow(mask[0], alpha=0.5, cmap='Reds')
    for box in input_boxes:
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='lime', facecolor='none', lw=2))
        ax.text(x0, y0, "example-object", color='lime', fontsize=10, backgroundcolor='black')
    ax.axis('off')
    sam_out = os.path.join(output_dir, "sam_segmentation.png")
    plt.savefig(sam_out, bbox_inches='tight', pad_inches=0)
    print(f"✅ SAM region visualization saved: {sam_out}")
    plt.close(fig)

    # 3. Semantic Segmentation with DeepLabV3
    print("🌈 Running DeepLabV3 semantic segmentation...")
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to("cuda")
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to("cuda")

    with torch.no_grad():
        seg_output = seg_model(input_tensor)["out"][0]
    seg_mask = seg_output.argmax(0).byte().cpu().numpy()

    seg_out_path = os.path.join(output_dir, "semantic_mask.png")
    plt.imsave(seg_out_path, seg_mask, cmap='tab20b')
    print(f"✅ Semantic segmentation saved: {seg_out_path}")

    print("🎯 Full multimodal pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal pipeline: Stable Diffusion + SAM + DeepLabV3")
    parser.add_argument('--prompt', type=str, required=False, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", help='Text prompt for image generation')
    parser.add_argument('--sam_ckpt', type=str, required=True, help='Path to SAM checkpoint (sam_vit_h_4b8939.pth)')
    parser.add_argument('--output_dir', type=str, default="./outputs", help='Directory to save outputs')
    args = parser.parse_args()
    run_multimodal_pipeline(args.prompt, args.sam_ckpt, args.output_dir)
