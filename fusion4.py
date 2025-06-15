import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import models, transforms
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from segment_anything import sam_model_registry, SamPredictor
import warnings
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)

def overlay_segmentation(image_np, sam_masks, deeplab_mask):
    overlay = Image.fromarray(image_np.copy()).convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Draw SAM masks in semi-transparent red
    for mask in sam_masks:
        mask_img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8)).resize(overlay.size)
        red_overlay = Image.new("RGBA", overlay.size, (255, 0, 0, 80))
        overlay.paste(red_overlay, mask=mask_img)

    # Draw semantic segmentation (e.g., pink)
    seg_img = Image.fromarray((deeplab_mask > 0).astype(np.uint8) * 255).resize(overlay.size)
    pink_overlay = Image.new("RGBA", overlay.size, (255, 105, 180, 80))
    overlay.paste(pink_overlay, mask=seg_img)

    return overlay

def generate_image(prompt, model_type="stable-diffusion", lora_path=None):
    if model_type == "stable-diffusion":
        pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="./models"
        ).to("cuda")
        image = pipe(prompt).images[0]
        image_path = "./outputs/generated_image.png"
        image.save(image_path)
        return image_path

    elif model_type == "flux-dev-lora":
        pipe = DiffusionPipeline.from_pretrained("flux-ai/flux1-dev", torch_dtype=torch.float16).to("cuda")
        if lora_path:
            pipe.load_lora_weights(lora_path)
        image = pipe(prompt).images[0]
        image_path = "./outputs/generated_image.png"
        image.save(image_path)
        return image_path

    elif model_type == "flux-dev-cogniwerk":
        print("🌐 Requesting image from Cogniwerk.ai API...")
        try:
            response = requests.post("https://cogniwerk.ai/run-model/flux-dev", json={"prompt": prompt})
            print(f"Cogniwerk.ai status code: {response.status_code}")
            print(f"Cogniwerk.ai response text: {response.text}")
            response.raise_for_status()
            try:
                image_url = response.json().get("image_url")
            except Exception as json_err:
                print(f"❌ JSON decode error: {json_err}")
                raise RuntimeError(f"Cogniwerk.ai did not return valid JSON. Response: {response.text}")
            if not image_url:
                raise RuntimeError("Failed to get image URL from Cogniwerk.ai response.")
            image_data = requests.get(image_url).content
            image_path = "./outputs/generated_image.png"
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"✅ Image downloaded from Cogniwerk.ai: {image_path}")
            return image_path
        except Exception as e:
            print(f"❌ Cogniwerk.ai image generation failed: {e}")
            raise
    else:
        raise ValueError("Unsupported model type")

def run_multimodal_pipeline(prompt, sam_checkpoint_path, output_dir="./outputs", model_type="stable-diffusion", lora_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate image
    print("🎨 Generating image...")
    image_path = generate_image(prompt, model_type=model_type, lora_path=lora_path)
    image = Image.open(image_path).convert("RGB")

    # Step 2: Region Segmentation (SAM)
    print("📦 Running SAM on selected box...")
    image_np = np.array(image)
    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to("cuda")
    predictor = SamPredictor(sam_model)
    predictor.set_image(image_np)

    H, W = image_np.shape[:2]
    box = np.array([[W // 4, H // 4, W // 2, H // 2]], dtype=np.float32)
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]

    masks = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=torch.tensor(box).to("cuda"),
        multimask_output=False
    )[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    for mask in masks:
        ax.imshow(mask[0], alpha=0.5, cmap='Reds')
    for b in box:
        x0, y0, x1, y1 = b
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='lime', facecolor='none', lw=2))
        ax.text(x0, y0, "example-object", color='lime', fontsize=10, backgroundcolor='black')
    ax.axis('off')
    sam_out = os.path.join(output_dir, "sam_segmentation.png")
    plt.savefig(sam_out, bbox_inches='tight', pad_inches=0)
    print(f"✅ SAM mask saved: {sam_out}")

    # Step 3: Semantic Segmentation (DeepLabV3)
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
    print(f"✅ Semantic mask saved: {seg_out_path}")

    # Step 4: Merge all masks
    print("🧠 Merging SAM + semantic masks...")
    merged = overlay_segmentation(image_np, masks, seg_mask)
    merged.save(os.path.join(output_dir, "merged_overlay.png"))
    print("✅ Final overlay saved.")

def main():
    parser = argparse.ArgumentParser(description="Multimodal Region Understanding Pipeline")
    parser.add_argument("--prompt", type=str, default="Futuristic city skyline at sunset", help="Text prompt")
    parser.add_argument("--sam_ckpt", type=str, required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output folder")
    parser.add_argument("--model_type", type=str, default="stable-diffusion", choices=["stable-diffusion", "flux-dev-lora", "flux-dev-cogniwerk"], help="Model to generate image")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA weights if using flux-dev-lora")
    args = parser.parse_args()

    run_multimodal_pipeline(
        prompt=args.prompt,
        sam_checkpoint_path=args.sam_ckpt,
        output_dir=args.output_dir,
        model_type=args.model_type,
        lora_path=args.lora
    )

if __name__ == "__main__":
    main()
