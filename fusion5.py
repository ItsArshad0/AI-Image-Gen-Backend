import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import models, transforms
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DiffusionPipeline
from segment_anything import sam_model_registry, SamPredictor
import warnings
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)

def overlay_segmentation(image_np, sam_masks, deeplab_mask):
    overlay = Image.fromarray(image_np.copy()).convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    for mask in sam_masks:
        mask_img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8)).resize(overlay.size)
        red_overlay = Image.new("RGBA", overlay.size, (255, 0, 0, 80))
        overlay.paste(red_overlay, mask=mask_img)

    seg_img = Image.fromarray((deeplab_mask > 0).astype(np.uint8) * 255).resize(overlay.size)
    pink_overlay = Image.new("RGBA", overlay.size, (255, 105, 180, 80))
    overlay.paste(pink_overlay, mask=seg_img)

    return overlay

def generate_image(prompt, model_type="stable-diffusion", lora_path=None, negative_prompt=None):
    if model_type == "stable-diffusion":
        pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="./models"
        ).to("cuda")
        image = pipe(prompt, negative_prompt=negative_prompt).images[0] if negative_prompt else pipe(prompt).images[0]
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
            response.raise_for_status()
            image_url = response.json().get("image_url")
            if not image_url:
                raise RuntimeError("Failed to get image URL from Cogniwerk.ai response.")
            image_data = requests.get(image_url).content
            image_path = "./outputs/generated_image.png"
            with open(image_path, "wb") as f:
                f.write(image_data)
            return image_path
        except Exception as e:
            raise RuntimeError(f"Cogniwerk.ai image generation failed: {e}")
    else:
        raise ValueError("Unsupported model type")

def load_inpaint_pipeline():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None
    return pipe

def prepare_inpaint_inputs(original_image, sam_mask):
    binary_mask = (sam_mask.squeeze() > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary_mask).convert("L").resize(original_image.size)
    image = original_image.resize(mask_image.size).convert("RGB")
    return image, mask_image

def select_box_on_image(image_np):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    box_coords = []
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        box_coords.clear()
        box_coords.append([min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)])
        plt.close()
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.set_title('Draw a box around the region to edit, then close the window')
    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    plt.show()
    if box_coords:
        x, y, w, h = box_coords[0]
        return np.array([[x, y, w, h]], dtype=np.float32)
    else:
        raise RuntimeError('No box selected!')

def run_multimodal_pipeline(prompt, sam_checkpoint_path, output_dir="./outputs", model_type="stable-diffusion", lora_path=None, inpaint_prompt=None, negative_prompt=None):
    os.makedirs(output_dir, exist_ok=True)

    print("🎨 Generating image...")
    image_path = generate_image(prompt, model_type=model_type, lora_path=lora_path, negative_prompt=negative_prompt)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    print("📦 Select region for SAM...")
    box = select_box_on_image(image_np)
    # Convert box from (x, y, w, h) to (x0, y0, x1, y1)
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]

    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to("cuda")
    predictor = SamPredictor(sam_model)
    predictor.set_image(image_np)
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

    print("🎯 Running inpainting with SAM mask...")
    sam_single_mask = masks[0][0]
    inpaint_image, inpaint_mask = prepare_inpaint_inputs(image, sam_single_mask)
    inpaint_prompt_final = inpaint_prompt if inpaint_prompt is not None else prompt + ", glowing blue chest panel"
    inpaint_pipe = load_inpaint_pipeline()
    inpaint_result = inpaint_pipe(prompt=inpaint_prompt_final, image=inpaint_image, mask_image=inpaint_mask).images[0]
    inpaint_out_path = os.path.join(output_dir, "inpainted_image.png")
    inpaint_result.save(inpaint_out_path)

    print("🌈 Running DeepLabV3 semantic segmentation...")
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to("cuda")
    seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to("cuda")
    with torch.no_grad():
        seg_output = seg_model(input_tensor)["out"][0]
    seg_mask = seg_output.argmax(0).byte().cpu().numpy()

    seg_out_path = os.path.join(output_dir, "semantic_mask.png")
    plt.imsave(seg_out_path, seg_mask, cmap='tab20b')

    print("🧠 Merging SAM + semantic masks...")
    merged = overlay_segmentation(image_np, masks, seg_mask)
    merged.save(os.path.join(output_dir, "merged_overlay.png"))

    print("✅ Pipeline complete. All outputs saved.")

def main():
    # Prompt user for input values
    prompt = input("Enter the main prompt for image generation: ").strip()
    negative_prompt = input("Enter a negative prompt (what to avoid, or leave blank): ").strip() or None
    inpaint_prompt = input("Enter the inpaint prompt (region edit, or leave blank to use main prompt): ").strip() or None

    sam_ckpt_path = "sam_vit_h_4b8939.pth"
    output_dir = "output"
    model_type = "stable-diffusion"

    run_multimodal_pipeline(
        prompt=prompt,
        sam_checkpoint_path=sam_ckpt_path,
        output_dir=output_dir,
        model_type=model_type,
        lora_path=None,
        inpaint_prompt=inpaint_prompt,
        negative_prompt=negative_prompt
    )

if __name__ == "__main__":
    main()