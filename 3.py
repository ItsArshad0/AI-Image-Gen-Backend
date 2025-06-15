import os
import torch
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def find_weight_file(filename):
    # Check current directory
    if os.path.exists(filename):
        return filename
    # Check Downloads folder
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    download_path = os.path.join(downloads, filename)
    if os.path.exists(download_path):
        return download_path
    raise FileNotFoundError(f"{filename} not found in current directory or Downloads folder.")

# --- Begin functions from grounded_sam_demo.py ---
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    # Dummy transform for compatibility
    image = np.array(image_pil)
    return image_pil, image

def load_model(model_type=None, checkpoint=None, device="cpu"):
    # Only support SAM model loading
    if model_type is not None and checkpoint is not None:
        return sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    raise NotImplementedError("Only SAM model loading is supported without GroundingDINO.")

def get_sam_output(model, image, boxes, device="cpu"):
    boxes = boxes.astype(np.float32)  # Ensure boxes are float for arithmetic
    predictor = SamPredictor(model)
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    predictor.set_image(image)
    size = image.shape[:2][::-1]
    H, W = size[1], size[0]
    for i in range(boxes.shape[0]):
        boxes[i] = boxes[i] * np.array([W, H, W, H])
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]
    # Dummy transformed_boxes for demonstration
    transformed_boxes = boxes
    masks = np.zeros((boxes.shape[0], image.shape[0], image.shape[1]), dtype=bool)
    return masks

def draw_results(image, masks, boxes, phrases, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, label in zip(boxes, phrases):
        show_box(box, plt.gca(), label)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)
# --- End functions from grounded_sam_demo.py ---

def generate_image():
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (256, 256), color='white')
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 200, 200], outline='blue', width=5)
    d.text((70, 120), "Test", fill=(0, 0, 0))
    img.save('generated_image.jpg')
    print("Generated image saved as 'generated_image.jpg'.")
    return 'generated_image.jpg'

# Example usage without GroundingDINO
input_action = input("Type 'generate' to create a test image or enter the path to your image: ").strip()
if input_action.lower() == 'generate':
    image_path = generate_image()
else:
    image_path = input_action

if not os.path.exists(image_path):
    print(f"Image file '{image_path}' not found. Please provide a valid image path.")
    import sys
    sys.exit(1)
image_pil = Image.open(image_path).convert("RGB")

sam_model = load_model(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth"
)

# Dummy boxes and phrases for demonstration
boxes = np.array([[50, 50, 150, 150]], dtype=np.float32)
phrases = ["example object"]

masks = get_sam_output(
    model=sam_model,
    image=image_pil,
    boxes=boxes
)

draw_results(
    image=image_pil,
    masks=masks,
    boxes=boxes,
    phrases=phrases,
    output_path="output.jpg"
)