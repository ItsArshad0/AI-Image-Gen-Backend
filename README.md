# AI-Image-Gen-Backend

A multimodal pipeline for generative AI image creation, region-based editing, and semantic segmentation using Stable Diffusion, SAM, and DeepLabV3.

---

## Features
- **Text-to-Image Generation**: Uses Stable Diffusion for high-quality image synthesis from prompts.
- **Manual Region Selection**: Select any region on the generated image for targeted editing.
- **Inpainting**: Edit the selected region with a custom prompt using Stable Diffusion Inpainting.
- **Negative Prompt Support**: Specify what you want to avoid in the generated image.
- **Semantic Segmentation**: DeepLabV3 for additional region understanding.
- **All outputs saved in the `output` folder.**

---

## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- [PyTorch](https://pytorch.org/), [diffusers](https://github.com/huggingface/diffusers), [segment-anything](https://github.com/facebookresearch/segment-anything), [torchvision], [matplotlib], [Pillow], [requests]

Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

---

## Model Weights (DO NOT PUSH MODELS)
- **Stable Diffusion**: Downloaded automatically from Hugging Face Hub.
- **SAM (Segment Anything Model)**: Download `sam_vit_h_4b8939.pth` from the [official Meta AI repo](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it in your project root.
- **Stable Diffusion Inpainting**: Downloaded automatically from Hugging Face Hub (`stabilityai/stable-diffusion-2-inpainting`).

---

## Usage
1. **Run the script:**
   ```bash
   python fusion5.py
   ```
2. **Follow the prompts:**
   - Enter your main prompt (e.g., "A futuristic city at dusk with glowing windows")
   - Enter a negative prompt (optional, e.g., "no people, no cars")
   - Enter an inpaint prompt for the selected region (optional, e.g., "Make the windows glow with a warm orange light")
3. **Select a region:**
   - A window will open with the generated image. Draw a box around the region you want to edit and close the window.
4. **Outputs:**
   - All results (generated image, masks, inpainted image, overlays) are saved in the `output` folder.

---

## Notes
- **Do not commit or push model weights.** Only reference the official model repositories.
- For best results, use a CUDA GPU and ensure all dependencies are up to date.
- If you encounter errors with model downloads, check your Hugging Face authentication and internet connection.

---

## References
- [Stable Diffusion (Hugging Face)](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [Stable Diffusion Inpainting (Hugging Face)](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth&ved=2ahUKEwiNnJjl-fKNAxWn1DgGHc5RCQMQFnoECAkQAQ&usg=AOvVaw2HDPrdVCaTaoOVanY7WKxU(Path File Download)
- [DeepLabV3 (torchvision)](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html)

---

## License
This project is for research and educational purposes. Please respect the licenses of all referenced models and libraries.
