from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Import your pipeline function
from fusion5 import run_multimodal_pipeline

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML frontend

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    inpaint_prompt = data.get('inpaint_prompt', '')

    # Always use the fixed checkpoint path
    sam_checkpoint_path = "sam_vit_h_4b8939.pth"

    # Run the pipeline in a thread to avoid blocking
    def run_pipeline():
        run_multimodal_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            inpaint_prompt=inpaint_prompt,
            sam_checkpoint_path=sam_checkpoint_path
        )
    thread = threading.Thread(target=run_pipeline)
    thread.start()
    thread.join()  # Wait for completion (or remove for async)

    # Return output image paths (relative to static folder)
    return jsonify({
        "generated_image": "output/generated_image.png",
        "sam_segmentation": "output/sam_segmentation.png",
        "inpainted_image": "output/inpainted_image.png",
        "semantic_mask": "output/semantic_mask.png"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)