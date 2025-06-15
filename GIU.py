import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from fusion5 import run_multimodal_pipeline  


class ImageGenerationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Generation and Segmentation")

        # Prompt input
        self.prompt_label = tk.Label(root, text="Enter the main prompt:")
        self.prompt_label.pack()
        self.prompt_entry = tk.Entry(root, width=50)
        self.prompt_entry.pack()

        # Negative prompt input
        self.negative_prompt_label = tk.Label(root, text="Enter a negative prompt (optional):")
        self.negative_prompt_label.pack()
        self.negative_prompt_entry = tk.Entry(root, width=50)
        self.negative_prompt_entry.pack()

        # Inpaint prompt input
        self.inpaint_prompt_label = tk.Label(root, text="Enter the inpaint prompt (optional):")
        self.inpaint_prompt_label.pack()
        self.inpaint_prompt_entry = tk.Entry(root, width=50)
        self.inpaint_prompt_entry.pack()

        # SAM checkpoint file selection
        self.sampler_ckpt_label = tk.Label(root, text="Select SAM checkpoint file:")
        self.sampler_ckpt_label.pack()
        self.sampler_ckpt_button = tk.Button(root, text="Browse...", command=self.load_sam_checkpoint)
        self.sampler_ckpt_button.pack()
        self.sampler_ckpt_path = None

        # Run button
        self.run_button = tk.Button(root, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.pack()

        # Output message display
        self.output_text = tk.Text(root, height=10, width=60)
        self.output_text.pack()

    def load_sam_checkpoint(self):
        """Load SAM checkpoint file"""
        self.sampler_ckpt_path = filedialog.askopenfilename(
            title="Select SAM Checkpoint", filetypes=(("Checkpoint files", "*.pth"), ("All files", "*.*"))
        )
        if not self.sampler_ckpt_path:
            messagebox.showwarning("File not selected", "Please select a SAM checkpoint file.")
    
    def update_output(self, message):
        """Update output text field"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.yview(tk.END)
    
    def run_pipeline(self):
        """Run the image generation pipeline in a separate thread to avoid blocking the UI"""
        prompt = self.prompt_entry.get().strip()
        negative_prompt = self.negative_prompt_entry.get().strip() or None
        inpaint_prompt = self.inpaint_prompt_entry.get().strip() or None

        if not prompt:
            messagebox.showwarning("Missing Prompt", "Please enter a main prompt for image generation.")
            return

        if not self.sampler_ckpt_path:
            messagebox.showwarning("Missing SAM Checkpoint", "Please select a SAM checkpoint file.")
            return

        # Disable the button to avoid re-running while processing
        self.run_button.config(state=tk.DISABLED)
        self.update_output("Running pipeline...")

        # Run the pipeline in a separate thread to avoid blocking the UI
        threading.Thread(target=self.run_pipeline_thread, args=(prompt, negative_prompt, inpaint_prompt)).start()

    def run_pipeline_thread(self, prompt, negative_prompt, inpaint_prompt):
        """Handle the image generation and processing pipeline"""
        try:
            # Replace this with the actual call to your pipeline function
            # Assuming your script functions are available in the current scope
            # Call the pipeline (assuming main logic from your provided script)
            run_multimodal_pipeline(
                prompt=prompt,
                sam_checkpoint_path=self.sampler_ckpt_path,
                output_dir="./outputs",
                model_type="stable-diffusion",
                lora_path=None,
                inpaint_prompt=inpaint_prompt,
                negative_prompt=negative_prompt
            )
            self.update_output("Pipeline complete. Outputs saved.")
        except Exception as e:
            self.update_output(f"Error: {str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)

# Create the main window
root = tk.Tk()
app = ImageGenerationApp(root)
root.mainloop()
