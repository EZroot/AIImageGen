import os
import uuid
import torch
import logging
import threading
import queue
from dataclasses import dataclass, field

from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
from diffusers import FluxPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("flux_app.log"), logging.StreamHandler()]
)

# Flask app initialization
app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['IMAGE_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_images')
os.makedirs(app.config['IMAGE_SAVE_DIRECTORY'], exist_ok=True)

# Global model unload delay (in seconds)
MODEL_UNLOAD_DELAY = 5

# Specify your Flux model ID – you can change this to your desired checkpoint.
flux_model_id = "black-forest-labs/FLUX.1-dev"
logging.info("Loading Flux model...")

# Load the FluxPipeline with appropriate optimizations
flux_pipe = FluxPipeline.from_pretrained(flux_model_id, torch_dtype=torch.bfloat16)
# For moderate VRAM GPUs, enable sequential CPU offload
flux_pipe.enable_sequential_cpu_offload()
# Reduce memory usage for the VAE component
flux_pipe.vae.enable_slicing()
flux_pipe.vae.enable_tiling()
# Cast the model to float16 for efficiency
flux_pipe.to(torch.float16)

logging.info("Flux model loaded successfully.")

@dataclass
class Job:
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    width: int
    height: int
    generator: torch.Generator
    event: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default=None)

# Create a queue for jobs
job_queue = queue.Queue()

def worker():
    while True:
        job = job_queue.get()
        if job is None:
            break  # exit signal
        try:
            logging.info(f"Generating image with prompt: {job.prompt}")
            # Generate the image using the Flux pipeline
            result = flux_pipe(
                job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.num_inference_steps,
                generator=job.generator,
                output_type="pil",
                height=job.height,
                width=job.width,
            )
            image: Image.Image = result.images[0]
            
            # Save the generated image
            unique_filename = f"{uuid.uuid4().hex}.png"
            file_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], unique_filename)
            image.save(file_path)
            absolute_file_path = os.path.abspath(file_path)

            with app.app_context():
                image_url = url_for('get_image', filename=unique_filename, _external=True)
            
            logging.info(f"Image saved at: {file_path}")
            job.result = {
                "message": "Success",
                "image_url": image_url,
                "file_path": absolute_file_path,
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt
            }
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            job.result = {"error": str(e)}
        finally:
            # Optional: free up GPU memory and delay for unloading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            job.event.set()
            job_queue.task_done()

# Start the worker thread to process jobs sequentially
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """Serve generated images."""
    return send_from_directory(app.config['IMAGE_SAVE_DIRECTORY'], filename)

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Expects a JSON payload such as:
    {
        "prompt": "a futuristic cityscape",
        "negative_prompt": "low quality, blurry",
        "num_inference_steps": 20,
        "width": 1024,
        "height": 1024
    }
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 20)
    width = data.get("width", 1024)
    height = data.get("height", 1024)

    # Set a seed for reproducibility (optional)
    generator = torch.Generator("cpu").manual_seed(data.get("seed", 0))

    # Create a job and enqueue it
    job = Job(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator
    )
    job_queue.put(job)
    job.event.wait()  # Wait until the job is processed

    if job.result.get("error"):
        return jsonify({"error": job.result["error"]}), 500

    return jsonify(job.result), 200

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
