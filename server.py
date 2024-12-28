import os
import uuid
import torch
from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
from diffusers import StableDiffusionPipeline
from colorama import Fore, Style, init
import threading
import queue
from dataclasses import dataclass, field
import logging

# Initialize colorama for colored output (optional)
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Create a Flask app
app = Flask(__name__)

# Configuration Settings
# Set SERVER_NAME to 'localhost:5000' for local development.
# Replace 'localhost:5000' with your domain or IP and port when deploying.
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['IMAGE_SAVE_DIRECTORY'] = 'generated_images'
os.makedirs(app.config['IMAGE_SAVE_DIRECTORY'], exist_ok=True)

# Specify the model path
local_model_path = './models/sd/2dnSD15_2.safetensors'
# local_model_path = './models/sdxl1/noobaiXLNAIXL_vPred10Version.safetensors'

# Check if the model file exists
if not os.path.exists(local_model_path):
    msg = f"Error: The path {local_model_path} does not exist."
    logging.error(msg)
    raise FileNotFoundError(msg)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model once
logging.info("Loading the model...")
pipe = StableDiffusionPipeline.from_single_file(
    local_model_path, 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe = pipe.to(device)

# Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()
pipe.enable_vae_tiling()

logging.info(f"Model loaded successfully on {device}.")

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
            break  # Exit signal
        try:
            logging.info(f"Generating image with prompt: {job.prompt}")
            result = pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.num_inference_steps,
                generator=job.generator,
                width=job.width,
                height=job.height
            )
            image = result.images[0]

            # Save the image
            unique_filename = f"{uuid.uuid4().hex}.png"
            file_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], unique_filename)
            image.save(file_path)

            logging.info(f"Image saved at: {file_path}")

            # Generate the URL for the saved image within an application context
            with app.app_context():
                image_url = url_for('get_image', filename=unique_filename, _external=True)

            # Store the result
            job.result = {
                "message": "Success",
                "image_url": image_url,
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt
            }
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            job.result = {
                "error": str(e)
            }
        finally:
            # Signal that the job is done
            job.event.set()
            job_queue.task_done()

# Start the worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint to serve generated images.
    Access via: /images/<filename>
    """
    return send_from_directory(app.config['IMAGE_SAVE_DIRECTORY'], filename)

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    POST JSON data to this endpoint, for example:
    {
        "prompt": "a scenic mountain landscape",
        "negative_prompt": "lowres, blurry",
        "num_inference_steps": 50,
        "width": 512,
        "height": 512
    }
    """
    data = request.get_json(force=True)

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Optional parameters
    neg_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 50)
    width = data.get("width", 512)
    height = data.get("height", 512)

    # You can also specify other params such as guidance_scale, seed, etc.
    # Example: guidance_scale = data.get("guidance_scale", 7.5)

    # Set a seed for reproducibility (optional)
    generator = torch.Generator(device=device).manual_seed(0)

    torch.cuda.empty_cache()

    # Create a Job instance
    job = Job(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator
    )

    # Put the job in the queue
    job_queue.put(job)

    # Wait for the job to be processed
    job.event.wait()

    # Check if there was an error
    if "error" in job.result:
        return jsonify({"error": job.result["error"]}), 500

    # Respond with JSON containing the image URL
    return jsonify(job.result), 200

if __name__ == '__main__':
    # Run on port 5000 by default, or specify another port if needed
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
