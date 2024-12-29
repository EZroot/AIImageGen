import os
import uuid
import torch
from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusion3Pipeline, FluxPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL
from peft import PeftModel  # Import PeftModel
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
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['SERVER_NAME'] = 'localhost:5000'  # For local development
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['IMAGE_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_images')
os.makedirs(app.config['IMAGE_SAVE_DIRECTORY'], exist_ok=True)

# Specify the model path
# local_model_path = './models/sd/2dnSD15_2.safetensors'
# local_model_path = './models/sdxldistilled/stableDiffusion3SD3_sd3Medium.safetensors'
# local_model_path = './models/sdxl1/noobaiXLNAIXL_vPred10Version.safetensors'
# local_model_path = './models/illustrious/waiNSFWIllustrious_v80.safetensors' #meh
# local_model_path = './models/illustrious/ntrMIXIllustriousXL_xiii.safetensors' #great!
# local_model_path = './models/illustrious/prefectiousXLNSFW_v10.safetensors' #good
# local_model_path = './models/pony/realismByStableYogi_ponyV3VAE.safetensors' # decent but only for realism
local_model_path = './models/pony/waiANINSFWPONYXL_v12.safetensors' #very cool!

# Check if the model file exists
if not os.path.exists(local_model_path):
    msg = f"Error: The path {local_model_path} does not exist."
    logging.error(msg)
    raise FileNotFoundError(msg)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model once
logging.info("Loading the model...")
pipe = StableDiffusionXLPipeline.from_single_file(
    local_model_path, 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# Load and setup Loras
# local_lora_path = './lora/add-detail-xl.safetensors'
# pipe.unet = PeftModel.from_pretrained(pipe.unet, local_lora_path)

pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")

# Load and setup custom VAE
custom_vae_path = './vae/sdxlVAE_sdxlVAE.safetensors'  # Ensure this path points to the directory containing the VAE files

# Load the custom VAE
custom_vae = AutoencoderKL.from_single_file(
    custom_vae_path,
    torch_dtype=torch.float16
).to(device)

# Assign the custom VAE to the pipeline
pipe.vae = custom_vae

#<lora:add-detail-xl:1> <lyco:add-detail-xl:1>

# USE THIS WITH 
# - noobaiXLNAIXL_vPred10Version (XL)
#scheduler_args = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)

#EULAR A Scheduler
# scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

pipe.enable_xformers_memory_efficient_attention()

# pipe = pipe.to(device)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

# Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

#Enabled xformers memory efficient attention
try:
    pipe.enable_xformers_memory_efficient_attention()
    logging.info("Xformers memory-efficient attention enabled.")
except Exception as e:
    logging.warning(f"Could not enable Xformers memory-efficient attention: {e}")

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
            lora_scale = 0.9
            logging.info(f"Generating image with prompt: {job.prompt}")
            result = pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.num_inference_steps,
                generator=job.generator,
                width=job.width,
                height=job.height,
                cross_attention_kwargs={"scale": lora_scale},
            )
            image = result.images[0]

            # Save the image
            unique_filename = f"{uuid.uuid4().hex}.png"
            file_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], unique_filename)
            image.save(file_path)

            logging.info(f"Image saved at: {file_path}")

            # Get absolute file path
            absolute_file_path = os.path.abspath(file_path)

            # Generate the URL for the saved image within an application context
            with app.app_context():
                image_url = url_for('get_image', filename=unique_filename, _external=True)

            # Store the result with both image_url and file_path
            job.result = {
                "message": "Success",
                "image_url": image_url,
                "file_path": absolute_file_path,
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

    # Respond with JSON containing the image URL and file path
    return jsonify({
        "message": job.result["message"],
        "image_url": job.result["image_url"],
        "file_path": job.result["file_path"],
        "prompt": job.result["prompt"],
        "negative_prompt": job.result["negative_prompt"]
    }), 200

if __name__ == '__main__':
    # Run on port 5000 by default, or specify another port if needed
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
