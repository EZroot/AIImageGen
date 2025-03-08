import os
import uuid
import torch
import logging
import threading
import queue
from time import sleep
from dataclasses import dataclass, field
from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    StableVideoDiffusionPipeline  # Video pipeline import
)
from diffusers.utils import export_to_video, load_image
from peft import PeftModel  # if needed
from colorama import init

# Initialize colorama (optional)
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Create Flask app and config
app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['IMAGE_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_images')
os.makedirs(app.config['IMAGE_SAVE_DIRECTORY'], exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Load the image generation pipeline (StableDiffusionXLPipeline)
local_model_path = './models/illustrious/waiNSFWIllustrious_v110.safetensors'
if not os.path.exists(local_model_path):
    msg = f"Error: The path {local_model_path} does not exist."
    logging.error(msg)
    raise FileNotFoundError(msg)

logging.info("Loading StableDiffusionXLPipeline model...")
sd_pipe = StableDiffusionXLPipeline.from_single_file(
    local_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe.scheduler = scheduler

try:
    sd_pipe.enable_xformers_memory_efficient_attention()
    logging.info("Xformers memory-efficient attention enabled for image gen.")
except Exception as e:
    logging.warning(f"Could not enable Xformers attention: {e}")

sd_pipe.enable_model_cpu_offload()
sd_pipe.enable_sequential_cpu_offload()
sd_pipe.enable_attention_slicing()
sd_pipe.enable_vae_slicing()
sd_pipe.enable_vae_tiling()

logging.info(f"StableDiffusionXLPipeline model loaded successfully on {device}.")

# ------------------------------
# Load the video generation pipeline (StableVideoDiffusionPipeline)
logging.info("Loading StableVideoDiffusionPipeline model...")
video_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
)
video_pipe.to(device)
# Reduce memory requirements
video_pipe.enable_model_cpu_offload()

logging.info("StableVideoDiffusionPipeline loaded successfully.")

# ------------------------------
# Job structure and worker for asynchronous image generation (if desired)
@dataclass
class Job:
    prompt: str
    negative_prompt: str
    num_inference_steps: int
    generator: torch.Generator
    event: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default=None)

job_queue = queue.Queue()

def worker():
    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            guidance_scale = 6
            logging.info(f"Generating image with prompt: {job.prompt}")
            # Force resolution to 1024x1024 here as well
            result = sd_pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_inference_steps=job.num_inference_steps,
                generator=job.generator,
                guidance_scale=guidance_scale,
                width=1024,
                height=1024,
            )
            image = result.images[0]
            unique_filename = f"{uuid.uuid4().hex}.png"
            file_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], unique_filename)
            image.save(file_path)
            logging.info(f"Image saved at: {file_path}")
            absolute_file_path = os.path.abspath(file_path)
            with app.app_context():
                image_url = url_for('get_image', filename=unique_filename, _external=True)
            job.result = {
                "message": "Success",
                "image_url": image_url,
                "file_path": absolute_file_path,
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt,
                "image": image  # include the PIL image for downstream video generation
            }
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            job.result = {"error": str(e)}
        finally:
            job.event.set()
            job_queue.task_done()

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(app.config['IMAGE_SAVE_DIRECTORY'], filename)

# ------------------------------
# Single endpoint: /generate
# This endpoint generates an image and then processes that image into a video.
@app.route('/generate', methods=['POST'])
def generate():
    """
    Expects a JSON payload such as:
    {
      "prompt": "a person standing on a busy street",
      "negative_prompt": "low quality, blurry",
      "num_inference_steps": 50,
      "video_fps": 7,           # playback fps for the exported video
      "num_frames": 10,         # number of frames to generate
      "motion_prompt": "walking with a steady pace"  # not used in video gen
    }
    The image is generated using the StableDiffusionXLPipeline.
    The video is then generated from the base image using the StableVideoDiffusionPipeline,
    which does not require a prompt.
    Resolution is forced to 1024x1024 for image generation.
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Retrieve the negative prompt and force-add unwanted tokens.
    base_neg_prompt = data.get("negative_prompt", "")
    forced_negatives = "inconsistent motion, blurry, jittery, distorted"
    if base_neg_prompt:
        neg_prompt = f"{base_neg_prompt}, {forced_negatives}"
    else:
        neg_prompt = forced_negatives

    num_inference_steps = data.get("num_inference_steps", 50)
    # Force resolution to 1024x1024 regardless of payload
    width = 1024
    height = 1024
    video_fps = data.get("video_fps", 7)
    num_frames = data.get("num_frames", 10)
    motion_prompt = data.get("motion_prompt", "")  # Optional; not used in video generation

    # Generate base image using the StableDiffusionXLPipeline
    generator = torch.Generator(device=device).manual_seed(0)
    try:
        sd_result = sd_pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=6,
            width=width,
            height=height,
        )
        base_image = sd_result.images[0]
    except Exception as e:
        logging.error(f"Error generating base image: {e}")
        return jsonify({"error": str(e)}), 500

    # Save the base image (optional)
    image_filename = f"{uuid.uuid4().hex}_base.png"
    image_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], image_filename)
    base_image.save(image_path)
    logging.info(f"Base image saved at: {image_path}")

    # Generate video from the base image using StableVideoDiffusionPipeline.
    # Note: This pipeline does not take a prompt.
    try:
        video_output = video_pipe(
            image=base_image,
            decode_chunk_size=8,
            generator=generator,
            num_frames=num_frames
        )
    except Exception as e:
        logging.error(f"Error generating video: {e}")
        return jsonify({"error": str(e)}), 500

    # Export the video as an MP4 file using the generated frames.
    video_filename = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(app.config['IMAGE_SAVE_DIRECTORY'], video_filename)
    try:
        export_to_video(video_output.frames[0], video_path, fps=video_fps)
    except Exception as e:
        logging.error(f"Error exporting video: {e}")
        return jsonify({"error": str(e)}), 500

    with app.app_context():
        video_url = url_for('get_image', filename=video_filename, _external=True)
        image_url = url_for('get_image', filename=os.path.basename(image_path), _external=True)
    logging.info(f"Video saved at: {video_path}")

    return jsonify({
        "message": "Success",
        "image_url": image_url,
        "video_url": video_url,
        "image_path": os.path.abspath(image_path),
        "video_path": os.path.abspath(video_path),
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "motion_prompt": motion_prompt
    }), 200

# ------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
