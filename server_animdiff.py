import os
import uuid
import torch
from flask import Flask, request, jsonify, send_from_directory, url_for
from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler, MotionAdapter, LCMScheduler, DDIMScheduler
from diffusers.utils import export_to_gif
import threading
import queue
from dataclasses import dataclass, field
import logging

# ------------------------- Configuration -------------------------
app = Flask(__name__)
app.config["SERVER_NAME"] = "127.0.0.1:5000"
app.config["PREFERRED_URL_SCHEME"] = "http"

app_root = os.path.dirname(os.path.abspath(__file__))
app.config['GIF_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_gifs')
os.makedirs(app.config['GIF_SAVE_DIRECTORY'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ------------------------- Model Loading -------------------------
logging.info("Loading AnimateDiff model and motion adapter...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# # 1. Load the motion adapter
# adapter = MotionAdapter.from_pretrained(
#     "guoyww/animatediff-motion-adapter-v1-5-2",
#     torch_dtype=torch.float16
# )
adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)

# 2. Load the Anything v3 base model and attach the adapter
# model_id = "cag/anything-v3-1"
model_id = "dreamlike-art/dreamlike-anime-1.0"
# model_id = "emilianJR/epiCRealism"
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16
)

# 3. Use the DPMSolverMultistepScheduler from the pipeline config
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

# 4. (Optional) Enable memory-efficient features
pipe.enable_vae_slicing()          # <--- Re-enabled
pipe.enable_model_cpu_offload()    # <--- Re-enabled

logging.info("AnimateDiff model loaded successfully.")

# ------------------------- Job Handling -------------------------
@dataclass
class Job:
    prompt: str
    negative_prompt: str
    num_frames: int
    guidance_scale: float
    num_inference_steps: int
    seed: int
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
            logging.info(f"Generating animation with prompt: {job.prompt}")

            # Matches the single-script approach, using a CPU-based generator
            generator = torch.Generator("cpu").manual_seed(job.seed)

            # 6. Generate the animation
            output = pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_frames=job.num_frames,
                guidance_scale=job.guidance_scale,
                num_inference_steps=job.num_inference_steps,
                generator=generator
            )

            # 7. Export frames to GIF (the first batch of frames is at index 0)
            frames = output.frames[0]
            unique_filename = f"{uuid.uuid4().hex}.gif"
            file_path = os.path.join(app.config['GIF_SAVE_DIRECTORY'], unique_filename)
            export_to_gif(frames, file_path)

            logging.info(f"Animation saved at: {file_path}")

            # Generate the URL for the saved GIF within an application context
            with app.app_context():
                gif_url = url_for('get_gif', filename=unique_filename, _external=True)

            job.result = {
                "message": "Success",
                "gif_url": gif_url,
                "file_path": os.path.abspath(file_path),
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt,
                "num_frames": job.num_frames,
                "guidance_scale": job.guidance_scale,
                "num_inference_steps": job.num_inference_steps,
                "seed": job.seed
            }
        except Exception as e:
            logging.error(f"Error generating animation: {e}")
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

# ------------------------- Routes -------------------------
@app.route('/gifs/<filename>', methods=['GET'])
def get_gif(filename):
    """
    Endpoint to serve generated GIF animations.
    Access via: /gifs/<filename>
    """
    return send_from_directory(app.config['GIF_SAVE_DIRECTORY'], filename)

@app.route('/generate', methods=['POST'])
def generate_animation():
    """
    POST JSON data to this endpoint, for example:
    {
        "prompt": "masterpiece, best quality, 1girl, solo, sitting",
        "negative_prompt": "lowres, bad anatomy, text",
        "num_frames": 16,
        "guidance_scale": 7.5,
        "num_inference_steps": 25,
        "seed": 42
    }
    """
    data = request.get_json(force=True)

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    negative_prompt = data.get("negative_prompt", "")
    num_frames = data.get("num_frames", 20)
    guidance_scale = data.get("guidance_scale", 7.5)
    num_inference_steps = data.get("num_inference_steps", 25)
    seed = data.get("seed", 42)

    # Clear CUDA cache (optional in some setups)
    torch.cuda.empty_cache()

    job = Job(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )

    job_queue.put(job)
    job.event.wait()

    if "error" in job.result:
        return jsonify({"error": job.result["error"]}), 500

    return jsonify(job.result), 200

# ------------------------- Main -------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
