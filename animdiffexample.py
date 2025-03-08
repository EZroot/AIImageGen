import os
import uuid
import torch
from flask import Flask, request, jsonify, send_from_directory, url_for
from diffusers import AnimateDiffPipeline, DPMSolverMultistepScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import threading
import queue
from dataclasses import dataclass, field
import logging

# ------------------------- Configuration -------------------------
app = Flask(__name__)

# 1) Tell Flask how to build external URLs outside request context
app.config["SERVER_NAME"] = "127.0.0.1:5000"
app.config["PREFERRED_URL_SCHEME"] = "http"

app_root = os.path.dirname(os.path.abspath(__file__))
app.config['GIF_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_gifs')
os.makedirs(app.config['GIF_SAVE_DIRECTORY'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logging.info("Loading AnimateDiff model and motion adapter...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load the pipeline
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float16
)
model_id = "cag/anything-v3-1"
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

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

job_queue = queue.Queue()

def worker():
    while True:
        job = job_queue.get()
        if job is None:
            break

        try:
            logging.info(f"Generating animation with prompt: {job.prompt}")
            generator = torch.Generator("cpu").manual_seed(job.seed)

            output = pipe(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                num_frames=job.num_frames,
                guidance_scale=job.guidance_scale,
                num_inference_steps=job.num_inference_steps,
                generator=generator
            )

            frames = output.frames[0]
            unique_filename = f"{uuid.uuid4().hex}.gif"
            file_path = os.path.join(app.config['GIF_SAVE_DIRECTORY'], unique_filename)
            export_to_gif(frames, file_path)
            logging.info(f"Animation saved at: {file_path}")

            # 3) Build the URL inside an application context:
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
            job.result = {"error": str(e)}
        finally:
            job.event.set()
            job_queue.task_done()

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.route('/gifs/<filename>', methods=['GET'])
def get_gif(filename):
    return send_from_directory(app.config['GIF_SAVE_DIRECTORY'], filename)

@app.route('/generate', methods=['POST'])
def generate_animation():
    data = request.get_json(force=True)

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    negative_prompt = data.get("negative_prompt", "")
    num_frames = data.get("num_frames", 16)
    guidance_scale = data.get("guidance_scale", 7.5)
    num_inference_steps = data.get("num_inference_steps", 25)
    seed = data.get("seed", 42)

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

if __name__ == '__main__':
    # 4) When SERVER_NAME is set, you can just run on 127.0.0.1:5000
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
