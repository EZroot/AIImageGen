import os
import uuid
import torch
import logging
from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image

# Import diffsynth and modelscope modules
from diffsynth import ModelManager, WanVideoPipeline, save_video
from modelscope import snapshot_download, dataset_snapshot_download

# ------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# ------------------------------
# Create Flask app and directories
app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['VIDEO_SAVE_DIRECTORY'] = os.path.join(app_root, 'generated_videos')
os.makedirs(app.config['VIDEO_SAVE_DIRECTORY'], exist_ok=True)

# ------------------------------
# Device selection and torch_dtype setup:
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16  # Use half precision on GPU to reduce VRAM usage
    logging.info("CUDA is available. Using GPU with torch.float16 for reduced VRAM usage.")
else:
    device = "cpu"
    torch_dtype = torch.float32
    logging.info("CUDA not available. Using CPU with torch.float32.")

# ------------------------------
# Download and load the WAN 480P model for diffsynth
model_dir = "models/Wan-AI/Wan2.1-I2V-14B-480P"
if not os.path.exists(model_dir):
    logging.info("Downloading WAN 480P model...")
    snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir=model_dir)
else:
    logging.info("WAN 480P model already exists.")

logging.info("Initializing ModelManager and loading models...")
# Create a ModelManager with the selected device
model_manager = ModelManager(device=device)
model_manager.load_models(
    [
        [
            os.path.join(model_dir, f"diffusion_pytorch_model-0000{i}-of-00007.safetensors")
            for i in range(1, 8)
        ],
        os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
    ],
    torch_dtype=torch_dtype
)

logging.info("Creating WanVideoPipeline...")
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch_dtype, device=device)

# Enable VRAM management on GPU (adjust value as needed for your 6GB VRAM)
if device == "cuda":
    pipe.enable_vram_management(num_persistent_param_in_dit=2 * 10**9)
    logging.info("Enabled VRAM management for GPU.")

# ------------------------------
# Download default input image if not already present
default_image_path = "./data/data/examples/wan/input_image.jpg"
default_image = Image.open(default_image_path)

# ------------------------------
# Route to serve generated video files
@app.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    return send_from_directory(app.config['VIDEO_SAVE_DIRECTORY'], filename)

# ------------------------------
# /generate endpoint: generate a video using diffsynth’s WanVideoPipeline
@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 50)
    video_fps = data.get("video_fps", 15)
    seed = data.get("seed", 0)

    logging.info(
        f"Generating video with settings:\n"
        f"  Prompt: {prompt}\n"
        f"  Negative Prompt: {negative_prompt}\n"
        f"  Num Inference Steps: {num_inference_steps}\n"
        f"  Video FPS: {video_fps}\n"
        f"  Seed: {seed}\n"
    )

    # Using the default image as base input.
    input_image = default_image

    try:
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            num_inference_steps=num_inference_steps,
            seed=seed,
            tiled=True
        )
    except Exception as e:
        logging.error(f"Error during video generation: {e}")
        return jsonify({"error": str(e)}), 500

    # Save the generated video to a unique file
    video_filename = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(app.config['VIDEO_SAVE_DIRECTORY'], video_filename)
    try:
        save_video(video, video_path, fps=video_fps, quality=5)
    except Exception as e:
        logging.error(f"Error saving video: {e}")
        return jsonify({"error": str(e)}), 500

    with app.app_context():
        video_url = url_for('get_video', filename=video_filename, _external=True)

    logging.info(f"Video generated and saved at: {video_path}")
    return jsonify({
        "message": "Success",
        "video_url": video_url,
        "video_path": os.path.abspath(video_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "settings": {
            "num_inference_steps": num_inference_steps,
            "video_fps": video_fps,
            "seed": seed
        }
    }), 200

# ------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
