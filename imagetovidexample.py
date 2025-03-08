import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# Set device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
print(f"Using device: {device}")

# Load the LTXImageToVideoPipeline from the Lightricks/LTX-Video checkpoint
pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=dtype)
pipe.to(device)

# Apply VRAM optimizations (if supported by the pipeline)
if hasattr(pipe, "enable_model_cpu_offload"):
    pipe.enable_model_cpu_offload()  # Offload model components to CPU if needed
if hasattr(pipe, "enable_sequential_cpu_offload"):
    pipe.enable_sequential_cpu_offload()  # For sequential offloading on moderate VRAM GPUs
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()  # Reduce memory usage in attention computations
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()  # Optimize VAE computations by slicing
if hasattr(pipe, "enable_vae_tiling"):
    pipe.enable_vae_tiling()  # Further optimize VAE computations

# Load an input image from URL
image = load_image("https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png")

# Define prompts for generation
prompt = ("A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. "
          "Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, "
          "a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, "
          "creating a poignant and emotionally charged scene.")
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Generate the video (returns an object with a .frames attribute)
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_frames=150,
    num_inference_steps=15,
)

# Export the video as an MP4 file using 24 FPS
export_to_video(output.frames[0], "output.mp4", fps=24)
print("Video saved as output.mp4")
