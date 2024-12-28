from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
import os
import uuid
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0

# Initialize colorama for colored output (optional)
from colorama import Fore, Style, init
init()

# Define the directory for saving images
image_save_directory = 'generated_images'
os.makedirs(image_save_directory, exist_ok=True)

# Load the model outside of the request to save loading time
local_model_path = './models/sd/2dnSD15_2.safetensors'
# local_model_path = './models/sdxl1/noobaiXLNAIXL_vPred10Version.safetensors'
if not os.path.exists(local_model_path):
    print(f"{Fore.RED}Error: The path {local_model_path} does not exist.{Style.RESET_ALL}")
    # In Colab, raise an exception instead of sys.exit()
    raise FileNotFoundError(f"The path {local_model_path} does not exist.")

device = "cuda" if torch.cuda.is_available() else "cpu"
#pipe = StableDiffusionXLPipeline.from_single_file(local_model_path, torch_dtype=torch.float16, variant="fp16")
pipe = StableDiffusionPipeline.from_single_file(local_model_path, torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduces memory usage for attention layers.
pipe.enable_vae_slicing()  # Reduces memory usage for VAE layers.
pipe.enable_sequential_cpu_offload()  # Offloads model parts to CPU to save GPU memory.
pipe.enable_vae_tiling()  # Further optimizes memory usage by processing VAE in tiles.

# Define your prompt
prompt = "highly detailed, realistic, fine fabric detail, absurdres, highly-detailed, best quality, masterpiece, very aesthetic, portrait, a horrifying creature lurking in the night."
neg_prompt = "lowres, worst quality, low quality, bad anatomy, bad hands, multiple views, abstract, signature, furry, anthro, bkub, 2koma, 4koma, comic, manga, sketch, ixy,"
# Generate image
generator = torch.cuda.manual_seed(0)

torch.cuda.empty_cache()
image = pipe(prompt, num_inference_steps=50, generator=generator, width=768, height=1088).images[0]

# Save the image file
unique_filename = f"{uuid.uuid4().hex}.png"
file_path = os.path.join(image_save_directory, unique_filename)
image.save(file_path)

print(f"{Fore.GREEN}Image saved at: {file_path}{Style.RESET_ALL}")

