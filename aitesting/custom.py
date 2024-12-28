import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    PNDMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
# Optional: if your pipeline includes a safety checker or feature extractor, import them as well:
# from diffusers import StableDiffusionSafetyChecker
# from transformers import CLIPImageProcessor
torch.backends.cuda.matmul.allow_tf32 = True
# Change this to the path where your local model is located
model_path = "/home/anon/Repos/AIImageGen/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/"

# 1. Load the tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained(
    model_path,
    subfolder="tokenizer",
)

text_encoder = CLIPTextModel.from_pretrained(
    model_path,
    subfolder="text_encoder",
    torch_dtype=torch.float32
)

# 2. Load the UNet model
unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
    torch_dtype=torch.float32
)

# 3. Load the VAE
vae = AutoencoderKL.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch.float32
)

# 4. Load the scheduler
scheduler = PNDMScheduler.from_pretrained(
    model_path,
    subfolder="scheduler"
)

# 5. (Optional) Load the safety checker and feature extractor if desired
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(
#     model_path,
#     subfolder="safety_checker",
#     torch_dtype=torch.float16
# )
feature_extractor = CLIPImageProcessor.from_pretrained(
    model_path,
    subfolder="feature_extractor"
)

# 6. Create the Stable Diffusion pipeline manually
pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=feature_extractor,
)

# 7. Move to GPU
pipe.to("cuda")  # Move everything to GPU

pipe.enable_attention_slicing()  # Reduces memory usage for attention layers.
pipe.enable_vae_slicing()  # Reduces memory usage for VAE layers.
pipe.enable_sequential_cpu_offload()  # Offloads model parts to CPU to save GPU memory.
pipe.enable_vae_tiling()  # Further optimizes memory usage by processing VAE in tiles.

# 8. Use the pipeline
prompt = "a photo of a dog"
negative_prompt = "blurry, distorted, grainy, low-res, ugly, text, watermark"

image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=8.0, num_inference_steps=30, height=768, width=512).images[0]
image.save("astronaut_rides_horse.png")

print("Image saved as astronaut_rides_horse.png")
