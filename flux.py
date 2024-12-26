import torch
from diffusers import FluxPipeline
from huggingface_hub import login
login() #token is: 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
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# 7. Move to GPU
pipe.to("cuda")  # Move everything to GPU

pipe.enable_attention_slicing()  # Reduces memory usage for attention layers.
pipe.enable_vae_slicing()  # Reduces memory usage for VAE layers.
pipe.enable_sequential_cpu_offload()  # Offloads model parts to CPU to save GPU memory.
pipe.enable_vae_tiling()  # Further optimizes memory usage by processing VAE in tiles.

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")