import torch
from diffusers import AnimateDiffSDXLPipeline, EulerDiscreteScheduler
from PIL import Image
import imageio
import os

def main():
    ####################################################################
    # 1. Load the AnimateDiff-SDXL pipeline with motion adapter, etc.
    #    (Replace "someuser/animatediff-sdxl-checkpoint" with the actual
    #     HF repo or local folder that includes motion_adapter etc.)
    ####################################################################
    pipe = AnimateDiffSDXLPipeline.from_pretrained(
        "someuser/animatediff-sdxl-checkpoint",
        torch_dtype=torch.float16
    )
    
    ####################################################################
    # 2. (Optional) Load custom UNet weights from .safetensors
    ####################################################################
    local_unet_path = "./models/illustrious/waiNSFWIllustrious_v110.safetensors"
    unet_state_dict = torch.load(local_unet_path, map_location="cpu")
    pipe.unet.load_state_dict(unet_state_dict, strict=False)
    
    ####################################################################
    # 3. (Optional) Attach a custom scheduler (Euler, etc.)
    ####################################################################
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "someuser/animatediff-sdxl-checkpoint",
        subfolder="scheduler"
    )
    pipe.scheduler = scheduler
    
    # Move pipeline to GPU
    pipe.to("cuda")
    
    ####################################################################
    # 4. Define your prompt and other generation settings
    ####################################################################
    prompt = "a majestic dragon flying through stormy skies, cinematic, highly detailed"
    negative_prompt = "blurry, low quality, bad anatomy"
    
    num_frames = 16
    num_inference_steps = 30
    guidance_scale = 7.5
    
    ####################################################################
    # 5. Generate the animation frames
    #    (Exact arguments depend on how your AnimateDiff fork is set up)
    ####################################################################
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
    )
    
    # Some forks return a list of PIL images directly; others .frames
    if hasattr(result, "frames"):
        frames = result.frames
    else:
        frames = result
    
    ####################################################################
    # 6. Save the frames to an animated GIF
    ####################################################################
    output_gif = "dragon_animation.gif"
    imageio.mimsave(output_gif, frames, fps=8)
    print(f"Animation saved to: {os.path.abspath(output_gif)}")

if __name__ == "__main__":
    main()
