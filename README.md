# AIImageGen
Create a /model/ folder with the corresponding models downloaded through civitai. Then in civittest.py you need to select the diffusion model the checkpoint is based on (SD, or SDXL).
Some civit models are pretty optimized, so you can generate large images. Some are not. Settings need to be independantly checked per model you load as well.

RECOMMENDED:

Create a python virtual env, to seperate out the pip modules (This is using python3 btw)

to create:
python3 -m venv sd-env (sd-env can be whatever)

to activate your env, do:
source sd-env/bin/activate

to deactivate:
deactivate


pip modules:

torch
diffusers
transformers
huggingface_hub


Run the AIDiscordBot using dotnet and fill out the config.json file it generates on first run to have a interface from discord to your image generation python server
