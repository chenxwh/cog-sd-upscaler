import os

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)


MODEL_IDs = [
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
    "andite/anything-v4.0",
]
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipes = {
            MODEL_ID.split("/")[-1]: StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                cache_dir=MODEL_CACHE,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            for MODEL_ID in MODEL_IDs
        }
        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        model: str = Input(
            choices=["stable-diffusion-2-1", "stable-diffusion-v1-5", "anything-v4.0"],
            default="stable-diffusion-2-1",
            description="Choose a model.",
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image before upscaling. Final output will double the width. Lower the setting if run out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Final output will double the height. Lower the setting if run out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe = self.pipes[model]
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        low_res_latents = pipe(
            prompt=prompt if prompt is not None else None,
            negative_prompt=negative_prompt if negative_prompt is not None else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="latent",
        ).images

        upscaled_image = self.upscaler(
            prompt=prompt if prompt is not None else None,
            negative_prompt=negative_prompt if negative_prompt is not None else None,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
        ).images[0]

        output_path = f"/tmp/out.png"
        upscaled_image.save(output_path)

        return Path(output_path)


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
