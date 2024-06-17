import warnings

from diffusers_sd3_control import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline  # noqa F401


warnings.warn(
    "The `inpainting.py` script is outdated. Please use directly `from diffusers_sd3_control import"
    " StableDiffusionInpaintPipeline` instead."
)
