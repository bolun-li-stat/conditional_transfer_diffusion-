"""Image diffusion processes."""

from .ddim import ImageDDIM
from .ddpm import ImageDDPM

__all__ = ["ImageDDIM", "ImageDDPM"]
