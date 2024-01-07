from typing import List, Optional
from pydantic import BaseModel


class ModelParams(BaseModel):
    # KID = Kernel Inception Distance, used for validation step
    kid_image_size: Optional[int] = 75
    kid_diffusion_steps: Optional[int] = 5  
    plot_diffusion_steps: Optional[int] = 10 #diffusion steps, also the denoise steps for prediction

    # scheduler parameters
    min_signal_rate: Optional[float] = 0.02
    max_signal_rate: Optional[float] = 0.95

    # block parameters
    widths: Optional[List[int]] = [32, 64, 96, 128]
    block_depth: Optional[int] = 2

    # optimization
    image_size: int = 128
    batch_size: int = 32
    ema: Optional[float] = 0.999