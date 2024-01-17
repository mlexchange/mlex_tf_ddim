from typing import List, Tuple, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, ValidationError
from pydantic.functional_validators import AfterValidator


def check8(v: int):
     assert v % 8 == 0, 'Element of image_size should be divided by 8!'
     return v

MyImageSize = Annotated[int, AfterValidator(check8)]


class ModelConfig(BaseModel):
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
    image_size: Tuple[MyImageSize, MyImageSize] = (128, 128)
    batch_size: int = 32
    ema: Optional[float] = 0.999
