import torch.nn as nn
from typing import Optional
from torch import Tensor

class LLaVA(nn.Module):
    def __init__(self, 
                 image_encoder: nn.Module,
                 text_encoder: nn.Module,
                 image_projection: nn.Module,
                 *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection

    def forward(self, 
                image:Optional[Tensor] = None,
                text: Optional[Tensor] = None
    ):
        pass