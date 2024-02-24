import torch.nn as nn
from typing import Optional
from torch import Tensor

class LLaVA(nn.Module):
    def __init__(self, 
                 image_encoder: nn.Module,
                 text_encoder: nn.Module,
                 image_projection: nn.Module,
                 multimodal_encoder: nn.Module,
                 *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.multimodal_encoder = multimodal_encoder

    def forward(self, 
                image: Optional[Tensor] = None,
                text: Optional[Tensor] = None
    ):
        image_embeddings = self._encode_and_project_image(image)
        text_embeddings = self.text_encoder(text)
        # Connect embeddings:
        joint_representation = self.multimodal_encoder(image_embeddings, text_embeddings)
        # Pass through language model:
        # Return response:
    
    def _encode_and_project_image(self,
                                  image: Tensor 
    ) -> Tensor:
        # Ideally:
        return self.image_projection(self.image_encoder(image))