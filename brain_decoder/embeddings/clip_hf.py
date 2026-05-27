from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from brain_decoder.utils.device import default_device


class CLIPImageExtractor:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: torch.device | None = None,
    ):
        self.model_id = model_id
        self.device = device or default_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self._embed_dim = self.model.config.projection_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def extract_batch(self, image_paths: list[Path]) -> np.ndarray:
        out = np.zeros((len(image_paths), self.embed_dim), dtype=np.float32)
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs).cpu().numpy()
            out[i] = emb
        return out
