import numpy as np
import torch
from torch.utils.data import DataLoader

from brain_decoder.models.encoders import BrainEncoder


@torch.no_grad()
def extract_embeddings(
    encoder: BrainEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    z_brain_all, z_target_all = [], []
    for fmri_batch, target_batch in loader:
        z_brain_all.append(encoder(fmri_batch.to(device)).cpu())
        z_target_all.append(target_batch)
    return torch.cat(z_brain_all).numpy(), torch.cat(z_target_all).numpy()
