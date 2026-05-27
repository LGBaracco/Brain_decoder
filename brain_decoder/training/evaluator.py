import torch
from torch.utils.data import DataLoader

from brain_decoder.models.encoders import BrainEncoder
from brain_decoder.utils.device import default_device
from brain_decoder.utils.retrieval import retrieval_accuracy


@torch.no_grad()
def evaluate_encoder(
    encoder: BrainEncoder,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[float, float]:
    device = device or default_device()
    encoder.eval()
    all_brain, all_target = [], []

    for fmri_batch, target_batch in loader:
        z = encoder(fmri_batch.to(device)).cpu()
        all_brain.append(z)
        all_target.append(target_batch)

    z_brain = torch.cat(all_brain)
    z_target = torch.cat(all_target)
    return retrieval_accuracy(z_brain, z_target)
