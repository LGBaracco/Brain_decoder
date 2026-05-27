import torch
import torch.nn.functional as F


def infonce_loss(z_brain: torch.Tensor, z_target: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    similarity = (z_brain @ z_target.T) / tau
    loss_brain_to_target = -torch.diag(F.log_softmax(similarity, dim=1)).mean()
    loss_target_to_brain = -torch.diag(F.log_softmax(similarity, dim=0)).mean()
    return (loss_brain_to_target + loss_target_to_brain) / 2
