import torch


@torch.no_grad()
def retrieval_accuracy(z_brain: torch.Tensor, z_target: torch.Tensor) -> tuple[float, float]:
    similarity = z_brain @ z_target.T
    ranks = similarity.argsort(dim=1, descending=True)
    labels = torch.arange(len(z_brain))

    top1 = (ranks[:, 0] == labels).float().mean().item()
    top5 = (ranks[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, top5
