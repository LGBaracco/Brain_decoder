from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brain_decoder.config import TrainConfig
from brain_decoder.losses.contrastive import infonce_loss
from brain_decoder.models.encoders import BrainEncoder
from brain_decoder.training.evaluator import evaluate_encoder


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: BrainEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: TrainConfig | None = None,
    ):
        self.encoder = encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or TrainConfig()

    def fit(self) -> float:
        cfg = self.config
        optimiser = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.n_epochs)
        best_top1 = 0.0

        for epoch in range(1, cfg.n_epochs + 1):
            self.encoder.train()
            epoch_loss = 0.0

            for fmri_batch, target_batch in self.train_loader:
                fmri_batch = fmri_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                z_brain = self.encoder(fmri_batch)
                loss = infonce_loss(z_brain, target_batch, tau=cfg.tau)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=cfg.grad_clip)  # type: ignore[arg-type]
                optimiser.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(self.train_loader)

            if epoch % cfg.eval_every == 0:
                top1, top5 = evaluate_encoder(self.encoder, self.val_loader, self.device)
                print(
                    f"Epoch {epoch:3d} | loss {avg_loss:.4f} | top-1 {top1:.3f} | top-5 {top5:.3f}"
                )
                if top1 > best_top1:
                    best_top1 = top1
                    self._save_checkpoint(cfg.checkpoint_path)

        print(f"\nBest top-1: {best_top1:.3f}")
        return best_top1

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
