import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler



class FMRIDataset(Dataset):
    def __init__(self, fmri: np.ndarray, clip_embs: torch.Tensor):
        # fmri:      (N, n_voxels) float32, already z-scored
        # clip_embs: (N, 768)      float32, already L2-normalised
        assert len(fmri) == len(clip_embs)
        self.fmri      = torch.from_numpy(fmri).float()
        self.clip_embs = clip_embs

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        return self.fmri[idx], self.clip_embs[idx]


class BrainEncoder(nn.Module):
    def __init__(self, n_voxels: int, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
    
def infonce_loss(z_fmri: torch.Tensor, z_clip: torch.Tensor, tau=0.07) -> torch.Tensor:

    S = (z_fmri @ z_clip.T) / tau               # (N, N)


    loss_fmri2clip = -torch.diag(F.log_softmax(S, dim=1)).mean()
    loss_clip2fmri = -torch.diag(F.log_softmax(S, dim=0)).mean()

    return (loss_fmri2clip + loss_clip2fmri) / 2

@torch.no_grad()
def retrieval_accuracy(z_fmri: torch.Tensor, z_clip: torch.Tensor):
    # For each fmri embedding, rank all clip embeddings by cosine similarity
    # Correct match is the same-index clip embedding
    S      = z_fmri @ z_clip.T                          # (N, N)
    ranks  = S.argsort(dim=1, descending=True)          # (N, N)
    labels = torch.arange(len(z_fmri))

    top1 = (ranks[:, 0] == labels).float().mean().item()
    top5 = (ranks[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, top5

@torch.no_grad()
def evaluate(encoder, loader):
    encoder.eval()
    all_fmri, all_clip = [], []

    for fmri_batch, clip_batch in loader:
        z = encoder(fmri_batch.cuda()).cpu()
        all_fmri.append(z)
        all_clip.append(clip_batch)

    z_fmri = torch.cat(all_fmri)
    z_clip = torch.cat(all_clip)
    return retrieval_accuracy(z_fmri, z_clip)

def ridge_baseline(fmri_train, clip_train, fmri_test, clip_test):
    print("\nFitting ridge regression baseline...")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(fmri_train)
    X_te   = scaler.transform(fmri_test)

    # RidgeCV picks regularisation via LOO cross-validation
    ridge  = RidgeCV(alphas=(1e2, 1e3, 1e4))
    ridge.fit(X_tr, clip_train)
    preds  = ridge.predict(X_te)
    preds  = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)

    top1, top5 = retrieval_accuracy(
        torch.from_numpy(preds),
        clip_test
    )
    print(f"Ridge baseline — top-1: {top1:.3f} | top-5: {top5:.3f}")

    return ridge