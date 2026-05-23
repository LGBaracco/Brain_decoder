import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models import ridge_baseline, FMRIDataset, BrainEncoder, evaluate, infonce_loss
from analysis import load_embeddings, rsa

# TODO refactor into library, including different datasets, embeddings, and analysis tools


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

clip_space = torch.from_numpy(np.load("data/clip_embeddings/train_vitl14.npy")).float().to(DEVICE)
fmri_space = torch.from_numpy(np.load("data/train_data/subj01/training_split/training_fmri/lh_training_fmri.npy")).float().to(DEVICE)

def train(encoder, train_loader, val_loader, n_epochs=50, lr=3e-4):
    optimiser = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    best_top1 = 0.0

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        epoch_loss = 0.0

        for fmri_batch, clip_batch in train_loader:
            fmri_batch = fmri_batch.to(DEVICE)
            clip_batch = clip_batch.to(DEVICE)

            z_fmri = encoder(fmri_batch)
            loss   = infonce_loss(z_fmri, clip_batch)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0) # type: ignore
            optimiser.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0:
            top1, top5 = evaluate(encoder, val_loader)
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | top-1 {top1:.3f} | top-5 {top5:.3f}")
            if top1 > best_top1:
                best_top1 = top1
                torch.save(encoder.state_dict(), "data/encoders/best_encoder.pt")

    print(f"\nBest top-1: {best_top1:.3f}")

scaler    = StandardScaler()
fmri_norm = scaler.fit_transform(fmri_space.cpu()) # type: ignore

fmri_train, fmri_val, clip_train, clip_val = train_test_split(
    fmri_norm, clip_space.cpu(), test_size=500, shuffle=False
)

# Ridge baseline
ridge = ridge_baseline(fmri_train, clip_train, fmri_val, clip_val)

train_loader = DataLoader(
    FMRIDataset(fmri_train, clip_train),
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    FMRIDataset(fmri_val, clip_val),
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

n_voxels = fmri_train.shape[1]
encoder  = BrainEncoder(n_voxels=n_voxels, embed_dim=768).to(DEVICE)
print(f"\nEncoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
print(f"Voxels: {n_voxels} | Device: {DEVICE}")
print(f"Expected random-chance loss: {np.log(512):.3f}\n")

train(encoder, train_loader, val_loader)

fmri_embeddings, clip_embeddings = load_embeddings(encoder, val_loader, DEVICE)

rsa(fmri_embeddings, clip_embeddings)