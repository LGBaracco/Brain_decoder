import numpy as np
import torch
from scipy.stats import spearmanr

def load_embeddings(encoder, loader, device):
    """Extract latents and paired CLIP embeddings from a trained encoder."""
    encoder.eval()
    z_fmri_all, z_clip_all = [], []
    with torch.no_grad():
        for fmri_batch, clip_batch in loader:
            z_fmri_all.append(encoder(fmri_batch.to(device)).cpu())
            z_clip_all.append(clip_batch)
    return torch.cat(z_fmri_all).numpy(), torch.cat(z_clip_all).numpy()

def cosine_rdm(embeddings: np.ndarray) -> np.ndarray:
    """Representational dissimilarity matrix — 1 minus cosine similarity."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return 1 - (normed @ normed.T)


def upper_tri(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def rsa(z_fmri, z_clip, n_bootstrap=1000):
    """
    Spearman correlation between fMRI RDM and CLIP RDM.
    Higher = fMRI geometry matches CLIP geometry.
    """
    rdm_fmri = upper_tri(cosine_rdm(z_fmri))
    rdm_clip = upper_tri(cosine_rdm(z_clip))

    r, p = spearmanr(rdm_fmri, rdm_clip)

    # Bootstrap confidence interval
    n = len(rdm_fmri)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx    = np.random.choice(n, n, replace=True)
        boot_r, _ = spearmanr(rdm_fmri[idx], rdm_clip[idx])
        boot_rs.append(boot_r)

    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])
    print(f"RSA — r={r:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]), p={p:.2e}")
    return r, (ci_low, ci_high)
