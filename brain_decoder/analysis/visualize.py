from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class AnalysisCache:
    z_brain: np.ndarray
    z_target: np.ndarray
    rsa_r: float
    ci_low: float
    ci_high: float


def save_analysis(
    path: Path | str,
    z_brain: np.ndarray,
    z_target: np.ndarray,
    rsa_r: float,
    ci_low: float,
    ci_high: float,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        z_brain=z_brain,
        z_target=z_target,
        rsa_r=np.float64(rsa_r),
        ci_low=np.float64(ci_low),
        ci_high=np.float64(ci_high),
    )
    print(f"Analysis cache saved to {output}")
    return output


def load_analysis(path: Path | str) -> AnalysisCache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Analysis cache not found: {cache_path}\n"
            "Run the analyze stage first (`--analyze`)."
        )
    data = np.load(cache_path)
    return AnalysisCache(
        z_brain=data["z_brain"],
        z_target=data["z_target"],
        rsa_r=float(data["rsa_r"]),
        ci_low=float(data["ci_low"]),
        ci_high=float(data["ci_high"]),
    )


def pca_reduce(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    n_samples, n_features = embeddings.shape
    k = min(n_components, n_samples, n_features)
    return PCA(n_components=k).fit_transform(embeddings)


def cross_similarity(z_brain: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    brain = z_brain / (np.linalg.norm(z_brain, axis=1, keepdims=True) + 1e-8)
    target = z_target / (np.linalg.norm(z_target, axis=1, keepdims=True) + 1e-8)
    return brain @ target.T


def sort_by_similarity(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - matrix
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return matrix[np.ix_(order, order)], order


def save_rsa_heatmaps(
    cache_path: Path | str,
    output_path: Path | str,
    *,
    n_components: int = 50,
) -> Path:
    import matplotlib.pyplot as plt

    cache = load_analysis(cache_path)
    z_brain = pca_reduce(cache.z_brain, n_components=n_components)
    z_target = pca_reduce(cache.z_target, n_components=n_components)

    cross_sim = cross_similarity(z_brain, z_target)
    sorted_sim, _ = sort_by_similarity(cross_sim)

    n_stimuli = sorted_sim.shape[0]
    n_pca = z_brain.shape[1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sorted_sim, cmap="viridis", vmin=cross_sim.min(), vmax=cross_sim.max(), aspect="auto")
    ax.set_xlabel(f"Target stimulus (N={n_stimuli}, sorted)")
    ax.set_ylabel(f"Brain stimulus (N={n_stimuli}, sorted)")
    ax.set_title(f"Brain–target cosine similarity (PCA k={n_pca} features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"RSA r={cache.rsa_r:.4f} (95% CI [{cache.ci_low:.4f}, {cache.ci_high:.4f}])")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"RSA heatmap saved to {output}")
    return output
