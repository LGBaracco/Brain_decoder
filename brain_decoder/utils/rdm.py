import numpy as np


def cosine_rdm(embeddings: np.ndarray) -> np.ndarray:
    """Representational dissimilarity matrix — 1 minus cosine similarity."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return 1 - (normed @ normed.T)


def upper_tri(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]
