import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from brain_decoder.utils.retrieval import retrieval_accuracy


def ridge_baseline(
    fmri_train: np.ndarray,
    target_train: torch.Tensor,
    fmri_test: np.ndarray,
    target_test: torch.Tensor,
    *,
    alphas: tuple[float, ...] = (1e2, 1e3, 1e4),
) -> RidgeCV:
    print("\nFitting ridge regression baseline...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(fmri_train)
    x_test = scaler.transform(fmri_test)

    ridge = RidgeCV(alphas=alphas)
    ridge.fit(x_train, target_train.numpy())
    preds = ridge.predict(x_test)
    preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)

    top1, top5 = retrieval_accuracy(torch.from_numpy(preds), target_test)
    print(f"Ridge baseline — top-1: {top1:.3f} | top-5: {top5:.3f}")
    return ridge
