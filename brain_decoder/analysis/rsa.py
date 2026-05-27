import numpy as np
from scipy.stats import spearmanr

from brain_decoder.utils.rdm import cosine_rdm, upper_tri


def run_rsa(
    z_a: np.ndarray,
    z_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
) -> tuple[float, tuple[float, float]]:
    """Spearman correlation between two representational dissimilarity matrices."""
    rdm_a = upper_tri(cosine_rdm(z_a))
    rdm_b = upper_tri(cosine_rdm(z_b))

    r, p = spearmanr(rdm_a, rdm_b)

    n = len(rdm_a)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_r, _ = spearmanr(rdm_a[idx], rdm_b[idx])
        boot_rs.append(boot_r)

    ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])
    print(f"RSA — r={r:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]), p={p:.2e}")
    return r, (ci_low, ci_high)
