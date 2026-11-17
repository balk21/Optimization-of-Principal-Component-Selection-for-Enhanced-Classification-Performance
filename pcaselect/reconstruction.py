"""PCA fit, reconstructions and reconstruction-error curves for every number of components."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X: np.ndarray) -> PCA:
    """Exact PCA keeping all components.

    ``svd_solver="full"`` makes the result deterministic. sklearn's default ("auto") switches to a
    randomized solver for most k on these datasets, which adds run-to-run noise to the error curve.
    """
    return PCA(svd_solver="full").fit(np.asarray(X, dtype=np.float64))


def error_curve(pca: PCA, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Mean Euclidean distance between the images in ``X`` and their reconstructions.

    Element ``k - 1`` is the error with the first k components, for k = 1 ... ``pca.n_components_``.
    This is the error measure of the report: ``mean_i ||x_i - x_i'||`` in pixel units.
    The components are orthonormal, so ``||x - x_k'||^2 = ||x - mean||^2 - sum_{j<=k} z_j^2`` and a
    single fit gives the whole curve.
    """
    total = np.zeros(pca.n_components_)
    for start in range(0, len(X), batch_size):
        xb = np.asarray(X[start:start + batch_size], dtype=np.float64)
        z = pca.transform(xb)
        norm2 = ((xb - pca.mean_) ** 2).sum(axis=1)
        residual2 = norm2[:, None] - np.cumsum(z ** 2, axis=1)
        total += np.sqrt(np.clip(residual2, 0.0, None)).sum(axis=0)
    return total / len(X)


def reconstruct(pca: PCA, X: np.ndarray, k: int) -> np.ndarray:
    """Reconstruction of ``X`` from its first ``k`` principal components."""
    z = pca.transform(np.asarray(X, dtype=np.float64))[:, :k]
    return pca.mean_ + z @ pca.components_[:k]


def rank(pca: PCA) -> int:
    """Rank of the centred data the PCA was fitted on (numpy's ``matrix_rank`` tolerance).

    The in-sample reconstruction error is zero from this k on.
    """
    s = pca.singular_values_
    tol = s.max() * max(pca.n_samples_, pca.n_features_in_) * np.finfo(np.float64).eps
    return int((s > tol).sum())
