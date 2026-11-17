import numpy as np
from sklearn.decomposition import PCA

from pcaselect import reconstruction


def _data(n=60, d=30, r=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, r)) @ rng.normal(size=(r, d)) * 10 + 100  # rank r after centring


def test_error_curve_matches_explicit_reconstruction():
    X = _data()
    pca = reconstruction.fit_pca(X)
    curve = reconstruction.error_curve(pca, X, batch_size=7)
    for k in (1, 3, 8, 12, 20):
        direct = PCA(n_components=k, svd_solver="full").fit(X)
        rec = direct.inverse_transform(direct.transform(X))
        # sqrt(||x||^2 - sum z^2) loses about sqrt(eps) * ||x|| where the error is ~0 (here k >= 12)
        assert np.isclose(curve[k - 1], np.linalg.norm(X - rec, axis=1).mean(), rtol=1e-8, atol=1e-4)


def test_error_curve_on_held_out_data():
    X = _data(seed=1)
    pca = reconstruction.fit_pca(X[:40])
    curve = reconstruction.error_curve(pca, X[40:])
    for k in (2, 5):
        rec = reconstruction.reconstruct(pca, X[40:], k)
        assert np.isclose(curve[k - 1], np.linalg.norm(X[40:] - rec, axis=1).mean())
    assert np.all(np.diff(curve) <= 1e-9)  # adding components never increases the error


def test_in_sample_error_vanishes_at_the_rank():
    X = _data(r=12)
    pca = reconstruction.fit_pca(X)
    curve = reconstruction.error_curve(pca, X)
    assert reconstruction.rank(pca) == 12
    assert curve[10] > 1.0
    assert curve[11] < 1e-3


def test_reconstruct_with_all_components_is_exact():
    X = _data()
    pca = reconstruction.fit_pca(X)
    assert np.allclose(reconstruction.reconstruct(pca, X, pca.n_components_), X)
