"""Rules that choose the number of components k.

All rules see only the data the PCA was fitted on.
"""
from __future__ import annotations

import numpy as np


def slope(ks, errors) -> np.ndarray:
    """First derivative dE/dk of the error curve (``np.gradient``, as in the report)."""
    return np.gradient(np.asarray(errors, dtype=np.float64), np.asarray(ks, dtype=np.float64))


def report_rule(ks, errors, atol: float = 1e-2) -> int:
    """Derivative rule of the report's notebooks.

    The first k whose slope is within ``atol`` of zero; if there is none, the k with the largest
    (least negative) slope. ``atol`` is in error units per component, so it depends on the pixel scale.
    On an in-sample error curve the slope reaches zero only where the error does, i.e. at the rank
    of the fitted data.
    """
    ks = np.asarray(ks)
    d = slope(ks, errors)
    flat = np.flatnonzero(np.abs(d) <= atol)
    i = flat[0] if flat.size else int(np.argmax(d))
    return int(ks[i])


def knee_rule(ks, errors) -> int:
    """Knee of a decreasing curve: the point farthest below the line joining its first and last points.

    This is the Kneedle point without smoothing. There the slope of the curve equals the slope of that
    line, so it is the derivative rule ``dE/dk = -lambda`` with the cost of one component, ``lambda``,
    set to the average gain per component over the range.
    """
    ks = np.asarray(ks, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (e - e[-1]) / (e[0] - e[-1])
    return int(ks[np.argmax((1.0 - x) - y)])


def variance_rule(explained_variance_ratio, threshold: float = 0.95) -> int:
    """Smallest k whose components explain at least ``threshold`` of the variance."""
    cum = np.cumsum(explained_variance_ratio)
    return int(min(np.searchsorted(cum, threshold) + 1, len(cum)))
