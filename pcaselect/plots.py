"""Figure helpers (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RULE_COLORS = {"report": "tab:red", "knee": "tab:green", "var95": "tab:purple"}


def save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_curve_figure(ks, errors, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, errors)
    ax.set(xlabel="number of components k", ylabel="mean reconstruction error (pixel units)", title=title)
    ax.grid(alpha=0.4)
    return fig


def slope_figure(ks, slopes, k_selected: int, title: str):
    ks = np.asarray(ks)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, slopes)
    ax.axhline(slopes[np.flatnonzero(ks == k_selected)[0]], color="k", ls="--", lw=1)
    ax.axvline(k_selected, color=RULE_COLORS["report"], ls="--", label=f"selected k = {k_selected}")
    ax.set(xlabel="number of components k", ylabel="slope dE/dk", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)
    return fig


def image_grid(rows, shape, col_titles=None, suptitle: str | None = None):
    """``rows[r][c]`` is a flat image with pixel values 0-255; all panels share that grey scale."""
    n_rows, n_cols = len(rows), len(rows[0])
    cell_h = 1.9
    cell_w = cell_h * shape[1] / shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows + 0.6), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r][c]
            ax.imshow(np.reshape(img, shape), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def accuracy_figure(ks, curves: dict, rule_ks: dict, title: str):
    """``curves[label] = (mean, std or None)`` accuracies in [0, 1]; ``rule_ks[rule] = (label, k)``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lowest = 100.0
    for label, (mean, std) in curves.items():
        mean = 100 * np.asarray(mean)
        ax.plot(ks, mean, label=label)
        if std is not None:
            std = 100 * np.asarray(std)
            ax.fill_between(ks, mean - std, mean + std, alpha=0.2)
        lowest = min(lowest, mean[np.asarray(ks) >= 20].min())
    for rule, (label, k) in rule_ks.items():
        ax.axvline(k, color=RULE_COLORS.get(rule, "grey"), ls="--", lw=1.2, label=f"{label}: k = {k:g}")
    ax.set_xscale("log")
    ax.set_xlim(5, max(ks) * 1.15)
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500) if t <= max(ks)] + [max(ks)]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(np.floor(lowest) - 3, 100)
    ax.set(xlabel="number of components k (log scale)", ylabel="test accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.4, which="both")
    ax.legend(fontsize=8, loc="lower right")
    return fig
