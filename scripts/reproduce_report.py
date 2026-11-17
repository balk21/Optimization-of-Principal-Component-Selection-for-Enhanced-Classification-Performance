"""Reproduce the report's analysis (its Figures 1-4) with exact, seeded PCA.

As in the report's notebooks, PCA is fitted on a set of images, the reconstruction error is measured
on the same images for k = 1, 1 + step, 1 + 2 * step, ... components, and k is picked with the
derivative rule (``pcaselect.selection.report_rule``). Defaults follow the notebooks: AT&T, all 400
images and step 5; MNIST, the first 2000 images and step 10.

    python scripts/reproduce_report.py --dataset att
    python scripts/reproduce_report.py --dataset mnist

Writes to results/<dataset>/report/: error_curve.png (report Fig. 1), slope.png (Fig. 2),
reconstructions.png (Fig. 3), selected_k.png (Fig. 4) and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcaselect import data, plots, reconstruction, selection  # noqa: E402

DEFAULTS = {"att": {"n_fit": 400, "step": 5}, "mnist": {"n_fit": 2000, "step": 10}}
NAMES = {"att": "AT&T", "mnist": "MNIST"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["att", "mnist"], required=True)
    p.add_argument("--n-fit", type=int, help="fit on the first N images (default: AT&T 400, MNIST 2000)")
    p.add_argument("--step", type=int, help="grid step of k (default: AT&T 5, MNIST 10)")
    p.add_argument("--seed", type=int, default=0, help="seed for the images shown in the figures")
    p.add_argument("--data-dir", help="dataset folder (default: data/)")
    p.add_argument("--out", default=str(ROOT / "results"), help="output folder (default: results/)")
    args = p.parse_args()
    n_fit = args.n_fit or DEFAULTS[args.dataset]["n_fit"]
    step = args.step or DEFAULTS[args.dataset]["step"]
    name = NAMES[args.dataset]
    out = Path(args.out) / args.dataset / "report"
    rng = np.random.default_rng(args.seed)

    X, y, shape = data.load(args.dataset, args.data_dir)
    X_fit, y_fit = X[:n_fit], y[:n_fit]
    pca = reconstruction.fit_pca(X_fit)
    errors = reconstruction.error_curve(pca, X_fit)
    n_comp = pca.n_components_
    ks = np.arange(1, n_comp + 1, step)
    e = errors[ks - 1]
    slopes = selection.slope(ks, e)
    k_sel = selection.report_rule(ks, e)

    title = f"{name}, {n_fit} images"
    plots.save(plots.error_curve_figure(ks, e, f"Reconstruction error, {title}"), out / "error_curve.png")
    plots.save(plots.slope_figure(ks, slopes, k_sel, f"First derivative of the error, {title}"), out / "slope.png")

    # Fig. 3: three random images with 10, 50, 100 and all components
    shown = rng.choice(n_fit, 3, replace=False)
    ks_show = [10, 50, 100, n_comp]
    recon = {k: reconstruction.reconstruct(pca, X_fit[shown], k) for k in ks_show}
    rows = [[X_fit[i]] + [recon[k][r] for k in ks_show] for r, i in enumerate(shown)]
    fig = plots.image_grid(rows, shape, ["original"] + [f"k = {k}" for k in ks_show],
                           f"{name}: original and PCA reconstructions")
    plots.save(fig, out / "reconstructions.png")

    # Fig. 4: one image from each of five random classes, reconstructed with the selected k
    classes = rng.choice(np.unique(y_fit), 5, replace=False)
    shown = np.array([rng.choice(np.flatnonzero(y_fit == c)) for c in classes])
    rec = reconstruction.reconstruct(pca, X_fit[shown], k_sel)
    fig = plots.image_grid([list(X_fit[shown]), list(rec)], shape, [f"class {c}" for c in classes],
                           f"{name}: top original, bottom reconstructed with k = {k_sel}")
    plots.save(fig, out / "selected_k.png")

    top = np.argsort(slopes)[::-1][:5]
    summary = {
        "dataset": args.dataset,
        "images_fitted": int(n_fit),
        "pixels": int(X.shape[1]),
        "components": int(n_comp),
        "rank_of_fitted_data": reconstruction.rank(pca),
        "grid_step": int(step),
        "selected_k": int(k_sel),
        "selected_by": "slope ~ 0" if np.abs(slopes[ks == k_sel][0]) <= 1e-2 else "largest slope",
        "error_at_k1": float(e[0]),
        "error_at_selected_k": float(errors[k_sel - 1]),
        "reduction_vs_components_pct": round(100 * (1 - k_sel / n_comp), 2),
        "reduction_vs_pixels_pct": round(100 * (1 - k_sel / X.shape[1]), 2),
        "five_largest_slopes": [[int(ks[i]), round(float(slopes[i]), 4)] for i in top],
        "knee_k": selection.knee_rule(np.arange(1, n_comp + 1), errors),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
