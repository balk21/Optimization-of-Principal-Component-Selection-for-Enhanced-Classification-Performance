"""Check what the report's derivative rule depends on.

The rule is applied to the in-sample error curve (PCA fitted and evaluated on the same images), as in
the report. Three things are varied:

1. the number of images the PCA is fitted on (AT&T: random subsets, --subsets per size;
   MNIST: the first n images, as in the report);
2. the grid step of k;
3. the PCA solver: the notebooks call ``PCA(n_components=k)`` with sklearn's default solver, which is
   randomized and unseeded for most k on AT&T. That procedure is repeated --solver-runs times.

    python scripts/check_rule.py
    python scripts/check_rule.py --from-json   # only redo the table and figure from rule_check.json

Writes results/rule_check.md, results/rule_check.json and results/rule_check.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pcaselect import data, plots, reconstruction, selection  # noqa: E402

ATT_SIZES = [100, 200, 300, 400]
MNIST_SIZES = [1000, 2000, 5000, 10000, 70000]
ATT_STEPS = [1, 5, 10]
MNIST_STEPS = [1, 10]
REPORT_K = {"att": (400, 271), "mnist": (2000, 631)}  # (images fitted, k) in the report


def selected_k(X_fit: np.ndarray, steps) -> dict:
    pca = reconstruction.fit_pca(X_fit)
    errors = reconstruction.error_curve(pca, X_fit)
    n = pca.n_components_
    out = {"rank": reconstruction.rank(pca)}
    for s in steps:
        ks = np.arange(1, n + 1, s)
        out[f"step{s}"] = selection.report_rule(ks, errors[ks - 1])
    return out


def notebook_procedure(X: np.ndarray, step: int) -> int:
    """The notebooks' loop: a new sklearn PCA with the default solver for every k."""
    X = X.astype(np.float32)  # the notebooks load the images as float32
    ks = np.arange(1, X.shape[0] + 1, step)
    errors = []
    for k in ks:
        pca = PCA(n_components=int(k))
        rec = pca.inverse_transform(pca.fit_transform(X))
        errors.append(np.linalg.norm(X - rec, axis=1).mean())
    return selection.report_rule(ks, np.array(errors))


def fmt(values) -> str:
    values = np.asarray(values)
    if values.min() == values.max():
        return f"{values[0]}"
    return f"{values.mean():.0f} ({values.min()}–{values.max()})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", type=int, default=5, help="random AT&T subsets per size (default 5)")
    p.add_argument("--solver-runs", type=int, default=5,
                   help="repeats of the notebooks' procedure on AT&T (default 5, up to a minute each; 0 skips)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", help="dataset folder (default: data/)")
    p.add_argument("--out", default=str(ROOT / "results"), help="output folder (default: results/)")
    p.add_argument("--from-json", action="store_true", help="skip the computation, reuse <out>/rule_check.json")
    args = p.parse_args()
    out = Path(args.out)
    if args.from_json:
        result = json.loads((out / "rule_check.json").read_text())
    else:
        result = compute(args)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rule_check.json").write_text(json.dumps(result, indent=1) + "\n")
    write_table(result, out)
    write_figure(result, out)
    print((out / "rule_check.md").read_text())


def compute(args) -> dict:
    rng = np.random.default_rng(args.seed)
    result = {"att": [], "mnist": [], "att_subsets": args.subsets}

    X, _, _ = data.load("att", args.data_dir)
    for n in ATT_SIZES:
        runs = [selected_k(X[np.sort(rng.choice(len(X), n, replace=False))] if n < len(X) else X, ATT_STEPS)
                for _ in range(args.subsets if n < len(X) else 1)]
        row = {"images": n, **{key: [r[key] for r in runs] for key in runs[0]}}
        result["att"].append(row)
        print("AT&T", row, flush=True)
    result["att_notebook_solver"] = [notebook_procedure(X, 5) for _ in range(args.solver_runs)]
    if args.solver_runs:
        print("AT&T, notebook procedure:", result["att_notebook_solver"], flush=True)

    X, _, _ = data.load("mnist", args.data_dir)
    for n in MNIST_SIZES:
        row = {"images": n, **{key: [v] for key, v in selected_k(X[:n], MNIST_STEPS).items()}}
        result["mnist"].append(row)
        print("MNIST", row, flush=True)
    return result


def write_table(result: dict, out: Path) -> None:
    solver_runs = result["att_notebook_solver"]
    lines = ["# What the report's derivative rule depends on", "",
             "In-sample error curves, as in the report. Each cell is the k the rule selects.", "",
             f"## AT&T (random subsets, {result['att_subsets']} per size; mean and range)", "",
             "| images fitted | rank of the data | step 1 | step 5 | step 10 | k / images (step 5) |",
             "|---|---|---|---|---|---|"]
    for r in result["att"]:
        lines.append(f"| {r['images']} | {fmt(r['rank'])} | {fmt(r['step1'])} | {fmt(r['step5'])} | "
                     f"{fmt(r['step10'])} | {np.mean(r['step5']) / r['images']:.2f} |")
    if solver_runs:
        lines += ["", f"All 400 images, step 5, the notebooks' procedure (sklearn's default solver) run "
                      f"{len(solver_runs)} times: k = {', '.join(map(str, solver_runs))}. "
                      f"The report's 271 is one such run."]
    lines += ["", "## MNIST (the first n images, as in the report)", "",
              "| images fitted | rank of the data | step 1 | step 10 |", "|---|---|---|---|"]
    for r in result["mnist"]:
        lines.append(f"| {r['images']} | {r['rank'][0]} | {r['step1'][0]} | {r['step10'][0]} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rule_check.md").write_text("\n".join(lines) + "\n")


def write_figure(result: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, step, title in [(axes[0], "att", 5, "AT&T (step 5)"), (axes[1], "mnist", 10, "MNIST (step 10)")]:
        rows = result[key]
        n = [r["images"] for r in rows]
        ax.plot(n, [np.mean(r[f"step{step}"]) for r in rows], "o-", color=plots.RULE_COLORS["report"],
                label="k selected by the rule (exact PCA)")
        ax.plot(n, [np.mean(r["rank"]) for r in rows], "s--", color="k", label="rank of the fitted data")
        if key == "att" and result["att_notebook_solver"]:
            runs = result["att_notebook_solver"]
            ax.plot([400] * len(runs), runs, "x", color="tab:blue", alpha=0.7,
                    label="notebook code, reruns (randomized solver)")
        ax.plot(*REPORT_K[key], "*", color="tab:blue", markersize=13, label=f"report: k = {REPORT_K[key][1]}")
        ax.set(xlabel="images the PCA is fitted on", ylabel="k", title=title)
        if key == "mnist":
            ax.set_xscale("log")
        ax.grid(alpha=0.4)
        ax.legend(loc="upper left" if key == "att" else "lower right", fontsize=8)
    fig.suptitle("The derivative rule follows the size of the fitted data", fontsize=11)
    fig.tight_layout()
    plots.save(fig, out / "rule_check.png")


if __name__ == "__main__":
    main()
