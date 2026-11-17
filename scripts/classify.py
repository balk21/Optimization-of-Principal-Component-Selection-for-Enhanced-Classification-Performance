"""Classification accuracy against the number of principal components, and the k each rule picks.

PCA is fitted on the training images only, the rules see only the training images, and accuracy is
measured on the test images, for 1-nearest neighbour and logistic regression on the PCA scores.
AT&T: --repeats random splits with --train-per-class images of every person for training
(default: 10 splits of 5 training + 5 test images per person). MNIST: the standard 60 000 / 10 000 split.

Rules: the report's derivative rule on the in-sample error curve (grid step as in the report),
the knee of the same curve, and 95 % explained variance.

    python scripts/classify.py --dataset att
    python scripts/classify.py --dataset mnist
    python scripts/classify.py --dataset att --from-json   # only redo table and figure

Writes to results/<dataset>/classification/: accuracy.png, accuracy.csv, rules.md and summary.json.
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
from pcaselect import classify, data, plots, reconstruction, selection  # noqa: E402

REPORT_STEP = {"att": 5, "mnist": 10}
NAMES = {"att": "AT&T", "mnist": "MNIST"}
RULES = {"report": "derivative rule of the report", "knee": "knee of the error curve",
         "var95": "95 % explained variance", "all": "all components"}
MARKERS = {"report": "derivative rule (report)", "knee": "knee of the error curve",
           "var95": "95 % explained variance"}
MNIST_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140,
              160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 784]


def run_split(X_tr, y_tr, X_te, y_te, step: int, grid, classifiers) -> dict:
    pca = reconstruction.fit_pca(X_tr)
    n = pca.n_components_
    errors = reconstruction.error_curve(pca, X_tr)
    ks_report = np.arange(1, n + 1, step)
    rules = {
        "report": selection.report_rule(ks_report, errors[ks_report - 1]),
        "knee": selection.knee_rule(np.arange(1, n + 1), errors),
        "var95": selection.variance_rule(pca.explained_variance_ratio_, 0.95),
        "all": int(n),
    }
    z_tr = pca.transform(X_tr) / 255.0  # pixel scale 0-1 for the classifiers
    z_te = pca.transform(X_te) / 255.0
    ks = sorted({k for k in grid if k <= n} | set(rules.values()))
    acc = {c: dict(zip(ks, classify.accuracy_curve(z_tr, y_tr, z_te, y_te, ks, c).tolist())) for c in classifiers}
    return {"rules": rules, "accuracy": acc}


def compute(args) -> dict:
    X, y, _ = data.load(args.dataset, args.data_dir)
    step = REPORT_STEP[args.dataset]
    splits = []
    if args.dataset == "att":
        rng = np.random.default_rng(args.seed)
        n_train = 40 * args.train_per_class
        grid = sorted(set(range(1, 31)) | set(range(35, n_train, 5)) | {n_train})
        for r in range(args.repeats):
            tr, te = data.split_per_class(y, args.train_per_class, rng)
            splits.append(run_split(X[tr], y[tr], X[te], y[te], step, grid, args.classifiers))
            print(f"split {r + 1}/{args.repeats}: rules {splits[-1]['rules']}", flush=True)
        protocol = (f"{args.repeats} random splits, {args.train_per_class} training and "
                    f"{10 - args.train_per_class} test images per person")
    else:
        n_tr = data.MNIST_N_TRAIN
        grid = MNIST_GRID
        splits.append(run_split(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], step, grid, args.classifiers))
        print(f"rules {splits[0]['rules']}", flush=True)
        protocol = "standard split, 60 000 training and 10 000 test images"
    return {"dataset": args.dataset, "protocol": protocol, "seed": args.seed, "classifiers": args.classifiers,
            "grid": grid, "splits": splits}


def write_outputs(result: dict, out: Path) -> None:
    classifiers, splits, protocol = result["classifiers"], result["splits"], result["protocol"]
    for s in splits:  # JSON turns the integer keys into strings
        s["accuracy"] = {c: {int(k): v for k, v in a.items()} for c, a in s["accuracy"].items()}
    ks = sorted(set(result["grid"]).intersection(*[set(s["accuracy"][classifiers[0]]) for s in splits]))
    curves = {}
    for c in classifiers:
        a = np.array([[s["accuracy"][c][k] for k in ks] for s in splits])
        curves[c] = (a.mean(0), a.std(0))
    multi = len(splits) > 1

    def cell(values) -> str:
        values = 100 * np.asarray(values)
        return f"{values.mean():.1f} ± {values.std():.1f}" if multi else f"{values[0]:.2f}"

    def k_text(values) -> str:
        values = np.asarray(values)
        return f"{values[0]}" if values.min() == values.max() else f"{values.mean():.0f} ({values.min()}–{values.max()})"

    lines = [f"# {NAMES[result['dataset']]}: k picked by each rule and test accuracy (%)", "", protocol + ".", "",
             "| rule | k | " + " | ".join(classify.CLASSIFIERS[c] for c in classifiers) + " |",
             "|---|---|" + "---|" * len(classifiers)]
    per_rule = {}
    for rule, label in RULES.items():
        k_values = [s["rules"][rule] for s in splits]
        accs = {c: [s["accuracy"][c][s["rules"][rule]] for s in splits] for c in classifiers}
        per_rule[rule] = {"k": k_values, "accuracy": accs}
        lines.append(f"| {label} | {k_text(k_values)} | " + " | ".join(cell(accs[c]) for c in classifiers) + " |")
    best = {c: (ks[int(np.argmax(curves[c][0]))], float(curves[c][0].max())) for c in classifiers}
    lines.append("| best k on the test set (reference, not a rule) | | " + " | ".join(
        f"{100 * best[c][1]:.{1 if multi else 2}f} (k = {best[c][0]})" for c in classifiers) + " |")
    lines.append("")
    if multi:
        diff = {c: 100 * (np.array(per_rule["knee"]["accuracy"][c]) - np.array(per_rule["report"]["accuracy"][c]))
                for c in classifiers}
        lines += ["k: mean (range) over splits. Accuracy: mean ± standard deviation over splits.",
                  "Knee minus derivative rule, paired by split: " + ", ".join(
                      f"{classify.CLASSIFIERS[c]} {d.mean():+.1f} ± {d.std():.1f} points" for c, d in diff.items()) + "."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.md").write_text("\n".join(lines) + "\n")

    with open(out / "accuracy.csv", "w") as f:
        f.write("k," + ",".join(f"{c}_mean,{c}_std" for c in classifiers) + "\n")
        for i, k in enumerate(ks):
            f.write(f"{k}," + ",".join(f"{curves[c][0][i]:.5f},{curves[c][1][i]:.5f}" for c in classifiers) + "\n")

    rule_ks = {r: (label, int(round(np.mean(per_rule[r]["k"])))) for r, label in MARKERS.items()}
    fig = plots.accuracy_figure(
        ks, {classify.CLASSIFIERS[c]: (curves[c][0], curves[c][1] if multi else None) for c in classifiers},
        rule_ks, f"{NAMES[result['dataset']]}: test accuracy against k\n{protocol}")
    plots.save(fig, out / "accuracy.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["att", "mnist"], required=True)
    p.add_argument("--classifiers", nargs="+", default=list(classify.CLASSIFIERS), choices=list(classify.CLASSIFIERS))
    p.add_argument("--repeats", type=int, default=10, help="AT&T: number of random splits (default 10)")
    p.add_argument("--train-per-class", type=int, default=5, help="AT&T: training images per person (default 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", help="dataset folder (default: data/)")
    p.add_argument("--out", default=str(ROOT / "results"), help="output folder (default: results/)")
    p.add_argument("--from-json", action="store_true", help="skip the computation, reuse summary.json")
    args = p.parse_args()
    out = Path(args.out) / args.dataset / "classification"
    if args.from_json:
        result = json.loads((out / "summary.json").read_text())
    else:
        result = compute(args)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    write_outputs(result, out)
    print((out / "rules.md").read_text())


if __name__ == "__main__":
    main()
