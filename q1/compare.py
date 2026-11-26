"""Compare aligned out-of-date predictions, weighting each held-out recording once."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss

from .data import CLASSES, load_data


def score(matrix):
    tp = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(3), where=(actual + predicted) > 0)
    return {"macro_f1": f1.mean(), "balanced_accuracy": (tp / actual).mean(),
            "accuracy": tp.sum() / matrix.sum()}


def compare(data, files, out):
    pred = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    reference = data.set_index("row_id").sort_index()
    if pred["model"].isna().any() or pred.duplicated(["model", "row_id"]).any():
        raise ValueError("Missing model name or overlapping (model, row_id) predictions.")
    out.mkdir(parents=True, exist_ok=True)
    scores, counts, by_date, arrays = [], [], [], {}
    dates = sorted(data.Date.unique())
    for model, group in pred.groupby("model"):
        rows = group.set_index("row_id").sort_index()
        if (not rows.index.equals(reference.index)
                or not rows.Date.equals(reference.Date)
                or not rows.actual.equals(reference.CallType)):
            raise ValueError(f"OOF identities, dates or true labels differ for {model}.")
        if not rows.prediction.isin(CLASSES).all():
            raise ValueError(f"Unrecognized prediction label for {model}.")
        prob = rows[[f"p_{c}" for c in CLASSES]].to_numpy()
        if not (np.isfinite(prob).all() and (prob >= 0).all()
                and (prob <= 1).all() and np.allclose(prob.sum(1), 1, atol=1e-6, rtol=0)):
            raise ValueError(f"Invalid probability for {model}.")
        # CSV writers may round the last decimal of a probability vector.
        prob = prob / prob.sum(axis=1, keepdims=True)
        cm = confusion_matrix(rows.actual, rows.prediction, labels=CLASSES)
        scores.append({"model": model, "n": len(rows), **score(cm),
                       "log_loss": log_loss(rows.actual, prob, labels=CLASSES)})
        daily = []
        for date in dates:
            mask = rows.Date.eq(date)
            day = confusion_matrix(rows.loc[mask, "actual"], rows.loc[mask, "prediction"], labels=CLASSES)
            daily.append(day)
            by_date.append({"model": model, "Date": date, "n": int(mask.sum()),
                            "accuracy": np.trace(day) / day.sum(),
                            "log_loss": log_loss(rows.loc[mask, "actual"], prob[mask], labels=CLASSES)})
        arrays[model] = np.array(daily)
        for i, c in enumerate(CLASSES):
            counts.append({"model": model, "actual": c, **dict(zip(CLASSES, cm[i]))})
    ranked = pd.DataFrame(scores).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    ranked.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "confusion.csv", index=False)
    pd.DataFrame(by_date).to_csv(out / "by_date.csv", index=False)
    # Conditional sensitivity to which dates were sampled, keeping fitted predictions fixed.
    # Dates missing an entire true class make the defined three-class BA inestimable.
    draws = np.random.default_rng(20261126).multinomial(len(dates), [1 / len(dates)] * len(dates), size=5000)
    boot = {}
    for model, daily in arrays.items():
        cubes = np.einsum("rd,dij->rij", draws, daily)
        valid = (cubes.sum(axis=2) > 0).all(axis=1)
        cubes = cubes[valid]
        tp = cubes.diagonal(axis1=1, axis2=2)
        boot[model] = (2 * tp / (cubes.sum(axis=2) + cubes.sum(axis=1))).mean(axis=1)
    comparisons = []
    for idx, name in enumerate(ranked.model):
        for other in ranked.model.iloc[idx + 1:]:
            diff = boot[name] - boot[other]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            point = ranked.set_index("model").macro_f1
            comparisons.append({"first": name, "second": other, "delta_macro_f1": point[name] - point[other],
                                "q025": lo, "q975": hi, "valid_date_draws": len(diff)})
    pd.DataFrame(comparisons).to_csv(out / "paired_date_sensitivity.csv", index=False)
    plot_confusion(arrays, ranked.model, out / "confusion.png")
    return ranked


def plot_confusion(arrays, names, path):
    columns = min(len(names), 3)
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.9 * rows), squeeze=False,
                             layout="constrained")
    for name, ax in zip(names, axes.flat):
        cm = arrays[name].sum(axis=0)
        fractions = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), color="white" if fractions[i, j] > .55 else "black",
                        ha="center", va="center")
        ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
               title=name, xlabel="Predicted", ylabel="Recorded")
    for ax in list(axes.flat)[len(names):]:
        ax.set_axis_off()
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/q1/comparison"))
    parser.add_argument("predictions", nargs="+", type=Path)
    args = parser.parse_args()
    print(compare(load_data(args.data), args.predictions, args.out).to_string(index=False))
