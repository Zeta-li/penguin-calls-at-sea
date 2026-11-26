"""Compare four small tabular classifiers with nested, date-blocked validation."""

from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import CLASSES, FEATURES


SEED = 20261126
PROB_COLS = [f"p_{name}" for name in CLASSES]
# Six settings per family. A numerical tie keeps the first listed setting.
GRIDS = {
    "ridge": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
    "lda": [{"shrinkage": s} for s in ("auto", 0.75, 0.5, 0.25, 0.1, 0.0)],
    "tree": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth, leaf in product((1, 2, 3), (6, 3))
    ],
    "svm": [
        {"C": c, "gamma": gamma}
        for c, gamma in product((0.3, 1.0, 3.0), (0.05, 0.3))
    ],
}
DROP_GROUPS = {
    "Duration": ("Duration",),
    "F01": ("F01",),
    "Fpeak": ("Fpeak",),
    "spectral_quartiles": ("Q25", "Q50", "Q75"),
}


def _group_splits(y, groups, n_splits):
    splits = list(GroupKFold(n_splits=n_splits).split(y, y, groups))
    for train, test in splits:
        if set(y[train]) != set(CLASSES) or np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Date split overlaps or omits a training class")
    return splits


def _predict(model, params, x, y, dates, query):
    if model == "ridge":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, tol=1e-8, **params))
    elif model == "lda":
        estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", **params))
    elif model == "tree":
        estimator = DecisionTreeClassifier(random_state=SEED, **params)
    elif model == "svm":
        # Calibrate RBF decision scores without exposing the calibration date
        # to the corresponding SVM fit (including its feature standardizer).
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", **params))
        estimator = CalibratedClassifierCV(
            svm, cv=_group_splits(y, dates, 3), method="sigmoid", ensemble=False
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    estimator.fit(x, y)
    if list(estimator.classes_) != list(CLASSES):
        raise ValueError("Unexpected output class order")
    probability = estimator.predict_proba(query)
    if model == "tree":
        # Add one observation of each class per leaf for finite log loss.
        # Leaves are unweighted, so class counts equal frequency times n_leaf.
        leaf_n = estimator.tree_.n_node_samples[estimator.apply(query), None]
        probability = (probability * leaf_n + 1) / (leaf_n + len(CLASSES))
    return probability


def _oof(data, model, features):
    x = data.loc[:, list(features)].to_numpy(dtype=float)
    y = data["CallType"].to_numpy()
    dates = data["Date"].to_numpy()
    oof = np.full((len(y), len(CLASSES)), np.nan)
    schedule = []
    for train, test in LeaveOneGroupOut().split(x, y, dates):
        xt, yt, dt = x[train], y[train], dates[train]
        folds = _group_splits(yt, dt, 4)
        fold_scores = []
        for params in GRIDS[model]:
            pred = np.empty(len(yt), dtype=y.dtype)
            for fit_idx, valid_idx in folds:
                p = _predict(model, params, xt[fit_idx], yt[fit_idx], dt[fit_idx], xt[valid_idx])
                pred[valid_idx] = np.take(CLASSES, np.argmax(p, axis=1))
            fold_scores.append(f1_score(yt, pred, labels=CLASSES, average="macro", zero_division=0))
        chosen = int(np.argmax(fold_scores))
        oof[test] = _predict(model, GRIDS[model][chosen], xt, yt, dt, x[test])
        for idx, (params, score) in enumerate(zip(GRIDS[model], fold_scores)):
            schedule.append({
                "Date": dates[test[0]], "n_test": len(test), "model": model,
                "parameters": json.dumps(params, sort_keys=True), "inner_macro_f1": score,
                "selected": idx == chosen,
            })
    if not np.isfinite(oof).all() or (oof < 0).any() or not np.allclose(oof.sum(axis=1), 1):
        raise ValueError("Invalid or missing out-of-fold probabilities")
    frame = data[["row_id", "Date", "CallType"]].rename(columns={"CallType": "actual"}).copy()
    frame["model"] = model
    frame[PROB_COLS] = oof
    frame["prediction"] = np.take(CLASSES, oof.argmax(axis=1))
    return frame, pd.DataFrame(schedule)


def _metrics(frame):
    y, pred = frame["actual"], frame["prediction"]
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "log_loss": log_loss(y, frame[PROB_COLS], labels=CLASSES),
    }


def _date_bootstrap(oof, draws=5000):
    """Resample whole dates using common draws across all fixed OOF predictions."""
    dates = sorted(oof["Date"].unique())
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(dates), np.full(len(dates), 1/len(dates)), size=draws)
    estimates = {}
    point = {}
    for model, frame in oof.groupby("model", sort=False):
        matrices = []
        losses = []
        for date in dates:
            g = frame.loc[frame["Date"] == date]
            matrices.append(confusion_matrix(g.actual, g.prediction, labels=CLASSES))
            losses.append(log_loss(g.actual, g[PROB_COLS], labels=CLASSES, normalize=False))
        cm = np.einsum("ij,jkl->ikl", weights, np.stack(matrices))
        tp = np.diagonal(cm, axis1=1, axis2=2)
        support, predicted = cm.sum(axis=2), cm.sum(axis=1)
        # Compare only samples containing all three true classes.
        valid = (support > 0).all(axis=1)
        n = support[valid].sum(axis=1)
        estimates[model] = {
            "macro_f1": (2*tp[valid]/(support[valid] + predicted[valid])).mean(axis=1),
            "balanced_accuracy": (tp[valid]/support[valid]).mean(axis=1),
            "accuracy": tp[valid].sum(axis=1)/n,
            "log_loss": (weights[valid] @ np.array(losses))/n,
        }
        point[model] = _metrics(frame)
    rows = []
    for a, b in combinations(estimates, 2):
        for metric, estimate in estimates[a].items():
            delta = estimate - estimates[b][metric]
            low, high = np.quantile(delta, [0.025, 0.975])
            rows.append({
                "model_a": a, "model_b": b, "metric": metric,
                "a_minus_b": point[a][metric] - point[b][metric],
                "lower_95": low, "upper_95": high,
                "valid_date_resamples": len(delta), "total_resamples": draws,
            })
    return pd.DataFrame(rows)


def _save(frame, out, name):
    precision = "%.17g" if set(PROB_COLS).issubset(frame.columns) else "%.12g"
    frame.to_csv(out / f"{name}.csv", index=False, float_format=precision)


def run(data: pd.DataFrame, out: Path):
    """Fit the four classical candidates; out is the classical result directory."""
    out.mkdir(parents=True, exist_ok=True)
    all_predictions, all_params, scores, counts, class_scores = [], [], [], [], []
    with threadpool_limits(limits=1):
        for model in GRIDS:
            frame, params = _oof(data, model, FEATURES)
            all_predictions.append(frame)
            all_params.append(params)
            scores.append({"model": model, **_metrics(frame)})
            cm = confusion_matrix(frame.actual, frame.prediction, labels=CLASSES)
            for i, label in enumerate(CLASSES):
                support, called = cm[i].sum(), cm[:, i].sum()
                class_scores.append({
                    "model": model, "class": label, "actual_n": support, "predicted_n": called,
                    "correct": cm[i, i], "recall": cm[i, i]/support,
                    "precision": cm[i, i]/called if called else np.nan,
                    "f1": 2*cm[i, i]/(support + called),
                })
                counts.append({"model": model, "actual": label, **dict(zip(CLASSES, cm[i]))})
    predictions = pd.concat(all_predictions, ignore_index=True)
    _save(predictions, out, "oof")
    _save(pd.DataFrame(scores), out, "metrics")
    _save(pd.concat(all_params, ignore_index=True), out, "selection")
    _save(pd.DataFrame(counts), out, "confusion")
    _save(pd.DataFrame(class_scores), out, "class_scores")
    _save(_date_bootstrap(predictions), out, "paired_date_intervals")
    return predictions


def ablate(data: pd.DataFrame, out: Path, model: str):
    """Re-tune the selected classical model in the same folds after dropping each feature group."""
    if model not in GRIDS:
        raise ValueError(f"Ablation requires a classical candidate; received {model}")
    # Recompute the full model, so stale artifacts cannot change the reference.
    out.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    with threadpool_limits(limits=1):
        for group, dropped in {"none": (), **DROP_GROUPS}.items():
            features = [f for f in FEATURES if f not in dropped]
            frame, _ = _oof(data, model, features)
            rows.append({"model": model, "omitted": group, **_metrics(frame)})
            predictions.append(frame.assign(omitted=group))
    result = pd.DataFrame(rows)
    for metric in ("macro_f1", "balanced_accuracy", "accuracy", "log_loss"):
        result[f"delta_{metric}"] = result[metric] - result.loc[0, metric]
    _save(result, out, "classical_ablation")
    _save(pd.concat(predictions, ignore_index=True), out, "classical_ablation_oof")
    return result
