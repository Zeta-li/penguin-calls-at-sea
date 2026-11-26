"""Out-of-group predictions and paired, fixed-prediction cluster bootstrap."""
import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from frozen_models import ASSETS, SEED, digest, mitra, setup, tabicl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from models_playback import design
from penguin_data import CALLS, FEATURES, input_path


def inputs(task):
    if task == "q1":
        data = pd.read_csv(input_path(1))
        baseline = pd.read_csv(PROJECT / "results/q1_out_of_date_predictions.csv")
        assert np.array_equal(data.Date, baseline.date) and np.array_equal(data.CallType, baseline.actual)
        x = data[list(FEATURES)].to_numpy()
        y = pd.Categorical(data.CallType, categories=CALLS).codes.copy()
        groups = pd.to_datetime(data.Date, format="%d/%m/%Y").to_numpy()
        reference = baseline[[f"p_{k}" for k in CALLS]].to_numpy()
        return x, y, groups, reference, LeaveOneGroupOut()
    data = pd.read_csv(input_path(2))
    baseline = pd.read_csv(PROJECT / "results/q3_held_out_probabilities.csv")
    for first, second in (("expID", "expID"), ("Type", "type"), ("CallType", "stimulus")):
        assert np.array_equal(data[first], baseline[second])
    x = design(data.CallType, data.Context, data.Distance, data.Npenguins)[:, 1:]
    y = data.ResponseYesNo.to_numpy(dtype=int)
    assert np.array_equal(y, baseline.state.ne(0))
    p0 = baseline.p_000.to_numpy()
    return x, y, data.expID.to_numpy(), np.column_stack([p0, 1-p0]), GroupKFold(5)


def predictions(task, cache, frozen, smoke):
    x, y, groups, reference, splitter = inputs(task)
    estimators = ["reference", "TabICLv2"] + (["Mitra-v2"] if task == "q1" else [])
    tables = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
        assert not np.intersect1d(groups[train], groups[test]).size
        assert set(y[train]) == set(np.arange(reference.shape[1]))
        results = [reference[test], tabicl(cache, x[train], y[train], x[test])]
        if task == "q1":
            results.append(mitra(frozen, x[train], y[train], x[test]))
        frame = pd.DataFrame({"row": test+1, "group": groups[test], "fold": fold, "actual": y[test]})
        for name, p in zip(estimators, results):
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
            for k, col in enumerate(p.T):
                frame[f"{name}_{k}"] = col.astype(float)
        tables.append(frame)
        if smoke:
            break
    out = pd.concat(tables).sort_values("row").reset_index(drop=True)
    assert not out.row.duplicated().any()
    if not smoke:
        np.testing.assert_array_equal(out.row, np.arange(1, len(y)+1))
    return out, estimators


def score(y, p, weights):
    # One weight per observation per bootstrap draw.
    pred = p.argmax(1)
    nclass = p.shape[1]
    total = weights.sum(axis=1)
    rows = np.arange(len(y))
    n = lambda mask: weights @ mask.astype(float)
    correct = np.stack([n((y == k) & (pred == k)) for k in range(nclass)], 1)
    actual = np.stack([n(y == k) for k in range(nclass)], 1)
    guessed = np.stack([n(pred == k) for k in range(nclass)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = correct / actual
        f1 = np.divide(2*correct, actual+guessed, out=np.zeros_like(correct), where=(actual+guessed)>0)
    return {"log_loss": weights @ -np.log(np.clip(p[rows, y], 1e-15, 1)) / total,
            "brier": weights @ ((p[:, 1]-y)**2) / total if nclass == 2 else
                     weights @ ((p - np.eye(nclass)[y])**2).sum(axis=1) / total,
            "accuracy": correct.sum(axis=1)/total, "balanced_accuracy": recall.mean(axis=1),
            "macro_f1": f1.mean(axis=1)}


def summarize(task, frame, names):
    counts = frame.group.nunique()
    index = pd.factorize(frame.group, sort=True)[0]
    rng = np.random.default_rng(SEED + (1 if task == "q1" else 3))
    draws = rng.multinomial(counts, np.full(counts, 1/counts), size=5000)
    weights = np.vstack((np.ones(len(frame)), draws[:, index]))
    y = frame.actual.to_numpy()
    rows, classes, reference = [], [], None
    for name in names:
        p = frame.filter(regex="^"+name+"_[0-9]+$").to_numpy()
        scores = score(y, p, weights)
        if reference is None:
            reference = scores
        for metric, values in scores.items():
            delta = values - reference[metric]
            valid = np.isfinite(delta[1:])
            lo, hi = np.quantile(delta[1:][valid], [.025,.975])
            rows.append((task,name,metric,values[0],delta[0],lo,hi,int(sum(valid))))
        for k in range(p.shape[1]):
            mask = y == k
            classes.append((task,name,k,np.sum(mask & (p.argmax(axis=1)==k)),mask.sum()))
    return rows, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / ".model_cache")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output == HERE / "results":
        args.output = HERE / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    model = setup(cache)
    rows, classes = [], []
    for task in ("q1", "q3"):
        frame, names = predictions(task, cache, model, args.smoke)
        frame.to_csv(args.output / f"{task}_oof.csv", index=False, float_format="%.12g")
        print(task, "rows", len(frame), "folds", frame.fold.nunique(), flush=True)
        if not args.smoke:
            metrics, correct = summarize(task, frame, names)
            rows.extend(metrics)
            classes.extend(correct)
    if rows:
        pd.DataFrame(rows,columns=["task","model","metric","estimate","delta","lower","upper","draws"]).to_csv(
            args.output/"scores.csv", index=False, float_format="%.10g")
        pd.DataFrame(classes,columns=["task","model","class","correct","total"]).to_csv(
            args.output/"classes.csv",index=False)
    evidence = [input_path(1), input_path(2), PROJECT/"results/q1_out_of_date_predictions.csv",
                PROJECT/"results/q3_held_out_probabilities.csv", PROJECT/"models_playback.py", PROJECT/"penguin_data.py",
                HERE/"evaluate.py", HERE/"frozen_models.py"]
    manifest = {"seed":SEED, "smoke":args.smoke, "bootstrap_draws":5000,
                "python":sys.version.split()[0], "versions":{k:version(k) for k in
                ["torch","tabicl","numpy","pandas","scikit-learn","einx","einops","safetensors","huggingface-hub"]},
                "sources":{str(f.relative_to(PROJECT)):digest(f) for f in evidence},
                "assets":{k:v[1] for k,v in ASSETS.items()},
                "results":{f.name:digest(f) for f in sorted(args.output.glob("*.csv"))}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
