"""Six-measurement challenger with fully date-nested training and paired comparisons."""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = np.array(["flat", "mod", "tv"])
FIELDS = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
PROB = ["p_"+n for n in NAMES]
PRIOR = ["train_prior_"+n for n in NAMES]
RULES = ("ordinary", "balanced")
SEED = 20261126
N_BOOT = 5000


@dataclass(frozen=True)
class Candidate:
    """Parameter is Gaussian bandwidth for kernel, leaf size for trees."""
    family: str
    representation: str
    parameter: float

    @property
    def key(self):
        return f"{self.family}_{self.representation}_{self.parameter:g}"


CANDIDATES = [
    Candidate(family, representation, value)
    for family, settings in [("kernel", (.35, .65, 1.)), ("trees", (2, 5, 10))]
    for representation in ("common", "acoustic")
    for value in settings
]

RESTRICTIONS = {
    "classical_nested": lambda model, rule: True,
    "common_six_only": lambda model, rule: model.representation == "common",
    "ordinary_decision_only": lambda model, rule: rule == "ordinary",
    "kernel_only": lambda model, rule: model.family == "kernel",
    "trees_only": lambda model, rule: model.family == "trees",
}


def extract(data):
    """All functions are pointwise in a single recording; frequencies use Hz."""
    raw = data[FIELDS].to_numpy(dtype=float)
    assert raw.shape[1] == 6 and np.all(np.isfinite(raw)) and np.min(raw) > 0
    _, f0, peak, q25, q50, q75 = raw.T
    assert np.all((q25 <= q50) & (q50 <= q75))
    log_six = np.log(raw)
    width = np.maximum(q75-q25, 1.)
    ratio = peak/f0
    extra = np.column_stack([np.log(ratio), np.abs(ratio-np.rint(ratio)),
                             np.log(q75/q25), np.log(q50/f0),
                             (q50-q25)/width, (peak-q25)/width])
    return {"common": log_six, "acoustic": np.column_stack((log_six, extra))}


def class_prior(y):
    counts = np.bincount(y, minlength=3)
    assert (counts > 0).all()
    return counts/len(y)


def predict_candidate(config, X, y, train, test):
    x = X[config.representation]
    prior = class_prior(y[train])
    if config.family == "kernel":
        center = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale == 0] = 1.
        v = (x[train]-center)/scale
        q = (x[test]-center)/scale
        dist = cdist(q, v, metric="sqeuclidean")/v.shape[1]
        weights = np.exp(-dist/(2*config.parameter**2))
        counts = weights @ np.eye(3)[y[train]] + 2*prior
        return counts/counts.sum(axis=1, keepdims=True)
    model = ExtraTreesClassifier(n_estimators=128, min_samples_leaf=int(config.parameter),
                                 max_features=1., random_state=SEED, n_jobs=1)
    model.fit(x[train], y[train])
    assert np.array_equal(model.classes_, np.arange(3))
    return .98*model.predict_proba(x[test]) + .02*prior


def decisions(p, ratio_prior, rule):
    return (p/ratio_prior if rule == "balanced" else p).argmax(axis=1)


def scores(y, p, pred):
    recall = [(pred[y == k] == k).mean() for k in range(3)]
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.))
    return [np.mean(recall), loss.mean(), (y == pred).mean(), *recall]


def validate_outer(X, y, dates, *, smoke=False, chronological=False):
    predictions, tuning, selections, splits = [], [], [], []
    dates_unique = np.unique(dates)
    if chronological:
        latest_year = max(d[:4] for d in dates_unique)
        last = np.array([d[:4] == latest_year for d in dates])
        outer_splits = [(latest_year, np.flatnonzero(~last), np.flatnonzero(last))]
    else:
        outer_splits = [(day, np.flatnonzero(dates != day), np.flatnonzero(dates == day))
                        for day in dates_unique]
    for outer, (held, train, test) in enumerate(outer_splits, start=1):
        selected_prior = class_prior(y[train])
        inner_splits = list(GroupKFold(4).split(train, y[train], dates[train]))
        val_prior = np.zeros((len(train), 3))
        seen = np.zeros(len(train), dtype=int)
        for inner, (sub, val) in enumerate(inner_splits, start=1):
            assert not set(dates[train[sub]]) & set(dates[train[val]])
            seen[val] += 1
            val_prior[val] = class_prior(y[train[sub]])
            splits.extend((outer, held, inner, int(i+1), dates[i]) for i in train[val])
        assert np.all(seen == 1)
        ranking = []
        for cfg in CANDIDATES:
            inner_p = np.empty((len(train), 3))
            for sub, val in inner_splits:
                inner_p[val] = predict_candidate(cfg, X, y, train[sub], train[val])
            for rule in RULES:
                pred = decisions(inner_p, val_prior, rule)
                ba, ll = scores(y[train], inner_p, pred)[:2]
                ranking.append((cfg, rule, ba, ll))
                tuning.append((outer, held, cfg.key, cfg.family, cfg.representation,
                               cfg.parameter, rule, ba, ll, len(train)))
        # Stable sort preserves the preregistered order for tied validation scores.
        ranking.sort(key=lambda r: (-round(r[2], 12), r[3]))
        pred_by_cfg = {}
        for method, allowed in RESTRICTIONS.items():
            cfg, rule, ba, ll = next(row for row in ranking if allowed(row[0], row[1]))
            if cfg not in pred_by_cfg:
                pred_by_cfg[cfg] = predict_candidate(cfg, X, y, train, test)
            p = pred_by_cfg[cfg]
            label = decisions(p, selected_prior, rule)
            selections.append((outer, held, method, cfg.key, rule, ba, ll, len(train), len(test)))
            for i, prob, chosen in zip(test, p, label):
                predictions.append((i+1, dates[i], NAMES[y[i]], method, *prob, NAMES[chosen],
                                    *selected_prior, cfg.key, rule, outer))
        # The training class-prior baseline makes ordinary prior-only decisions.
        for i in test:
            predictions.append((i+1, dates[i], NAMES[y[i]], "train_prior", *selected_prior,
                                NAMES[selected_prior.argmax()], *selected_prior, "prior", "ordinary", outer))
        if smoke:
            break
    oof = pd.DataFrame(predictions, columns=["row", "date", "actual", "method", *PROB,
                       "decision", *PRIOR, "candidate", "decision_rule", "fold"])
    tune = pd.DataFrame(tuning, columns=["fold", "held_date", "candidate", "family",
                        "representation", "parameter", "decision_rule", "inner_balanced_accuracy",
                        "inner_log_loss", "n_train"])
    selected = pd.DataFrame(selections, columns=["fold", "held_date", "method", "candidate",
                            "decision_rule", "inner_balanced_accuracy", "inner_log_loss", "n_train", "n_test"])
    inner = pd.DataFrame(splits, columns=["fold", "held_date", "inner_fold", "validation_row", "validation_date"])
    return oof, tune, selected, inner


def benchmarks(oof):
    main = oof[oof.method == "classical_nested"].sort_values("row").reset_index(drop=True)
    archived = pd.read_csv(ROOT/"第四版实验/results/q1_predictions.csv")
    out = []
    for method in ("Ridge", "RDA", "Mitra-v2", "TabICLv2"):
        rows = archived[archived.method == method].sort_values("row")
        np.testing.assert_array_equal(rows.row, main.row)
        np.testing.assert_array_equal(rows.actual, main.actual)
        np.testing.assert_array_equal(rows.held_date, main.date)
        for rule in (RULES if method == "TabICLv2" else ("ordinary",)):
            copy = main.copy()
            p = rows[PROB].to_numpy()
            copy[PROB] = p
            copy["decision"] = NAMES[decisions(p, copy[PRIOR].to_numpy(), rule)]
            copy["decision_rule"] = rule
            copy["candidate"] = method
            copy["method"] = f"v4_{method}_{rule}"
            out.append(copy)
    return pd.concat([oof, *out], ignore_index=True).sort_values(["method", "row"])


def summarise(oof):
    base = oof[oof.method == "v4_TabICLv2_ordinary"].sort_values("row")
    y = pd.Categorical(base.actual, categories=NAMES).codes.astype(int)
    dates, groups = np.unique(base.date, return_inverse=True)
    rng = np.random.default_rng(SEED)
    cluster = rng.multinomial(len(dates), np.full(len(dates), 1/len(dates)), size=N_BOOT)
    row_w = cluster[:, groups]
    den_class = row_w @ np.eye(3)[y]
    valid = (den_class > 0).all(axis=1)
    signs, results, intervals = {}, [], []
    for method, subset in oof.groupby("method", sort=False):
        subset = subset.sort_values("row")
        np.testing.assert_array_equal(subset.row, base.row)
        p = subset[PROB].to_numpy()
        assert np.allclose(p.sum(axis=1), 1., atol=1e-5) and np.all(p >= 0)
        pred = pd.Categorical(subset.decision, categories=NAMES).codes
        metrics = scores(y, p, pred)
        results.append((method, len(y), *metrics))
        num_class = row_w @ (np.eye(3)[y]*(pred == y)[:, None])
        b_bal = (num_class[valid]/den_class[valid]).mean(axis=1)
        loss = -np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
        b_loss = (row_w @ loss)/row_w.sum(axis=1)
        signs[method] = (metrics, b_bal, b_loss)
    for method, (metrics, b_bal, b_loss) in signs.items():
        for ref in ("v4_TabICLv2_ordinary", "v4_TabICLv2_balanced"):
            base_m, ref_ba, ref_ll = signs[ref]
            for ix, name, draws in ((0, "balanced_accuracy", b_bal-ref_ba),
                                    (1, "log_loss", b_loss-ref_ll)):
                intervals.append((method, ref, name, metrics[ix]-base_m[ix],
                                  *np.quantile(draws, [.025,.975]), len(draws)))
    return (pd.DataFrame(results, columns=["method", "n", "balanced_accuracy", "log_loss", "accuracy",
                                          *["recall_"+s for s in NAMES]]),
            pd.DataFrame(intervals, columns=["method","reference","metric","difference","q025","q975","valid_draws"]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    source = list((ROOT/"date").glob("Data01_*.csv"))
    assert len(source) == 1
    # Even ingestion excludes F02, PulseRate and Location.
    df = pd.read_csv(source[0], usecols=["Date","CallType", *FIELDS])
    dates = pd.to_datetime(df.Date, format="%d/%m/%Y").dt.strftime("%Y-%m-%d").to_numpy()
    y = pd.Categorical(df.CallType, categories=NAMES).codes.astype(int)
    assert y.min() == 0 and len(y) == 108 and len(np.unique(dates)) == 16
    oof, tune, selected, splits = validate_outer(extract(df), y, dates, smoke=args.smoke)
    if args.smoke:
        assert len(tune) == 24 and np.allclose(oof[PROB].sum(axis=1), 1.)
        print(f"First date {oof.date.iloc[0]}; {len(tune)} candidate rules; probability checks pass")
        return
    temporal, year_tune, year_sel, year_splits = validate_outer(extract(df), y, dates, chronological=True)
    per_year = []
    for method, frame in temporal.groupby("method"):
        truth = pd.Categorical(frame.actual, categories=NAMES).codes.astype(int)
        pred = pd.Categorical(frame.decision, categories=NAMES).codes
        per_year.append((method, frame.date.min(), frame.date.max(), len(frame), frame.date.nunique(),
                         *scores(truth, frame[PROB].to_numpy(), pred)))
    year_scores = pd.DataFrame(per_year, columns=["method","first_test_date","last_test_date","test_n",
                  "test_dates","balanced_accuracy","log_loss","accuracy",*["recall_"+s for s in NAMES]])
    merged = benchmarks(oof)
    performance, paired = summarise(merged)
    # Post-fit descriptive exhibit only; it cannot affect any held-out prediction.
    ratio = (df.Fpeak/df.F01).to_numpy()
    offset = np.abs(ratio-np.rint(ratio))
    quantiles = []
    for k, name in enumerate(NAMES):
        z = offset[y == k]
        quantiles.append((name, len(z), *np.quantile(z, [.25, .5, .75])))
    harmonic = pd.DataFrame(quantiles, columns=["class", "n", "q25", "median", "q75"])
    args.output.mkdir(parents=True,exist_ok=True)
    for key, frame in [("oof",merged),("inner",tune),("selection",selected),
                       ("splits",splits),("scores",performance),("paired",paired),("harmonic_description",harmonic),
                       ("temporal_oof",temporal),("temporal_inner",year_tune),("temporal_selection",year_sel),
                       ("temporal_splits",year_splits),("temporal_scores",year_scores)]:
        frame.to_csv(args.output/f"q1c_{key}.csv",index=False,float_format="%.12g")
    print(performance[["method","balanced_accuracy","log_loss","accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
