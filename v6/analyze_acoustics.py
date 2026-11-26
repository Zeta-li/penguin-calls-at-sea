"""Full Q1 and Q2 analysis."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from models_acoustic import (
    BLOCK_SAMPLES, C_GRID, EQUAL_WEIGHTS, THRESHOLDS, acoustic_range, block_means,
    evaluate_nodes, monotone_nodes, smooth_fit, smooth_value, train_classifier,
)
from penguin_data import CALLS, DISTANCES, FEATURES, METRICS, SEED

FEATURE_GROUPS = {
    "Duration": ("Duration",),
    "First fundamental": ("F01",),
    "Peak frequency": ("Fpeak",),
    "Spectral quartiles": ("Q25", "Q50", "Q75"),
}
WEIGHTS = {
    "equal": EQUAL_WEIGHTS,
    "spectrogram": np.array([0.5, 0.25, 0.25]),
    "spectrum": np.array([0.25, 0.5, 0.25]),
    "envelope": np.array([0.25, 0.25, 0.5]),
}
SCAN_THRESHOLDS = np.arange(40, 81) / 100
N_PERMUTATIONS = 3999
N_BOOT = 500


def write_table(table, results, name):
    table.to_csv(results / f"{name}.csv", index=False, float_format="%.10g")


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.clip(np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)), 0, 1)
    return adjusted


def rank_tests(data, rng):
    x = data[list(FEATURES)].to_numpy()
    dates = data.session_date.to_numpy()
    codes = pd.Categorical(data.CallType, categories=CALLS).codes
    ranks = rankdata(x, axis=0)
    n = len(x)
    group_size = np.bincount(codes)
    blocks = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    expected = (n + 1) / 2
    correction = np.array([
        1.0 - np.sum(counts**3 - counts) / (n**3 - n)
        for counts in (np.unique(x[:, i], return_counts=True)[1] for i in range(x.shape[1]))
    ])

    def statistic(labels):
        rank_sums = np.stack([ranks[labels == k].sum(axis=0) for k in range(len(CALLS))])
        terms = (rank_sums - expected * group_size[:, None]) ** 2 / group_size[:, None]
        return 12 * terms.sum(axis=0) / (n * (n + 1) * correction)

    base = statistic(codes)
    more_extreme = np.ones(len(FEATURES))
    for _ in range(N_PERMUTATIONS):
        shuffled = codes.copy()
        for block in blocks:
            shuffled[block] = rng.permutation(codes[block])
        more_extreme += statistic(shuffled) >= base - 1e-10
    conditional_p = more_extreme / (N_PERMUTATIONS + 1)
    standard = [kruskal(*(data.loc[data.CallType.eq(k), f] for k in CALLS)) for f in FEATURES]
    assert np.allclose(base, [item.statistic for item in standard])
    omnibus = pd.DataFrame({
        "feature": FEATURES, "H": base, "p_asymptotic": [test.pvalue for test in standard],
        "p_date_permutation": conditional_p, "p_date_holm": holm(conditional_p),
    })

    records = []
    for feature in FEATURES + ("PulseRate",):
        pairs = (("mod", "tv"),) if feature == "PulseRate" else combinations(CALLS, 2)
        for first, second in pairs:
            a = data.loc[data.CallType.eq(first), feature].dropna()
            b = data.loc[data.CallType.eq(second), feature].dropna()
            test = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            records.append({
                "feature": feature, "first": first, "second": second, "n_first": len(a),
                "n_second": len(b), "U": test.statistic, "p": test.pvalue,
                "cliffs_delta": 2 * test.statistic / (len(a) * len(b)) - 1,
            })
    pairwise = pd.DataFrame(records)
    primary = pairwise.feature.ne("PulseRate")
    pairwise.loc[primary, "p_holm"] = holm(pairwise.loc[primary, "p"])
    pairwise.loc[~primary, "p_holm"] = pairwise.loc[~primary, "p"]
    return omnibus, pairwise


def pick_c(x, y, groups):
    split = list(GroupKFold(4).split(x, y, groups))
    scores = []
    for c in C_GRID:
        total = 0.0
        for train, test in split:
            proba = train_classifier(x[train], y[train], c).predict_proba(x[test])
            total += log_loss(y[test], proba, labels=CALLS, normalize=False)
        scores.append(total / len(y))
    return C_GRID[int(np.argmin(scores))]


def date_predictions(x, y, groups):
    probability = np.empty((len(x), len(CALLS)))
    baseline = np.empty_like(probability)
    selection = np.empty(len(x))
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        assert not set(groups[train]) & set(groups[test])
        c = pick_c(x[train], y[train], groups[train])
        model = train_classifier(x[train], y[train], c)
        probability[test] = model.predict_proba(x[test])
        baseline[test] = [(y[train] == label).mean() for label in CALLS]
        selection[test] = c
    return probability, baseline, selection


def accuracy_scores(y, p):
    guess = np.array(CALLS)[p.argmax(axis=1)]
    return np.array([
        accuracy_score(y, guess),
        balanced_accuracy_score(y, guess),
        f1_score(y, guess, labels=CALLS, average="macro", zero_division=0),
        log_loss(y, p, labels=CALLS),
    ])


def quantify_acoustics(data, results):
    rng = np.random.default_rng(SEED + 1)
    descriptors = []
    for feature in ("Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"):
        for call in CALLS:
            observed = data.loc[data.CallType.eq(call), feature].dropna()
            quartiles = observed.quantile([0.25, 0.50, 0.75]).to_numpy()
            descriptors.append({
                "feature": feature, "call": call, "available_n": len(observed),
                "unit": "s" if feature == "Duration" else "Hz",
                "q25": quartiles[0], "median": quartiles[1], "q75": quartiles[2],
                "min": observed.min(), "max": observed.max(),
            })
    description = pd.DataFrame(descriptors)
    omnibus, pairwise = rank_tests(data, rng)
    y = data.CallType.to_numpy()
    groups = data.session_date.to_numpy()
    x = data[list(FEATURES)].to_numpy()
    predicted, baseline, choices = date_predictions(x, y, groups)
    point = accuracy_scores(y, predicted)
    date_rows = [np.flatnonzero(groups == date) for date in np.unique(groups)]
    bootstrap = []
    while len(bootstrap) < N_BOOT:
        rows = np.concatenate([date_rows[k] for k in rng.integers(len(date_rows), size=len(date_rows))])
        if set(y[rows]) == set(CALLS):
            bootstrap.append(accuracy_scores(y[rows], predicted[rows]))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    performance = pd.DataFrame({
        "metric": ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"], "estimate": point,
        "lower": lower, "upper": upper, "training_prior_baseline": accuracy_scores(y, baseline),
    })
    ablations = []
    for label, excluded in FEATURE_GROUPS.items():
        subset = [j for j, name in enumerate(FEATURES) if name not in excluded]
        smaller, _, _ = date_predictions(x[:, subset], y, groups)
        score = accuracy_scores(y, smaller)
        ablations.append((label, score[1], score[-1], score[-1] - point[-1]))
    ablation = pd.DataFrame(ablations, columns=["dropped", "balanced_accuracy", "log_loss", "change_log_loss"])
    final_c = pick_c(x, y, groups)
    final = train_classifier(x, y, final_c)
    scaler = final.named_steps["standardscaler"]
    coefficient = final.named_steps["logisticregression"]
    predictions = pd.DataFrame({
        "date": data.Date, "actual": y, "predicted": np.array(CALLS)[predicted.argmax(axis=1)],
        "inner_selected_C": choices,
        **{f"p_{call}": predicted[:, i] for i, call in enumerate(CALLS)},
    })
    parameter = pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "scale": scaler.scale_})
    for i, call in enumerate(CALLS):
        parameter[f"coefficient_{call}"] = coefficient.coef_[i]
    intercept = pd.DataFrame([{
        "feature": "intercept", **{f"coefficient_{call}": coefficient.intercept_[i] for i, call in enumerate(CALLS)}
    }])
    parameter = pd.concat([parameter, intercept], ignore_index=True)
    parameter["final_C"] = final_c
    for table, name in (
        (description, "q1_description"), (omnibus, "q1_rank_tests"),
        (pairwise, "q1_pairwise"), (performance, "q1_performance"),
        (predictions, "q1_out_of_date_predictions"), (ablation, "q1_ablation"),
        (parameter, "q1_classifier_parameters"),
    ):
        write_table(table, results, name)
    return {"data": data, "predictions": predictions, "omnibus": omnibus,
            "ablation": ablation, "performance": performance}


def quantify_propagation(correlation, playback, results):
    means, counts = block_means(correlation)
    block_summary = np.nanmean(means, axis=2)
    nodes = monotone_nodes(block_summary)
    row_nodes = monotone_nodes(np.nanmean(correlation, axis=2))
    sampled_nodes = np.stack([
        monotone_nodes(np.nanmean(np.take(means, draw, axis=2), axis=2))
        for draw in BLOCK_SAMPLES
    ])
    deleted_nodes = np.stack([
        monotone_nodes(np.nanmean(np.delete(means, b, axis=2), axis=2)) for b in range(3)
    ])
    composite = EQUAL_WEIGHTS @ nodes
    dense = 2.0 ** np.linspace(2, 8, 241)
    records, range_records, sensitivity, validation, curves, long = [], [], [], [], [], []
    for k, call in enumerate(CALLS):
        fit = smooth_fit(DISTANCES, EQUAL_WEIGHTS @ block_summary[k])
        for m, metric in enumerate(METRICS):
            for i, distance in enumerate(DISTANCES):
                records.append((call, metric, distance, counts[k, m, :, i].sum(),
                                np.nanmean(correlation[k, m, :, i]), block_summary[k, m, i],
                                nodes[k, m, i], *means[k, m, :, i]))
        for tau in THRESHOLDS:
            estimate, status = acoustic_range(composite[k], tau)
            draws = [acoustic_range(EQUAL_WEIGHTS @ sample[k], tau) for sample in sampled_nodes]
            loo = [acoustic_range(EQUAL_WEIGHTS @ sample[k], tau)[0] for sample in deleted_nodes]
            lo, hi = np.quantile([result[0] for result in draws], [0.025, 0.975])
            status_list = [result[1] for result in draws]
            range_records.append((call, tau, estimate, status, lo, hi, min(loo), max(loo),
                                  status_list.count("below_4m"), status_list.count("right_boundary"),
                                  *acoustic_range(EQUAL_WEIGHTS @ row_nodes[k], tau)))
        for label, weights in WEIGHTS.items():
            scores = weights @ nodes[k]
            for tau in SCAN_THRESHOLDS:
                radius, status = acoustic_range(scores, tau)
                sensitivity.append((call, label, tau, radius, status))
        for j in range(1, len(DISTANCES) - 1):
            keep = np.arange(len(DISTANCES)) != j
            train_d = DISTANCES[keep]
            iso = monotone_nodes(block_summary[k][:, keep], train_d)
            smooth = smooth_fit(train_d, EQUAL_WEIGHTS @ block_summary[k][:, keep])
            validation.append((call, DISTANCES[j], EQUAL_WEIGHTS @ block_summary[k, :, j],
                               evaluate_nodes(EQUAL_WEIGHTS @ iso, DISTANCES[j], train_d),
                               smooth_value(smooth, DISTANCES[j])))
        for distance, fitted, alternative in zip(
            dense, evaluate_nodes(composite[k], dense), smooth_value(fit, dense)
        ):
            curves.append((call, distance, fitted, alternative))
        far = dense[dense >= 64 - 1e-9]
        area = np.trapezoid(evaluate_nodes(composite[k], far), np.log2(far)) / 2.0
        long.append((call, area, *evaluate_nodes(composite[k], [64, 128, 256])))
    fields = [
        "call", "representation", "distance_m", "observed_entries", "row_mean", "block_mean",
        "fitted", "block_1", "block_2", "block_3",
    ]
    measurements = pd.DataFrame(records, columns=fields)
    ranges = pd.DataFrame(range_records, columns=[
        "call", "threshold", "distance_m", "status", "lower_m", "upper_m", "leave_one_min_m",
        "leave_one_max_m", "draws_below", "draws_right_boundary", "row_mean_distance_m", "row_status",
    ])
    sensitivities = pd.DataFrame(sensitivity, columns=["call", "weights", "threshold", "distance_m", "status"])
    diagnostics = pd.DataFrame(validation, columns=[
        "call", "distance_m", "held_out_block_mean", "isotonic_prediction", "smooth_prediction",
    ])
    curve = pd.DataFrame(curves, columns=["call", "distance_m", "fidelity", "smooth_fidelity"])
    far_range = pd.DataFrame(long, columns=["call", "log_distance_area", "at_64m", "at_128m", "at_256m"])
    exposed = playback[playback.Type.eq("call")][["expID", "CallType", "Distance", "Context"]].copy()
    exposed["fidelity"] = np.nan
    for i, call in enumerate(CALLS):
        matched = exposed.CallType.eq(call)
        exposed.loc[matched, "fidelity"] = evaluate_nodes(composite[i], exposed.loc[matched, "Distance"])
    assert np.isfinite(exposed.fidelity).all()
    for tau in THRESHOLDS:
        exposed[f"at_least_{tau:.1f}"] = exposed.fidelity.ge(tau)
    for table, name in (
        (measurements, "q2_propagation_nodes"), (ranges, "q2_ranges"),
        (sensitivities, "q2_sensitivity"), (diagnostics, "q2_held_out_distances"),
        (curve, "q2_curves"), (far_range, "q2_long_distance"),
        (exposed, "q2_playback_exposure"),
    ):
        write_table(table, results, name)
    return {
        "measurements": measurements, "node_fit": composite, "sampled_nodes": EQUAL_WEIGHTS @ sampled_nodes,
        "ranges": ranges, "sensitivity": sensitivities, "diagnostics": diagnostics, "curve": curve,
        "far_range": far_range,
    }
