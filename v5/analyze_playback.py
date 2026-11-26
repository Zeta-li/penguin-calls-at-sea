"""Full paired playback response and onset analysis."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.model_selection import GroupKFold

from analyze_acoustics import N_BOOT, N_PERMUTATIONS, write_table
from models_playback import (
    FEATURE_NAMES, LAMBDAS, LATENCY_SHIFT, design, response_fit, response_log_probs,
    response_probs, time_cdf, time_fit,
)
from penguin_data import (
    BEHAVIOURS, CONTEXTS, HORIZON, REPRESENTATIVE_N, SEED, SHARED_RANGE,
    STATE_NAMES, STATES, STIMULI, context_weights, event_flags, latency_values, state_index,
)

GRID_DISTANCES = np.array([50, 100, 150])
SCORE_TIMES = np.array([3, 10, 30, 60])


def testing_rates(playback, y, rng):
    tab = pd.crosstab(playback.Type, y).reindex(index=["call", "control"], columns=range(8), fill_value=0)
    frequencies = tab.rename(columns=dict(enumerate(STATE_NAMES))).reset_index()
    cells = playback.assign(state=y).pivot(index="expID", columns="Type", values="state")
    natural = cells["call"].to_numpy()
    control = cells["control"].to_numpy()
    first_only = np.count_nonzero((natural > 0) & (control == 0))
    second_only = np.count_nonzero((natural == 0) & (control > 0))
    marginal = np.eye(8)[natural] - np.eye(8)[control]
    signs = rng.choice([-1, 1], size=(N_PERMUTATIONS, len(cells)))
    permutations = np.sum((signs @ marginal) ** 2, axis=1)
    observed = np.sum(marginal.sum(axis=0) ** 2)
    indices = rng.integers(len(cells), size=(N_BOOT, len(cells)))
    diff = (natural > 0).astype(int) - (control > 0).astype(int)
    row = {
        "n_pairs": len(cells), "natural_any": int(np.sum(natural > 0)), "control_any": int(np.sum(control > 0)),
        "natural_rate": np.mean(natural > 0), "control_rate": np.mean(control > 0),
        "call_only": first_only, "control_only": second_only, "risk_difference": diff.mean(),
        "paired_lower": np.quantile(diff[indices].mean(axis=1), 0.025),
        "paired_upper": np.quantile(diff[indices].mean(axis=1), 0.975),
        "exact_mcnemar_p": binomtest(first_only, first_only + second_only).pvalue,
        "paired_permutation_statistic": observed,
        "paired_permutation_p": (1 + np.count_nonzero(permutations >= observed)) / (N_PERMUTATIONS + 1),
    }
    return pd.DataFrame([row]), frequencies


def select_penalty(x, y, groups, n_folds, dependent):
    splits = list(GroupKFold(n_folds).split(x, y, groups))
    total = []
    for penalty in LAMBDAS:
        accumulated = 0.0
        for train, test in splits:
            beta = response_fit(x[train], y[train], penalty, dependent)
            accumulated -= response_log_probs(beta, x[test])[np.arange(len(test)), y[test]].sum()
        total.append(accumulated / len(y))
    return LAMBDAS[int(np.argmin(total))], total


def fit_timing(x, flags, onset, shift=LATENCY_SHIFT):
    return [time_fit(x[flags[:, j]], onset[flags[:, j], j], shift) for j in range(3)]


def cross_validate(x, y, groups, flags, onset, dependent, penalty=None, timing=False):
    probabilities = np.empty((len(y), 8))
    times = np.full((len(y), 3, len(SCORE_TIMES)), np.nan)
    reference = np.full_like(times, np.nan)
    observed = flags[..., None] & (onset[..., None] <= SCORE_TIMES)
    for train, test in GroupKFold(5).split(x, y, groups):
        assert not set(groups[train]) & set(groups[test])
        strength = penalty
        if strength is None:
            strength, _ = select_penalty(x[train], y[train], groups[train], 4, dependent)
        beta = response_fit(x[train], y[train], strength, dependent)
        probabilities[test] = response_probs(beta, x[test])
        if timing:
            fits = fit_timing(x[train], flags[train], onset[train])
            marginal = probabilities[test] @ STATES
            reference[test] = observed[train].mean(axis=0)
            for j, fit in enumerate(fits):
                if fit is not None:
                    for i, time in enumerate(SCORE_TIMES):
                        times[test, j, i] = marginal[:, j] * time_cdf(fit, x[test], time)
    assert np.allclose(probabilities.sum(axis=1), 1)
    risk = {
        "multiclass_log_loss": -np.log(probabilities[np.arange(len(y)), y]).mean(),
        "multiclass_brier": np.square(probabilities - np.eye(8)[y]).sum(axis=1).mean(),
        "any_brier": np.square((1 - probabilities[:, 0]) - (y > 0)).mean(),
    }
    timing_records = []
    if timing:
        for j, behaviour in enumerate(BEHAVIOURS):
            for i, time in enumerate(SCORE_TIMES):
                valid = np.isfinite(times[:, j, i])
                actual = observed[valid, j, i]
                timing_records.append((behaviour, time, valid.sum(), actual.mean(),
                                       np.mean((times[valid, j, i] - actual) ** 2),
                                       np.mean((reference[valid, j, i] - actual) ** 2)))
    return probabilities, risk, timing_records


def representative_design():
    stimuli = np.repeat(STIMULI, len(GRID_DISTANCES))
    distances = np.tile(GRID_DISTANCES, len(STIMULI))
    x = design(stimuli[:, None], np.asarray(CONTEXTS), distances[:, None], REPRESENTATIVE_N)
    return stimuli, distances, x


def standardized_timing(beta, fits, grid, context, shift=LATENCY_SHIFT):
    conditional_weight = (response_probs(beta, grid) @ STATES) * context[None, :, None]
    group_event = conditional_weight.sum(axis=1)
    medians = np.full_like(group_event, np.nan)
    for j, fit in enumerate(fits):
        if fit is None:
            continue
        weights = conditional_weight[:, :, j]
        x = grid
        low = np.zeros(len(x))
        high = np.full(len(x), HORIZON)
        np.testing.assert_allclose(time_cdf(fit, x, 0, shift), 0.0, atol=1e-12)
        np.testing.assert_allclose(time_cdf(fit, x, HORIZON, shift), 1.0, atol=1e-10)
        for _ in range(32):
            mid = (low + high) / 2
            cdf = time_cdf(fit, x, mid[:, None], shift)
            below = (weights * cdf).sum(axis=1) < group_event[:, j] / 2
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        medians[:, j] = (low + high) / 2
    return group_event, medians


def intensity_summary(playback, flags, onset):
    records = []
    for j, (behaviour, field) in enumerate(zip(BEHAVIOURS, ("HeadsUpDuration", "VocalNcalls", "ApproachDuration"))):
        for typ in STIMULI:
            use = (playback.CallType == typ).to_numpy() & flags[:, j] & playback[field].notna().to_numpy()
            observed = playback.loc[use, field]
            statistic = np.nan
            if len(observed) >= 5 and observed.nunique() >= 2 and np.unique(onset[use, j]).size >= 2:
                statistic = spearmanr(onset[use, j], observed).statistic
            records.append((behaviour, typ, field, "count" if field == "VocalNcalls" else "s",
                            len(observed), observed.quantile(0.25), observed.median(),
                            observed.quantile(0.75), statistic))
    return pd.DataFrame(records, columns=[
        "behaviour", "stimulus", "intensity_field", "unit", "events", "q25", "median", "q75", "spearman_with_latency",
    ])


def analyze_playbacks(playback, results):
    rng = np.random.default_rng(SEED + 3)
    x = design(playback.CallType, playback.Context, playback.Distance, playback.Npenguins)
    flags = event_flags(playback)
    onset = latency_values(playback)
    y = state_index(flags)
    groups = playback.expID.to_numpy()
    tests, raw_combos = testing_rates(playback, y, rng)
    penalty, loss_path = select_penalty(x, y, groups, 5, True)
    beta = response_fit(x, y, penalty)
    timing_fits = fit_timing(x, flags, onset)
    cv_rows = []
    oof, score, time_cv = cross_validate(x, y, groups, flags, onset, True, timing=True)
    cv_rows.append({"validation": "expID_nested", "dependence": "joint", **score})
    _, score, _ = cross_validate(x, y, groups, flags, onset, False)
    cv_rows.append({"validation": "expID_nested", "dependence": "independent", **score})
    for column in ("pair_track", "Date"):
        _, score, _ = cross_validate(x, y, playback[column].to_numpy(), flags, onset, True, penalty=penalty)
        cv_rows.append({"validation": column + "_fixed_penalty", "dependence": "joint", **score})
    oof_table = pd.DataFrame({
        "expID": groups, "type": playback.Type, "stimulus": playback.CallType, "state": y,
        **{f"p_{name}": oof[:, i] for i, name in enumerate(STATE_NAMES)},
    })

    stimuli, distances, grid = representative_design()
    context = context_weights(playback)
    point_p, point_t = standardized_timing(beta, timing_fits, grid, context)
    reference = playback.loc[playback.Distance.between(*SHARED_RANGE)]
    compare = np.stack([design(k, reference.Context, reference.Distance, reference.Npenguins) for k in STIMULI])
    standardized = response_probs(beta, compare).mean(axis=1)
    exp_ids = np.unique(groups)
    pair_rows = [np.flatnonzero(groups == exp_id) for exp_id in exp_ids]
    boot_beta, samples_std, samples_p, samples_t = [], [], [], []
    for _ in range(N_BOOT):
        sampled_rows = np.concatenate([pair_rows[i] for i in rng.integers(len(pair_rows), size=len(pair_rows))])
        coef = response_fit(x[sampled_rows], y[sampled_rows], penalty)
        fit = fit_timing(x[sampled_rows], flags[sampled_rows], onset[sampled_rows])
        event, medians = standardized_timing(coef, fit, grid, context)
        boot_beta.append(coef)
        samples_std.append(response_probs(coef, compare).mean(axis=1))
        samples_p.append(event)
        samples_t.append(medians)
    assert np.isfinite(samples_t).all()
    samples_std = np.asarray(samples_std)
    std_low, std_high = np.quantile(samples_std, (0.025, 0.975), axis=0)
    d_low, d_high = np.quantile(samples_std - samples_std[:, :1], (0.025, 0.975), axis=0)
    standardized_rows = [
        (typ, state, standardized[i, j], std_low[i, j], std_high[i, j],
         standardized[i, j] - standardized[0, j], d_low[i, j], d_high[i, j])
        for i, typ in enumerate(STIMULI) for j, state in enumerate(STATE_NAMES)
    ]
    standard_table = pd.DataFrame(standardized_rows, columns=[
        "stimulus", "state", "probability", "lower", "upper", "difference_from_control",
        "difference_lower", "difference_upper",
    ])
    p_bounds = np.quantile(samples_p, (0.025, 0.975), axis=0)
    t_bounds = np.quantile(samples_t, (0.025, 0.975), axis=0)
    timing_rows = []
    for i, (stimulus, distance) in enumerate(zip(stimuli, distances)):
        matching = (playback.CallType.eq(stimulus) & playback.Distance.eq(distance)).to_numpy()
        for j, behaviour in enumerate(BEHAVIOURS):
            n = matching.sum()
            events = np.count_nonzero(flags[matching, j])
            status = "Estimated" if events else "No response" if n else "No trials"
            timing_rows.append((stimulus, distance, REPRESENTATIVE_N, behaviour, n, events, status,
                                point_p[i, j], *p_bounds[:, i, j],
                                point_t[i, j] if events else np.nan,
                                *(t_bounds[:, i, j] if events else (np.nan, np.nan)),
                                point_t[i, j]))
    latency = pd.DataFrame(timing_rows, columns=[
        "stimulus", "distance_m", "group_size", "behaviour", "trials_at_distance", "events_at_distance",
        "status", "p_window", "p_lower", "p_upper", "median_s", "median_lower_s", "median_upper_s",
        "unmasked_model_projection_s",
    ])

    sensitivity = []
    alt_flags = event_flags(playback, time_priority=True)
    alt_beta = response_fit(x, state_index(alt_flags), penalty)
    experiments = [(f"shift_{shift:g}", beta, flags, shift) for shift in (0.25, 1.0)]
    experiments.append(("time_priority", alt_beta, alt_flags, LATENCY_SHIFT))
    for mode, coef, definition, shift in experiments:
        fits = fit_timing(x, definition, onset, shift)
        event, medians = standardized_timing(coef, fits, grid, context, shift)
        for i, (stimulus, distance) in enumerate(zip(stimuli, distances)):
            for j, behaviour in enumerate(BEHAVIOURS):
                sensitivity.append((mode, stimulus, distance, behaviour, event[i, j],
                                    event[i, j] - point_p[i, j], medians[i, j], medians[i, j] - point_t[i, j]))
    intensity = intensity_summary(playback, flags, onset)

    grid_rows = []
    for stimulus in STIMULI:
        for ctx in CONTEXTS:
            for d in GRID_DISTANCES:
                for n in (2, 8, 20):
                    p = response_probs(beta, design(stimulus, ctx, d, n))
                    exact = playback[(playback.CallType == stimulus) & (playback.Context == ctx) & (playback.Distance == d)]
                    grid_rows.append((stimulus, ctx, d, n, len(exact), 1 - p[0], *(p @ STATES)))
    model_grid = pd.DataFrame(grid_rows, columns=[
        "stimulus", "context", "distance_m", "group_size", "exact_distance_rows",
        "any_response", *BEHAVIOURS,
    ])
    raw_rates = (playback.assign(any_response=y > 0)
                 .groupby(["Type", "CallType", "Context"]).any_response
                 .agg(trials="size", responses="sum", response_rate="mean").reset_index())
    names = [name + ":" + b for name in FEATURE_NAMES for b in BEHAVIOURS] + [
        "HeadsUp:Vocal", "HeadsUp:Approach", "Vocal:Approach", "HeadsUp:Vocal:Approach"
    ]
    parameters = pd.DataFrame({"coefficient": names, "value": beta, "penalty": penalty})
    latency_parameters = pd.DataFrame(
        np.stack(timing_fits).T,
        columns=BEHAVIOURS,
    ).assign(parameter=(*FEATURE_NAMES, "log_sigma"))
    assert all(np.isfinite(np.exp(fit[-1])) for fit in timing_fits)
    for table, name in (
        (tests, "q3_paired_tests"), (raw_combos, "q3_combination_counts"),
        (raw_rates, "q3_raw_rates"), (pd.DataFrame(cv_rows), "q3_response_validation"),
        (oof_table, "q3_held_out_probabilities"), (standard_table, "q3_adjusted_combinations"),
        (model_grid, "q3_context_distance_group"), (latency, "q3_latency"), (intensity, "q3_intensity"),
        (parameters, "q3_response_parameters"), (latency_parameters, "q3_latency_parameters"),
        (pd.DataFrame({"penalty": LAMBDAS, "cv_log_loss": loss_path}), "q3_penalty_path"),
        (pd.DataFrame(time_cv, columns=[
            "behaviour", "time_s", "scored_rows", "empirical_event_rate", "brier", "baseline_brier",
        ]), "q3_timing_validation"),
        (pd.DataFrame(sensitivity, columns=[
            "sensitivity", "stimulus", "distance_m", "behaviour", "p_window", "p_difference",
            "unmasked_median_s", "median_difference_s",
        ]), "q3_timing_sensitivity"),
    ):
        write_table(table, results, name)
    return {
        "raw": playback, "tests": tests, "raw_combos": raw_combos, "cv": pd.DataFrame(cv_rows),
        "oof": oof_table, "parameter": beta, "boot_parameter": np.asarray(boot_beta),
        "standardized": standard_table, "context": context, "latency": latency,
        "model_grid": model_grid, "penalty": penalty,
    }
