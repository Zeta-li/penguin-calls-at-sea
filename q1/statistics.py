"""Q1 call-level summaries and rank comparisons under recording-date blocks.

The omnibus null reassigns labels within dates, keeping the global ranks and
all date-by-class counts fixed. Pairwise exact tests condition on those same
counts but compute ranks locally, using only days containing both classes.
Both nulls require label exchangeability within day; bird IDs are unavailable.
"""

from itertools import combinations
from math import comb, prod
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, mannwhitneyu, rankdata, tiecorrect


CALLS = ("flat", "mod", "tv")
COMMON = ("Duration", "F01", "Fpeak", "Q25", "Q50", "Q75")
ALL_FEATURES = (*COMMON, "F02", "PulseRate")
N_PERMUTATIONS = 3999
PERMUTATION_SEED = 20261127


def holm(p_values):
    """Return step-down familywise adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def describe_calls(data):
    rows = []
    for call in CALLS:
        part = data.loc[data.CallType.eq(call)]
        for feature in ALL_FEATURES:
            values = part[feature].dropna()
            n_missing = len(part) - len(values)
            structurally_absent = (feature == "F02" and call != "tv") or (
                feature == "PulseRate" and call == "flat"
            )
            if structurally_absent and len(values):
                raise ValueError(f"Unexpected measured {feature} in {call}; recheck applicability")
            rows.append({
                "feature": feature, "type": call, "n_total": len(part),
                "n_measured": len(values), "n_not_applicable": n_missing if structurally_absent else 0,
                "n_missing_unknown": 0 if structurally_absent else n_missing,
                "mean": values.mean(), "sd": values.std(),
                "min": values.min(), "q25": values.quantile(0.25),
                "median": values.median(), "q75": values.quantile(0.75), "max": values.max(),
            })
    return pd.DataFrame(rows)


def omnibus_tests(data):
    ranks = rankdata(data.loc[:, COMMON].to_numpy(), axis=0)
    classes = pd.Categorical(data.CallType, categories=CALLS).codes
    dates = pd.to_datetime(data.Date, format="%d/%m/%Y")
    day_indices = list(data.groupby(dates, sort=True).indices.values())
    n = len(data)
    group_sizes = np.bincount(classes)
    ties = np.array([tiecorrect(column) for column in ranks.T])
    if (ties == 0).any():
        raise ValueError("Constant common feature: omnibus rank test undefined")
    centered = ranks - (n + 1) / 2

    def statistic(labels):
        totals = np.stack([centered[labels == k].sum(axis=0) for k in range(len(CALLS))])
        return 12 * (totals**2 / group_sizes[:, None]).sum(axis=0) / (n * (n + 1) * ties)

    h = statistic(classes)
    rng = np.random.default_rng(PERMUTATION_SEED)
    shuffled = classes.copy()
    extreme = np.zeros(len(COMMON), dtype=int)
    for _ in range(N_PERMUTATIONS):
        for indices in day_indices:
            shuffled[indices] = rng.permutation(classes[indices])
        extreme += statistic(shuffled) >= h - 1e-10
    p = (1 + extreme) / (1 + N_PERMUTATIONS)
    return pd.DataFrame({
        "feature": COMMON, "H": h, "p_asymptotic": chi2.sf(h, len(CALLS) - 1),
        "p_date_permutation": p, "p_date_holm": holm(p), "permutations": N_PERMUTATIONS,
        "seed": PERMUTATION_SEED,
    })


def day_rank_law(values, first):
    """Exact integer counts for 2U - n1*n2 in one day, preserving tied ranks."""
    n_first = int(np.count_nonzero(first))
    n = len(values)
    ranks = np.rint(2 * rankdata(values)).astype(int)
    shift = n_first * (n + 1)
    observed = int(ranks[first].sum()) - shift
    # counts[k, s] counts the subsets of k labelled positions with doubled
    # rank sum s. Equal ranks keep their original assignment multiplicities.
    max_sum = int(np.sort(ranks)[-n_first:].sum())
    counts = np.zeros((n_first + 1, max_sum + 1), dtype=object)
    counts[0, 0] = 1
    for i, rank in enumerate(ranks):
        for k in range(min(i + 1, n_first), 0, -1):
            counts[k, rank:] += counts[k - 1, :-rank]
    support = np.flatnonzero(counts[-1])
    lo, hi = int(support[0]), int(support[-1])
    distribution = counts[-1, lo:hi + 1]
    if distribution.sum() != comb(n, n_first):
        raise ArithmeticError("Assignment count mismatch in exact rank test")
    return observed, lo - shift, distribution


def stratified_pair(data, feature, first_call, second_call):
    """Sum day-specific rank effects, weighting equally each comparable pair."""
    subset = data.loc[data.CallType.isin([first_call, second_call]), ["Date", "CallType", feature]]
    distribution = np.array([1], dtype=object)
    observed = offset = pairs = mixed_dates = n1 = n2 = 0
    counts = []
    for _, day in subset.dropna().groupby("Date"):
        first = day.CallType.eq(first_call).to_numpy()
        n_first, n_second = int(first.sum()), int((~first).sum())
        if not n_first or not n_second:
            continue
        score, lower, law = day_rank_law(day[feature].to_numpy(), first)
        observed += score
        offset += lower
        distribution = np.convolve(distribution, law)
        counts.append(comb(len(day), n_first))
        mixed_dates += 1
        n1 += n_first
        n2 += n_second
        pairs += n_first * n_second
    if pairs == 0:
        raise ValueError(f"No mixed day for {feature}: {first_call} / {second_call}")
    assignments = prod(counts)
    extremes = np.abs(np.arange(len(distribution)) + offset) >= abs(observed)
    return {
        "mixed_dates": mixed_dates, "n_first": n1, "n_second": n2, "comparable_pairs": pairs,
        "statistic": observed, "effect": observed / pairs, "permutations": assignments,
        "p_exact": int(distribution[extremes].sum()) / assignments,
    }


def pairwise_tests(data):
    pooled_rows, within_rows = [], []
    for feature in COMMON:
        for first, second in combinations(CALLS, 2):
            key = {"feature": feature, "first": first, "second": second}
            a = data.loc[data.CallType.eq(first), feature]
            b = data.loc[data.CallType.eq(second), feature]
            u, p = mannwhitneyu(a, b, method="asymptotic", alternative="two-sided")
            pooled_rows.append(dict(key, U=u, p=p, n_first=len(a), n_second=len(b),
                                    cliffs_delta=2 * u / (len(a) * len(b)) - 1))
            within_rows.append(dict(key, **stratified_pair(data, feature, first, second)))
    pooled, within = pd.DataFrame(pooled_rows), pd.DataFrame(within_rows)
    pooled["p_holm"] = holm(pooled.p)
    within["p_holm"] = holm(within.p_exact)
    return pooled, within


FIGURE_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                "font.size": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                "svg.fonttype": "none", "svg.hashsalt": "q1-acoustic-distributions"}
COLORS = ("#307482", "#996332", "#756894")


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def plot_peak_frequency(data, out):
    """One dot per recording. The box spans Q1-Q3; line = median, whiskers = 1.5 IQR."""
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.2), layout="constrained")
        groups = [data.loc[data.CallType.eq(call), "Fpeak"].to_numpy() for call in CALLS]
        rng = np.random.default_rng(131)
        # Jitter only the category coordinate, never the measured frequency.
        for pos, (values, color) in enumerate(zip(groups, COLORS), 1):
            ax.scatter(pos + rng.uniform(-0.12, 0.12, size=len(values)), values, s=9,
                       color=color, alpha=0.6, linewidth=0, zorder=2)
        ax.boxplot(groups, widths=0.42, showfliers=False, zorder=3,
                   medianprops={"color": "#15252b", "linewidth": 1.6},
                   boxprops={"color": "#38464c"}, whiskerprops={"color": "#38464c"},
                   capprops={"color": "#38464c"})
        ax.set_xticks(range(1, 4), [f"{call} (n={len(g)})" for call, g in zip(CALLS, groups)])
        ax.set_ylabel("Peak frequency (Hz)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e9e9", linewidth=0.6)
        save_figure(fig, out, "peak_frequency")


def plot_feature_summary(summary, out):
    """Compare medians and IQRs on each feature's observed min–max scale."""
    order = ("Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate")
    labels = ("Duration", "First fundamental", "Second fundamental", "Peak frequency",
              "Spectral quartile 1", "Spectral quartile 2", "Spectral quartile 3", "Pulse rate")
    lower = summary.groupby("feature")["min"].min()
    upper = summary.groupby("feature")["max"].max()
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
        for call, color, marker, shift in zip(CALLS, COLORS, ("o", "s", "^"), (-0.20, 0, 0.20)):
            rows = summary[summary.type.eq(call)].set_index("feature").loc[list(order)]
            rows = rows[rows.n_measured.gt(0)]
            # Per-feature scaling is solely for plotting disparate physical units;
            # rank tests and the separate peak-frequency panel retain native units.
            spread = (upper - lower).loc[rows.index].replace(0, 1)
            q = rows[["q25", "median", "q75"]].sub(lower.loc[rows.index], axis=0).div(spread, axis=0)
            y = np.array([order.index(f) for f in rows.index]) + shift
            ax.hlines(y, q.q25, q.q75, color=color, linewidth=1.8)
            ax.scatter(q["median"], y, color=color, marker=marker, s=22, zorder=3, label=call)
        ax.set_yticks(range(len(order)), labels)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Position within each feature's observed min–max range")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e9e9", linewidth=0.6)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        save_figure(fig, out, "feature_median_iqr")


def run(data: pd.DataFrame, out: Path):
    """Write group summaries, the six omnibus and 18 pairwise tests, and a figure."""
    if data.loc[:, COMMON].isna().any().any():
        raise ValueError("The six common features must be complete for this protocol")
    if data.Date.isna().any() or set(data.CallType.unique()) != set(CALLS):
        raise ValueError("Expected three named call types and a known date for each recording")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pooled, within = pairwise_tests(data)
    summary = describe_calls(data)
    for name, table in {
        "feature_summary": summary, "omnibus": omnibus_tests(data),
        "pairwise_pooled": pooled, "pairwise_within_date": within,
        "date_type_counts": pd.crosstab(data.Date, data.CallType).reset_index(),
    }.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
    plot_peak_frequency(data, out)
    plot_feature_summary(summary, out)
