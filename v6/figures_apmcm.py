"""Publication figures backed by the emitted statistical results."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.ticker import MaxNLocator, PercentFormatter, ScalarFormatter
from PIL import Image

from penguin_data import BEHAVIOURS, CALLS, CONTEXTS, DISTANCES, FEATURES, METRICS, SEED
from utils.export_figure import export_figure
from utils.plot_style import apply_publication_style, audit_design, audit_layout

WIDTH = 6.55
LABELS = {"flat": "Flat", "mod": "Modulated", "tv": "Two-voice", "control": "Wave control"}
COLORS = {"flat": "#0072B2", "mod": "#D55E00", "tv": "#008B69", "control": "#686E77"}
MARKS = {"flat": "o", "mod": "s", "tv": "^", "control": "D"}
CONTEXT_LABELS = ["Commuting", "Searching", "After feeding"]
BEHAVIOUR_LABELS = ("Head raise", "Vocalise", "Approach")


def save(fig, folder, name):
    problems = audit_layout(fig) + audit_design(fig)
    if problems:
        raise ValueError(f"{name}: {'; '.join(problems)}")
    stem = folder / name
    export_figure(fig, str(stem), formats=("svg", "png"), dpi=300, tight=False)
    qa = folder / "_qa"
    qa.mkdir(exist_ok=True)
    with Image.open(stem.with_suffix(".png")) as original:
        original.convert("L").save(qa / f"{name}_grayscale.png", dpi=(300, 300))
    plt.close(fig)


def panel(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontweight="bold", fontsize=8.2)


def bounded_probability(ax, axis="y"):
    if axis == "y":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))


def heatmap(ax, matrix, xlabels, ylabels, cmap, vmin, vmax, value_format):
    edges_x = np.arange(len(xlabels) + 1) - 0.5
    edges_y = np.arange(len(ylabels) + 1) - 0.5
    image = ax.pcolormesh(edges_x, edges_y, matrix, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_ylim(edges_y[-1], edges_y[0])
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    for row, col in np.ndindex(matrix.shape):
        value = matrix[row, col]
        r, g, b, _ = image.cmap(image.norm(value))
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(col, row, value_format(value), ha="center", va="center",
                fontsize=7.3, color="white" if brightness < 0.49 else "#17202B")
    return image


def q1_figures(q1, folder):
    rng = np.random.default_rng(SEED + 31)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.20), layout="constrained")
    feature_labels = ("Duration", "First fundamental", "Peak frequency", "Q25", "Q50", "Q75")
    for j, (axis, feature, heading) in enumerate(zip(axes.flat, FEATURES, feature_labels)):
        data = [q1["data"].loc[q1["data"].CallType.eq(call), feature].to_numpy() for call in CALLS]
        box = axis.boxplot(data, widths=0.43, patch_artist=True, showfliers=False,
                           medianprops={"color": "#263342", "linewidth": 1.1})
        for i, (patch, values, call) in enumerate(zip(box["boxes"], data, CALLS), 1):
            patch.set_facecolor(COLORS[call])
            patch.set_alpha(0.16)
            axis.scatter(i + rng.uniform(-0.19, 0.19, len(values)), values, s=7,
                         color=COLORS[call], marker=MARKS[call], alpha=0.56, linewidth=0)
        axis.set_xticks((1, 2, 3), ["Flat", "Mod.", "Two-voice"])
        axis.set_ylabel("Seconds" if feature == "Duration" else "Frequency (Hz)")
        panel(axis, chr(97 + j), heading)
    save(fig, folder, "raw_q1_acoustic")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.65), layout="constrained")
    for j, (axis, feature, title) in enumerate(zip(
        axes, ("F02", "PulseRate"), ("Second fundamental", "Pulse rate")
    )):
        tick_labels = []
        for i, call in enumerate(CALLS):
            group = q1["data"].loc[q1["data"].CallType.eq(call), feature].dropna().to_numpy()
            short = ("Flat", "Mod.", "Two-voice")[i]
            tick_labels.append(f"{short}\n(n={len(group)})")
            if not len(group):
                continue
            axis.boxplot(group, positions=[i], widths=0.43, showfliers=False,
                         medianprops={"color": "#263342", "linewidth": 1.0})
            axis.scatter(i + rng.uniform(-0.15, 0.15, len(group)), group, s=13,
                         color=COLORS[call], marker=MARKS[call], alpha=0.64, linewidth=0)
        axis.set_xticks(range(3), tick_labels)
        axis.set_xlim(-0.45, 2.45)
        axis.set_ylabel("Frequency (Hz)" if j == 0 else "Pulse rate (Hz)")
        panel(axis, chr(97 + j), title)
    save(fig, folder, "raw_q1_voice_measures")

    fig, ax = plt.subplots(figsize=(4.25, 3.05), layout="constrained")
    predictions = q1["predictions"]
    cells = pd.crosstab(predictions.actual, predictions.predicted).reindex(index=CALLS, columns=CALLS, fill_value=0)
    heatmap(ax, cells.to_numpy(), [LABELS[k] for k in CALLS], [LABELS[k] for k in CALLS],
            "Blues", 0, int(cells.to_numpy().max()), lambda v: str(int(v)))
    ax.set(xlabel="Prediction on held-out date", ylabel="Annotated type")
    save(fig, folder, "process_q1_confusion")

    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.75), width_ratios=(1.03, 1.25), layout="constrained")
    performance = q1["performance"].iloc[:3]
    y = np.arange(len(performance))
    left.hlines(y, performance.lower, performance.upper, color=COLORS["flat"], linewidth=1.6)
    left.scatter(performance.estimate, y, color=COLORS["flat"], zorder=3, label="Classifier")
    left.scatter(performance.training_prior_baseline, y, color=COLORS["control"],
                 marker="x", zorder=3, label="Training prior")
    left.set_yticks(y, ["Accuracy", "Balanced accuracy", "Macro F1"])
    left.invert_yaxis()
    left.set_xlim(0, 0.55)
    fig.legend(*left.get_legend_handles_labels(), loc="outside upper center", ncols=2,
               frameon=False, fontsize=7)
    bounded_probability(left, "x")
    panel(left, "a", "Date-held-out scores")
    ablation = q1["ablation"].sort_values("change_log_loss")
    y = np.arange(len(ablation))
    right.axvline(0, color="#919BA6", linewidth=0.8)
    right.hlines(y, 0, ablation.change_log_loss, color="#7D8E9F", linewidth=1.2)
    right.scatter(ablation.change_log_loss, y, color="#233C56", zorder=3)
    right.set_yticks(y, ablation.dropped)
    right.set_xlabel("Held-out log-loss change")
    panel(right, "b", "Remove one feature group")
    save(fig, folder, "result_q1_predictors")


def log_distance(axis):
    axis.set_xscale("log", base=2)
    axis.set_xticks(DISTANCES, [f"{value:g}" for value in DISTANCES])
    axis.xaxis.set_minor_formatter(ScalarFormatter())


def q2_figures(q2, folder):
    measured = q2["measurements"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.25), layout="constrained")
    counts = measured[measured.representation.eq(METRICS[0])].pivot(
        index="call", columns="distance_m", values="observed_entries"
    ).reindex(CALLS)
    image = heatmap(ax, counts.to_numpy(), [f"{d:g}" for d in DISTANCES], [LABELS[k] for k in CALLS],
                    "Blues", 180, 300, lambda v: f"{v:g}")
    scale = fig.colorbar(image, ax=ax, label="Valid comparison entries", fraction=0.03, pad=0.025)
    scale.solids.set_rasterized(False)
    ax.set_xlabel("Propagation distance (m)")
    save(fig, folder, "raw_q2_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.70), sharey=True, layout="constrained")
    palettes = ("#176589", "#A45627", "#4C5964")
    styles = ("-", "--", ":")
    for j, (axis, call) in enumerate(zip(axes, CALLS)):
        for metric, color, style, marker in zip(METRICS, palettes, styles, ("o", "s", "^")):
            table = measured[(measured["call"] == call) & (measured.representation == metric)]
            axis.plot(table.distance_m, table.fitted, color=color, linestyle=style, label=metric)
            axis.scatter(table.distance_m, table.block_mean, facecolor="white", edgecolor=color,
                         linewidth=0.9, s=17, marker=marker, zorder=3)
        log_distance(axis)
        axis.set_ylim(0.08, 1.0)
        panel(axis, chr(97 + j), LABELS[call])
    axes[0].set_ylabel("Correlation with 1 m")
    fig.supxlabel("Propagation distance (m)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, frameon=False, loc="outside upper center", fontsize=7.5)
    save(fig, folder, "process_q2_profiles")

    fig, ax = plt.subplots(figsize=(WIDTH, 2.83), layout="constrained")
    ranges = q2["ranges"]
    for i, call in enumerate(CALLS):
        table = ranges[ranges["call"].eq(call)]
        places = np.arange(3) + (i - 1) * 0.23
        ax.hlines(places, table.lower_m, table.upper_m, color=COLORS[call], linewidth=1.4)
        ax.scatter(table.distance_m, places, color=COLORS[call], marker=MARKS[call], label=LABELS[call], zorder=3)
    ax.set_yticks(np.arange(3), ["Potential  0.50", "Recognition  0.60", "Reliable  0.70"])
    ax.set_ylim(2.55, -0.5)
    ax.set_xlim(0, 185)
    ax.set_xlabel("Effective acoustic distance (m)")
    fig.legend(*ax.get_legend_handles_labels(), ncols=3, loc="outside upper center", frameon=False)
    save(fig, folder, "result_q2_ranges")


def q3_figures(q3, folder):
    table = q3["raw_combos"]
    states = list(table.columns[1:])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.73), layout="constrained")
    x = np.arange(8)
    for index, (name, color, hatch) in enumerate((
        ("Natural calls", "#176589", None), ("Wave control", "#B1B9C3", "///")
    )):
        counts = table.iloc[index, 1:].to_numpy(dtype=float)
        values = counts / counts.sum()
        ax.bar(x + (index - 0.5) * 0.34, values, width=0.32,
               color=color, edgecolor="#324457", linewidth=0.45, hatch=hatch, label=name)
    ax.set_xticks(x, states)
    ax.set(xlabel="State (head raise, vocalisation, approach)", ylabel="Observed proportions", ylim=(0, 1))
    bounded_probability(ax)
    ax.legend(frameon=False, loc="upper right")
    save(fig, folder, "raw_q3_combinations")

    fig, ax = plt.subplots(figsize=(4.25, 3.00), layout="constrained")
    ordered = q3["oof"].sort_values("p_000", ascending=False)
    bins = [ordered.iloc[idx] for idx in np.array_split(np.arange(len(ordered)), 5)]
    p_emp = np.array([(rows.state > 0).mean() for rows in bins])
    p_fit = np.array([(1 - rows.p_000).mean() for rows in bins])
    ax.plot([0, 1], [0, 1], color="#9CA9B5", linestyle="--", label="Equal proportions")
    ax.scatter(p_fit, p_emp, s=36, color="#176589", label="Held-out bins", zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted response", ylabel="Observed response")
    bounded_probability(ax)
    bounded_probability(ax, "x")
    ax.legend(frameon=False, loc="lower right")
    save(fig, folder, "process_q3_calibration")

    fig, ax = plt.subplots(figsize=(WIDTH, 2.48), layout="constrained")
    table = q3["standardized"].pivot(index="stimulus", columns="state", values="difference_from_control").reindex(CALLS)
    deltas = 100 * table.to_numpy()
    span = max(10.0, np.ceil(np.max(np.abs(deltas)) / 10) * 10)
    image = heatmap(ax, deltas, states, [LABELS[k] for k in CALLS], "RdBu", -span, span, lambda v: f"{v:+.1f}")
    scale = fig.colorbar(image, ax=ax, label="Change (percentage points)", fraction=0.035, pad=0.035)
    scale.solids.set_rasterized(False)
    ax.set_xlabel("State (head raise, vocalisation, approach)")
    save(fig, folder, "result_q3_combinations")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.02), sharey=True, layout="constrained")
    latency = q3["latency"]
    for j, (axis, behaviour) in enumerate(zip(axes, BEHAVIOURS)):
        table = latency[latency.behaviour.eq(behaviour)]
        for i, call in enumerate(CALLS):
            rows = table[table.stimulus.eq(call)]
            located = rows.distance_m.to_numpy() + (i - 1) * 4
            observed = rows.status.eq("Estimated").to_numpy()
            axis.vlines(located[observed], rows.loc[observed, "median_lower_s"],
                       rows.loc[observed, "median_upper_s"], color=COLORS[call], linewidth=1.1)
            axis.scatter(located[observed], rows.loc[observed, "median_s"],
                         color=COLORS[call], marker=MARKS[call], s=23, label=LABELS[call], zorder=3)
            axis.scatter(located[~observed], np.full((~observed).sum(), -3), color=COLORS[call],
                         marker="x", s=15, linewidth=0.8, clip_on=False)
        panel(axis, chr(97 + j), BEHAVIOUR_LABELS[j])
        axis.set_xticks([50, 100, 150])
        axis.set_xlim(39, 161)
        axis.axhline(0, color="#9CA9B5", linewidth=0.8)
        upper = max(15, latency.median_upper_s.max() * 1.08)
        axis.set_ylim(-5, upper)
        ticks = MaxNLocator(nbins=5).tick_values(0, upper)
        axis.set_yticks(ticks[(ticks >= 0) & (ticks <= upper)])
    axes[0].set_ylabel("Positive-event median (s)")
    fig.supxlabel("Playback distance (m)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#686E77", linestyle="none", marker="x", markersize=4))
    labels.append("No observed response")
    fig.legend(handles, labels, ncols=4, loc="outside upper center", frameon=False, fontsize=7)
    save(fig, folder, "result_q3_latencies")


def q4_figures(q4, folder):
    fig, ax = plt.subplots(figsize=(5.25, 3.05), layout="constrained")
    for ctx, long, color, style in zip(CONTEXTS, CONTEXT_LABELS, ("#176589", "#A45627", "#4C5964"), ("-", "--", ":")):
        values = np.sort(q4["raw_groups"].loc[q4["raw_groups"].Context.eq(ctx), "Npenguins"])
        proportions = np.r_[np.arange(len(values) + 1) / len(values), 1.0]
        ax.step(np.r_[0, values, 40], proportions, where="post",
                linewidth=1.6, color=color, linestyle=style, label=f"{long}  (n={len(values)})")
    ax.axvline(8, color="#9CA9B5", linewidth=0.8)
    ax.set(xlim=(0, 40), ylim=(0, 1.02), xlabel="Observed group size (penguins)", ylabel="Cumulative proportion")
    ax.legend(frameon=False, loc="lower right")
    bounded_probability(ax)
    save(fig, folder, "raw_q4_groups")

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.65), sharey=True, layout="constrained")
    curves = q4["curves"]
    display = ["Threshold 0.60", "Threshold 0.70", "Fidelity loss 20%", "Combined stress"]
    styles = ["-", "--", "-.", ":"]
    shades = ["#176589", "#3F4856", "#A45627", "#7C7D83"]
    for j, (axis, call) in enumerate(zip(axes, CALLS)):
        for name, style, color in zip(display, styles, shades):
            curve = curves[(curves.context == "pooled") & (curves["call"] == call) & (curves.scenario == name)]
            axis.plot(curve.distance_m, curve.retained_excess, label=name, linestyle=style, color=color)
        axis.set(xlim=(20, 150), ylim=(0, 1.02))
        panel(axis, chr(97 + j), LABELS[call])
        bounded_probability(axis)
    axes[0].set_ylabel("Retained stimulus contrast")
    fig.supxlabel("Distance (m)", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=4, loc="outside upper center", frameon=False, fontsize=7)
    save(fig, folder, "process_q4_retention")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.03), sharey=True,
                             width_ratios=(1.15, 0.85), layout="constrained")
    scenarios = q4["scenarios"]
    display = ["Baseline", "Threshold 0.60", "Threshold 0.70", "Fidelity loss 20%"]
    for j, call in enumerate(CALLS):
        table = scenarios[(scenarios["call"] == call) & (scenarios.context == "pooled")].set_index("scenario").loc[display]
        places = np.arange(len(display)) + (j - 1) * 0.22
        axes[0].hlines(places, table.radius_lower_m, table.radius_upper_m, color=COLORS[call], linewidth=1.2)
        closed = table.radius_status.ne("right_boundary").to_numpy()
        axes[0].scatter(table.functional_radius_m[closed], places[closed], color=COLORS[call],
                        marker=MARKS[call], label=LABELS[call], s=23, zorder=3)
        axes[0].scatter(table.functional_radius_m[~closed], places[~closed], edgecolor=COLORS[call],
                        facecolor="white", marker=">", s=24, zorder=4)
        axes[1].hlines(places, table.p_lower, table.p_upper, color=COLORS[call], linewidth=1.15)
        axes[1].scatter(table.any_at_100m, places, color=COLORS[call], marker=MARKS[call], s=23)
    axes[0].set_yticks(np.arange(len(display)), ["Baseline", "Threshold 0.60", "Threshold 0.70", "20% fidelity loss"])
    axes[0].set(xlim=(20, 155), ylim=(3.45, -0.45), xlabel="Functional radius (m)")
    axes[0].axvline(150, color="#9CA9B5", linewidth=0.8, linestyle=":")
    axes[1].set(xlim=(0, 0.85), xlabel="Probability at 100 m")
    bounded_probability(axes[1], "x")
    panel(axes[0], "a", "Functional radius")
    panel(axes[1], "b", "Any group response")
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=3, loc="outside upper center", frameon=False)
    save(fig, folder, "result_q4_noise")


def box(ax, x, y, w, h, text, kind="model"):
    dark = "#33475B"
    if kind == "data":
        delta = w * 0.07
        artist = Polygon(
            [(x + delta, y), (x + w, y), (x + w - delta, y + h), (x, y + h)],
            closed=True, facecolor="#EFF3F6", edgecolor=dark, linewidth=0.85,
        )
    else:
        rounding = 0.018 if kind == "terminal" else 0.003
        artist = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.002,rounding_size={rounding}",
                                facecolor="#E4EDF3" if kind == "terminal" else "white",
                                edgecolor=dark, linewidth=0.85)
    ax.add_patch(artist)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=7.8, color="#182D41", linespacing=1.25)


def arrow(ax, start, end, style="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                linewidth=0.9, color="#485C70", connectionstyle=style, shrinkA=2, shrinkB=2))


def flow_overall(folder):
    fig, ax = plt.subplots(figsize=(WIDTH, 4.15), layout="none")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    columns = (0.025, 0.357, 0.689)
    texts = [
        ("108 acoustic recordings", "Q1  Call classification\nRidge multinomial model",
         "Hold out recording dates\nCompare six acoustic features"),
        ("Nine propagation tables", "Q2  Signal fidelity\nThree monotone curves",
         "Hold out distances\nResample position blocks"),
        ("139 paired playback groups", "Q3  Joint behaviour\nEight states + onset times",
         "Hold out paired experiments\nResample experiment pairs"),
    ]
    w = 0.282
    for x, items in zip(columns, texts):
        box(ax, x, 0.83, w, 0.11, items[0], "data")
        box(ax, x, 0.61, w, 0.15, items[1])
        box(ax, x, 0.39, w, 0.15, items[2])
        centre = x + w / 2
        arrow(ax, (centre, 0.83), (centre, 0.76))
        arrow(ax, (centre, 0.61), (centre, 0.54))
    box(ax, 0.27, 0.17, 0.46, 0.145,
        "Q4  Noise scenarios at matched distances\nGate the natural-call contrast; weight behaviours")
    box(ax, 0.025, 0.195, 0.205, 0.085, "Call interpretation", "terminal")
    arrow(ax, (columns[0] + w / 2, 0.39), (0.13, 0.28))
    for x, end_x in zip(columns[1:], (0.50, 0.68)):
        arrow(ax, (x + w / 2, 0.39), (end_x, 0.315))
    box(ax, 0.12, 0.015, 0.76, 0.085,
        "Functional radii  |  Group-response probabilities  |  Scenario bounds", "terminal")
    arrow(ax, (0.50, 0.17), (0.50, 0.10))
    save(fig, folder, "flow_overall_model")


def flow_questions(folder):
    all_steps = [
        ("Six shared\nfeatures", "Within-date ranks\n+ nested fitting", "Out-of-date\npredictions", "Ablation +\nclass errors"),
        ("Three acoustic\nrepresentations", "Block averages +\nmonotone fit", "Threshold\ninversion", "Ranges +\nblock sensitivity"),
        ("Matched call and\ncontrol responses", "Joint eight-state\nlikelihood", "Conditional onset\nmodel", "Hold-out +\npaired inference"),
        ("Fidelity +\nstate probabilities", "Noise gate +\nrelative contrast", "Context-specific\nutility", "Radii +\nresponse counts"),
    ]
    for i, steps in enumerate(all_steps, 1):
        fig, ax = plt.subplots(figsize=(WIDTH, 1.75), layout="none")
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.08, top=0.92)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        positions = (0.02, 0.27, 0.52, 0.77)
        for pos, text, kind in zip(positions, steps, ("data", "model", "model", "terminal")):
            box(ax, pos, 0.27, 0.21, 0.46, text, kind)
        for current, following in zip(positions, positions[1:]):
            arrow(ax, (current + 0.21, 0.5), (following, 0.5))
        save(fig, folder, f"flow_q{i}_model")


def render_figures(folder: Path, q1, q2, q3, q4):
    folder.mkdir(parents=True, exist_ok=True)
    apply_publication_style(language="en")
    plt.rcParams["svg.hashsalt"] = "apmcm-c-20261126"
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1795651200")
    q1_figures(q1, folder)
    q2_figures(q2, folder)
    q3_figures(q3, folder)
    q4_figures(q4, folder)
    flow_overall(folder)
    flow_questions(folder)
