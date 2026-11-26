"""Re-evaluate recognition and context-specific behaviour on the common 20–150 m domain.

Only the playback bootstrap below has experimental pair sampling units. Acoustic
curves, context weights, penalties and scenario parameters stay fixed in those draws.
"""
import argparse
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from analyze_effectiveness import DESIGN, probabilities
from models_effectiveness import BASE_WIDTH, SCENARIOS, UTILITY_THRESHOLD, ecological_values, functional_radius
from propagation_v4 import fit_primary, predict, summary_cells
from models_playback import design, response_fit
from penguin_data import CALLS, CONTEXTS, REPRESENTATIVE_N, SEED, STATE_NAMES, STATES, context_weights, event_flags, read_inputs, state_index

MECHANISMS = ("shared", "vigilance", "selective", "both")
ENDPOINTS = ("contact_net_vocal", "recruitment_approach", "context_composite")
CRITERIA = ("mean", "all")
AT_100 = np.flatnonzero(DESIGN == 100).item()
PENALTY = 0.5  # previously selected on complete-pair CV, fixed in conditional resampling
N_BOOT = 400
BOOT_SEED = SEED + 444
ANY = (STATES.any(axis=1)).astype(float)
UTILITIES = np.stack((np.tile(STATES[:, 1], (3, 1)), np.tile(STATES[:, 2], (3, 1)),
                      ecological_values()), axis=-1)


def stress_kernel(mechanism, loss):
    """Transition between BEFORE/AFTER response states (rows/columns), independent bit flips.

    Strength refers only to relative acoustic loss. Tightening an operational cutoff on
    the same recording does not alter the waves control.
    """
    s = loss / 0.20
    heads_up = .20 * s if mechanism in ("vigilance", "both") else 0.
    vocal_off = .15 * s if mechanism in ("selective", "both") else 0.
    approach_off = .40 * s if mechanism in ("selective", "both") else 0.
    one = np.column_stack((STATES[:, 0] + (1 - STATES[:, 0]) * heads_up,
                           STATES[:, 1] * (1 - vocal_off), STATES[:, 2] * (1 - approach_off)))
    transitions = np.where(STATES[None, :, :], one[:, None, :], 1 - one[:, None, :]).prod(axis=-1)
    np.testing.assert_allclose(transitions.sum(axis=1), 1, atol=1e-13)
    assert (transitions >= 0).all()
    return transitions


def gates(fidelity, tau, loss):
    baseline = expit((fidelity - .50) / BASE_WIDTH)
    current = expit(((1 - loss) * fidelity - tau) / BASE_WIDTH)
    retention = np.minimum(1, current / baseline)
    assert (retention >= 0).all() and (retention <= 1).all()
    return baseline, retention


def radius_row(curve):
    clipped, flag, later = functional_radius(DESIGN, curve, UTILITY_THRESHOLD)
    return (float(clipped), flag, later)


def radius_grid(curves):
    """Segment touching the first observed distance. 20 and 150 are clipped codes."""
    # Input [call, distance, endpoint]; reorder so distance is last.
    by_distance = curves.transpose(0, 2, 1)
    below = by_distance < UTILITY_THRESHOLD
    first_fail = np.argmax(below, axis=-1)
    left = below[:, :, 0]
    right = ~below.any(axis=-1)
    j = np.maximum(1, first_fail)[..., None]
    prev = np.take_along_axis(by_distance, j - 1, axis=2).squeeze(-1)
    cur = np.take_along_axis(by_distance, j, axis=2).squeeze(-1)
    frac = np.divide(prev - UTILITY_THRESHOLD, prev - cur,
                     out=np.zeros_like(cur), where=cur != prev)
    radius = DESIGN[j.squeeze(-1) - 1] + frac * (DESIGN[1] - DESIGN[0])
    return np.where(left, DESIGN[0], np.where(right, DESIGN[-1], radius)), left, right


def point_tables(beta, playback, fidelities, out):
    weights = context_weights(playback)
    probs = probabilities(beta)
    p0, difference = probs[0], probs[1:] - probs[0]
    radii, counts, curves, state_rows = [], [], [], []
    for criterion, fidelity in fidelities.items():
        for scenario, tau, loss in SCENARIOS:
            g0, h = gates(fidelity, tau, loss)
            for mechanism in MECHANISMS:
                kernel = stress_kernel(mechanism, loss)
                q0 = p0 @ kernel
                q = (p0[None, ...] + h[:, None, :, None] * difference) @ kernel
                assert np.min(q) >= -1e-12 and np.min(q0) >= -1e-12
                np.testing.assert_allclose(q.sum(axis=-1), 1, atol=2e-12)
                np.testing.assert_allclose(q0.sum(axis=-1), 1, atol=2e-12)
                if loss == 0:
                    np.testing.assert_allclose(q0, p0)
                if scenario == "Baseline":
                    np.testing.assert_allclose(q, probs[1:], atol=2e-12)
                score = g0[:, None, :, None] * np.einsum("czds,zse->czde", q - q0, UTILITIES)
                pooled_score = np.einsum("czde,z->cde", score, weights)
                pooled_q = np.einsum("czds,z->cds", q, weights)
                pooled_q0 = np.einsum("zds,z->ds", q0, weights)
                pooled_radius, _, _ = radius_grid(pooled_score)
                for ci, call in enumerate(CALLS):
                    common = (criterion, scenario, tau, loss, mechanism, call)
                    for zi, ctx in enumerate((*CONTEXTS, "pooled")):
                        scores = pooled_score[ci] if ctx == "pooled" else score[ci, zi]
                        natural = pooled_q[ci, AT_100] if ctx == "pooled" else q[ci, zi, AT_100]
                        waves = pooled_q0[AT_100] if ctx == "pooled" else q0[zi, AT_100]
                        for sj, state in enumerate(STATE_NAMES):
                            state_rows.append((*common, ctx, state, natural[sj], waves[sj], natural[sj]-waves[sj]))
                        for ej, endpoint in enumerate(ENDPOINTS):
                            clipped, flag, later = radius_row(scores[:, ej])
                            if ctx == "pooled":
                                np.testing.assert_allclose(clipped, pooled_radius[ci,ej], atol=1e-8, rtol=0)
                            radii.append((*common, ctx, endpoint, clipped if flag == "crossing" else np.nan,
                                          clipped, flag, later, scores[AT_100, ej]))
                        for label, utility in ((ENDPOINTS[0], STATES[:, 1]), (ENDPOINTS[1], STATES[:, 2]),
                                               ("any_flag", ANY)):
                            p = natural @ utility
                            control = waves @ utility
                            for fraction in (0., .25, .5, 1.):
                                counts.append((*common, ctx, label, p, control, p-control, fraction,
                                               p*(1+(REPRESENTATIVE_N-1)*fraction), p, REPRESENTATIVE_N*p))
                    # 1 m increments store all pooled predicted signals, controls, and score curves.
                    # Avoid writing redundant dense tables for the two one-factor ablations.
                    if mechanism in ("shared", "both"):
                        any_group = pooled_q[ci] @ ANY
                        heads = pooled_q[ci] @ STATES[:, 0]
                        vocal = pooled_q[ci] @ STATES[:, 1]
                        approach = pooled_q[ci] @ STATES[:, 2]
                        for j in range(0, len(DESIGN), 4):
                            curves.append((*common, DESIGN[j], fidelity[ci,j], g0[ci,j], h[ci,j],
                                           *pooled_score[ci,j], any_group[j], heads[j], vocal[j], approach[j],
                                           pooled_q0[j] @ ANY))
    key = ["criterion", "scenario", "tau", "loss", "mechanism", "call"]
    point = pd.DataFrame(radii, columns=key + ["context", "endpoint", "interior_radius_m", "clipped_radius_m",
                                               "status", "later_segment", "index_100m"])
    tables = {
        "radii": point,
        "curves": pd.DataFrame(curves, columns=key + ["distance_m", "fidelity", "baseline_gate", "retention",
                                                     *ENDPOINTS, "p_any", "p_heads", "p_net_vocal", "p_approach",
                                                     "p_any_waves"]),
        "responders": pd.DataFrame(counts, columns=key + ["context", "event", "p_natural", "p_waves", "excess_probability",
                                                          "fraction_among_others", "expected_responders", "ident_lower", "ident_upper"]),
        "states_100m": pd.DataFrame(state_rows, columns=key + ["context", "state", "p_natural", "p_waves", "difference"]),
    }
    for name, table in tables.items():
        table.to_csv(out/f"q4_{name}.csv", index=False, float_format="%.11g")
    return point, tables["curves"]


def shape_sensitivity(fits, beta, playback, out):
    probs = probabilities(beta)
    difference = (probs[1:] - probs[0]) * context_weights(playback)[None,:,None,None]
    rows = []
    for model in ("pava", "spline", "parametric"):
        for criterion in CRITERIA:
            acoustic = predict(fits, DESIGN, model, criterion)
            for name, tau, loss in SCENARIOS:
                g0, h = gates(acoustic, tau, loss)
                for mechanism in ("shared", "both"):
                    utility = stress_kernel(mechanism, loss) @ UTILITIES
                    score = np.einsum("czds,zse->cde", difference, utility) * (g0*h)[...,None]
                    for ci,call in enumerate(CALLS):
                        for ei,endpoint in enumerate(ENDPOINTS):
                            clipped, flag, later = radius_row(score[ci,:,ei])
                            rows.append((model,criterion,name,mechanism,call,endpoint,
                                         clipped,flag,later,score[ci,AT_100,ei]))
    cols = ["model","criterion","scenario","mechanism","call","endpoint","clipped_radius_m",
            "status","later_segment","index_100m"]
    pd.DataFrame(rows,columns=cols).to_csv(out/"q4_acoustic_shape_sensitivity.csv",
                                          index=False,float_format="%.11g")


def conditional_bootstrap(playback, x, y, acoustic, out, reps):
    weights = context_weights(playback)
    groups = list(playback.groupby("expID", sort=True).indices.values())
    assert all(len(g) == 2 for g in groups)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(groups), size=(reps, len(groups)))
    scenarios = []
    for name, tau, loss in SCENARIOS:
        g0, h = gates(acoustic, tau, loss)
        for mechanism in MECHANISMS:
            kernel = stress_kernel(mechanism, loss)
            transformed = kernel @ UTILITIES
            key = (name, mechanism)
            scenarios.append((key, tau, loss, (g0 * h)[...,None], transformed))
    shape = (reps, len(scenarios), len(CALLS), len(ENDPOINTS))
    radii = np.empty(shape)
    left = np.empty(shape, dtype=bool)
    right = np.empty_like(left)
    indices = np.empty_like(radii)
    coefficients = []
    groups = np.asarray(groups)
    for i, idx in enumerate(draws):
        rows = groups[idx].ravel()
        beta = response_fit(x[rows], y[rows], PENALTY)
        coefficients.append(beta)
        probs = probabilities(beta)
        diff = (probs[1:] - probs[0]) * weights[None, :, None, None]
        for j, (_, _, _, gate, utility) in enumerate(scenarios):
            score = np.einsum("czds,zse->cde", diff, utility) * gate
            radii[i,j], left[i,j], right[i,j] = radius_grid(score)
            indices[i,j] = score[:,AT_100]
    np.savez_compressed(out/"q4_paired_bootstrap_draws.npz", pair_ids=np.array(sorted(playback.expID.unique())),
                        sampled_pair_indices=draws, fitted_coefficients=np.array(coefficients),
                        scenarios=np.array([key for key, *_ in scenarios]), calls=np.array(CALLS),
                        endpoints=np.array(ENDPOINTS), clipped_radii=radii, no_segment=left,
                        right_boundary=right, score_at_100m=indices)
    rows = []
    for j, ((name, mechanism), tau, loss, _, _) in enumerate(scenarios):
        for ci, call in enumerate(CALLS):
            for ei, endpoint in enumerate(ENDPOINTS):
                r = radii[:,j,ci,ei]
                s = indices[:,j,ci,ei]
                rows.append((name, tau, loss, mechanism, call, endpoint, reps,
                             *np.quantile(r, [.025,.50,.975]), left[:,j,ci,ei].mean(),
                             right[:,j,ci,ei].mean(), *np.quantile(s, [.025,.975])))
    result = pd.DataFrame(rows, columns=["scenario","tau","loss","mechanism","call","endpoint","draws",
                                         "clipped_q025_m","clipped_q50_m","clipped_q975_m","fraction_no_segment",
                                         "fraction_right_boundary","score100_q025","score100_q975"])
    result.to_csv(out/"q4_behavior_uncertainty.csv", index=False, float_format="%.11g")
    return result


def priority_table(curves, out):
    """Where is the contrast between shared and stressed transfer largest? Supports allocation only."""
    keys = ["criterion","scenario","call","distance_m"]
    primary = curves.query("criterion == 'mean' and mechanism == 'shared'").set_index(keys)
    varied = curves.query("criterion == 'mean' and mechanism == 'both'").set_index(keys)
    delta = varied[list(ENDPOINTS)] - primary[list(ENDPOINTS)]
    rows = []
    for (criterion, scenario, call), group in delta.groupby(level=[0,1,2]):
        if np.max(np.abs(group.to_numpy())) < 1e-12:
            continue
        for endpoint in ENDPOINTS:
            gap = group[endpoint].droplevel([0,1,2])
            d = gap.abs().idxmax()
            rows.append((scenario,call,endpoint,int(d),gap.at[d]))
    pd.DataFrame(rows, columns=["scenario","call","endpoint","peak_separation_distance_m","index_difference_both_minus_shared"]).to_csv(
        out/"q4_mechanism_sampling_targets.csv", index=False,float_format="%.11g")


def plot(points, folder):
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style("en")
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1795651200")
    plt.rcParams["svg.hashsalt"] = "apmcm-q4-v4"
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Noto Sans CJK SC", "Hiragino Sans GB", "DejaVu Sans"]
    sub = points.query("criterion == 'mean' and context == 'pooled'")
    colors = ["#206888", "#C6582B"]
    fig, axes = plt.subplots(1,2,figsize=(6.6,2.95),layout="none",sharey=True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.185, top=.74, wspace=.13)
    for e, (ax, endpoint, title) in enumerate(zip(axes, ENDPOINTS, ("a　联系代理：净增鸣叫","b　招募代理：靠近"))):
        for a, (scenario, mechanism, label) in enumerate((("Fidelity loss 20%","shared","统一门控"),
                                                           ("Fidelity loss 20%","both","差异抑制与警戒"))):
            ss = sub[(sub.endpoint==endpoint)&(sub.scenario==scenario)&(sub.mechanism==mechanism)].set_index("call").loc[list(CALLS)]
            for j, row in enumerate(ss.itertuples()):
                marker = {"crossing": "o", "right_boundary": "^", "no_segment_at_lower_bound": "v"}[row.status]
                ax.scatter(j+(a-.5)*.16, row.clipped_radius_m, marker=marker, s=38,
                           edgecolor=colors[a], facecolor=colors[a] if a==0 else "white", zorder=5,
                           linewidth=1.4, clip_on=False)
        ax.set(xticks=[0,1,2], xticklabels=["平直型","调制型","双声型"], xlim=(-.45,2.45),
               ylim=(16,154), yticks=[20,50,75,100,125,150])
        ax.set_title(title,loc="left",fontsize=9,pad=6)
        ax.grid(axis="y",color=".86",lw=.6)
    axes[0].set_ylabel("功能半径（m）")
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],ls="none",marker="o",mfc=c if i==0 else "white",mec=c,mew=1.4,label=l)
               for i,(c,l) in enumerate(zip(colors, ("统一门控","差异抑制与警戒")))]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
    save(fig,folder,"result_q4_contact_recruitment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--figures",type=Path,default=HERE/"figures")
    parser.add_argument("--replicates",type=int,default=N_BOOT)
    parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    _, playback, correlation = read_inputs()
    assert np.array_equal(playback.VocalYesNo, playback.VocalNcalls > playback.VocalNcallsBefore)
    x = design(playback.CallType,playback.Context,playback.Distance,playback.Npenguins)
    y = state_index(event_flags(playback))
    beta = response_fit(x,y,PENALTY)
    fits = fit_primary(summary_cells(correlation)[1])
    fidelities = {c: predict(fits, DESIGN, "pava", c) for c in CRITERIA}
    shape_sensitivity(fits, beta, playback, args.output)
    points, curves = point_tables(beta,playback,fidelities,args.output)
    priority_table(curves,args.output)
    if not args.no_figures:
        args.figures.mkdir(parents=True,exist_ok=True)
        plot(points,args.figures)
    conditional_bootstrap(playback,x,y,fidelities["mean"],args.output,args.replicates)
    print(f"Q4 completed: {len(points)} scenario-context-endpoint estimates, {args.replicates} paired refits.")


if __name__ == "__main__":
    main()
