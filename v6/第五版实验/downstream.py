"""Fourth-to-fifth edition comparisons of propagation, observed onset and functional range.

Fixed-weight interpolators are compared at held distances; onset probabilities use
the pair-held-out fourth-edition predictions. Q4 is conditional scenario analysis.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "第四版实验"
sys.path[:0] = [str(ROOT), str(PREVIOUS)]
from analyze_effectiveness import DESIGN, probabilities
from ecology import AT_100, ENDPOINTS, PENALTY, UTILITIES, gates, radius_row, stress_kernel
from models_effectiveness import SCENARIOS
from models_playback import design, response_fit
from penguin_data import CALLS, DISTANCES, METRICS, SEED, context_weights, event_flags, read_inputs, state_index
from propagation_v4 import CRITERIA, MODELS, THRESHOLDS, aggregate, components, fit_primary, invert, summary_cells

BLEND = "spline_sigmoid_equal"
ALL_MODELS = (*MODELS, BLEND)
SEED_PAIRS = SEED + 505
N_DRAWS = 5000


def mixed_components(fit, distance):
    return (components(fit, distance, "spline") + components(fit, distance, "parametric")) / 2


def evaluate_distances(means):
    baseline = pd.read_csv(PREVIOUS / "results/q2_held_distances.csv")
    rows = []
    for hold in range(1, len(DISTANCES)-1):
        keep = np.arange(len(DISTANCES)) != hold
        fit = fit_primary(means[:, :, keep], DISTANCES[keep])
        mixture = mixed_components(fit, [DISTANCES[hold]])
        for criterion in CRITERIA:
            observed = aggregate(means, criterion)[:, hold]
            prediction = aggregate(mixture, criterion)[:, 0]
            for call, value, estimate in zip(CALLS, observed, prediction):
                rows.append((call, DISTANCES[hold], criterion, BLEND, value, estimate, estimate-value))
    combined = pd.concat([baseline, pd.DataFrame(rows, columns=baseline.columns)], ignore_index=True)
    scores = []
    for (criterion, model), group in combined.groupby(["criterion", "model"]):
        for call, part in [("pooled", group), *group.groupby("call")]:
            errors = part.error.to_numpy()
            scores.append((criterion, model, call, len(part), np.sqrt((errors**2).mean()), np.abs(errors).mean()))
    return combined, pd.DataFrame(scores, columns=["criterion", "model", "call", "n", "rmse", "mae"])


def propagation_results(means):
    fit = fit_primary(means)
    grid = np.exp2(np.linspace(2, 8, 241))
    parts = mixed_components(fit, grid)
    assert np.isfinite(parts).all() and np.max(np.diff(parts, axis=2)) < 1e-8
    curves, ranges = [], []
    for k, call in enumerate(CALLS):
        series = [*parts[k], *(aggregate(parts, c)[k] for c in CRITERIA)]
        for name, values in zip((*METRICS, *CRITERIA), series):
            curves.extend((call, name, d, v) for d, v in zip(grid, values))
        for criterion in CRITERIA:
            def value(d):
                return aggregate(mixed_components(fit, d), criterion)[k]
            for threshold in THRESHOLDS:
                rad, status, left, right = invert(value, threshold)
                ranges.append((call, criterion, threshold, rad, status, left, right))
    return (fit, pd.DataFrame(curves, columns=["call","component_or_criterion","distance_m","fidelity"]),
            pd.DataFrame(ranges, columns=["call","criterion","threshold","radius_m","status",
                                          "last_passing_node_m","next_failing_node_m"]))


def onset_results(playback):
    frame = pd.read_csv(PREVIOUS / "results/q3v4_timing_oof.csv")
    key = ["source_csv_line","behaviour","time_s"]
    assert len(frame) == 278*12 and not frame.duplicated(key).any()
    assert frame.groupby("expID").fold.nunique().eq(1).all()
    onset = {"HeadsUp":"HeadsUpYesNo", "Vocal":"VocalNcalls", "Approach":"ApproachYesNo"}
    for name, flag in onset.items():
        sub = frame[frame.behaviour == name]
        data = playback.iloc[sub.source_csv_line.to_numpy()-2]
        observed = (data[flag].to_numpy()>0) & (data[name+"Latency"].to_numpy()<=sub.time_s.to_numpy())
        np.testing.assert_array_equal(observed, sub.observed.to_numpy().astype(bool))
        np.testing.assert_array_equal(data.expID.to_numpy(), sub.expID.to_numpy())
    frame["mixture"] = .75 * frame.two_part + .25 * frame.cox
    assert frame.mixture.between(0,1).all()
    for _, sample in frame.sort_values("time_s").groupby(["source_csv_line","behaviour"]):
        assert (np.diff(sample.mixture) >= -1e-12).all()
    score_cols = ["two_part", "cox", "mixture"]
    loss = frame.copy()
    loss[score_cols] = (frame[score_cols].sub(frame.observed, axis="index"))**2
    grouped = loss.groupby(["expID","behaviour","time_s"])[score_cols].mean()
    rng = np.random.default_rng(SEED_PAIRS)
    n = playback.expID.nunique()
    weights = rng.multinomial(n, np.full(n, 1/n), size=N_DRAWS) / n
    sections = [("all",0,grouped)]
    for (name,t), sample in grouped.groupby(level=["behaviour","time_s"]):
        sections.append((name,t,sample))
    for name, sample in grouped.groupby(level="behaviour"):
        sections.append((name,0,sample))
    scores=[]
    for name,t,sample in sections:
        per_pair = sample.groupby(level="expID").mean()
        assert len(per_pair) == n
        delta = per_pair.mixture - per_pair.two_part
        lo, hi = np.quantile(weights @ delta.to_numpy(), [.025,.975])
        scores.append((name,t or np.nan,*per_pair[score_cols].mean(),delta.mean(),lo,hi,n,N_DRAWS))
    return frame, pd.DataFrame(scores, columns=["behaviour","time_s","two_part_brier","cox_brier",
                          "mixture_brier","delta_mix_minus_two_part","lower","upper","pairs","bootstrap_draws"])


def scenario_results(fit, playback):
    x = design(playback.CallType,playback.Context,playback.Distance,playback.Npenguins)
    beta = response_fit(x,state_index(event_flags(playback)),PENALTY)
    p = probabilities(beta)
    weights = context_weights(playback)
    diff = (p[1:]-p[0]) * weights[None,:,None,None]
    models = {m: components(fit,DESIGN,m) for m in MODELS}
    models[BLEND] = mixed_components(fit,DESIGN)
    records = []
    for model, comp in models.items():
        for criterion in CRITERIA:
            f = aggregate(comp,criterion)
            for name, tau, loss in SCENARIOS:
                g0, h = gates(f,tau,loss)
                for mechanism in ("shared","both"):
                    benefit = np.einsum("czds,zse->cde",diff,stress_kernel(mechanism,loss)@UTILITIES)
                    curve = (g0*h)[:,:,None]*benefit
                    for k,call in enumerate(CALLS):
                        for j,endpoint in enumerate(ENDPOINTS):
                            r,flag,later = radius_row(curve[k,:,j])
                            records.append((model,criterion,name,mechanism,call,endpoint,r,flag,
                                            later,curve[k,AT_100,j]))
    cols = ["model","criterion","scenario","mechanism","call","endpoint","clipped_radius_m","status",
            "later_segment","index_100m"]
    detail = pd.DataFrame(records,columns=cols)
    previous = pd.read_csv(PREVIOUS/"results/q4_acoustic_shape_sensitivity.csv")
    matched=detail[detail.model!=BLEND].merge(previous,on=cols[:6],validate="one_to_one",suffixes=("","_v4"))
    assert len(matched) == len(previous)
    for col in ("clipped_radius_m","index_100m"):
        np.testing.assert_allclose(matched[col],matched[col+"_v4"],atol=2e-7,rtol=1e-7)
    for col in ("status","later_segment"):
        np.testing.assert_array_equal(matched[col],matched[col+"_v4"])

    robust=[]
    keys=["criterion","scenario","call","endpoint"]
    for key, sub in detail.groupby(keys,sort=False):
        assert len(sub)==8
        no_seg = sub.status.eq("no_segment_at_lower_bound")
        cap = sub.status.eq("right_boundary")
        # The connected segment starting at 20 m is an interval, so the joint interval
        # ends at the earliest INDIVIDUAL crossing; tied statuses retain their meaning.
        lo, hi = sub.clipped_radius_m.min(), sub.clipped_radius_m.max()
        status = "no_segment_at_lower_bound" if no_seg.any() else ("right_boundary" if cap.all() else "crossing")
        witness = sub[no_seg] if no_seg.any() else sub[np.isclose(sub.clipped_radius_m,lo,rtol=0,atol=1e-8)]
        limiting = "|".join(witness.model+"/"+witness.mechanism)
        robust.append((*key,lo,status,hi,int(no_seg.sum()),int(cap.sum()),len(sub),limiting))
    return detail, pd.DataFrame(robust,columns=keys+["common_clipped_radius_m","common_status","largest_clipped_radius_m",
                                      "n_no_segment","n_right_boundary","specifications","limiting_specs"])


def plot_support(scenarios, output):
    """Compare signal-contact and approach radii in two distinct operating scenarios."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style("en", "report")
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1795651200")
    plt.rcParams.update({"svg.hashsalt":"q4-v5-model-support", "figure.autolayout":False,
        "figure.constrained_layout.use":False,
        "font.sans-serif":["Arial Unicode MS","Noto Sans CJK SC","Hiragino Sans GB","DejaVu Sans"]})
    fig, axes = plt.subplots(1,2,figsize=(6.55,3.1),layout="none",sharey=True)
    fig.subplots_adjust(left=.105,right=.975,bottom=.16,top=.71,wspace=.15)
    colors = {"shared":"#186583","both":"#a94829"}
    labels = ["保序","样条","参数","混合"]
    subset = scenarios.query("criterion=='mean' and call=='tv'")
    for ax, scenario, title in zip(axes, ("Threshold 0.70","Fidelity loss 20%"),
          ("a　阈值 0.70，损失 0%","b　阈值 0.50，损失 20%")):
        sub = subset[subset.scenario == scenario]
        for e,endpoint in enumerate(ENDPOINTS[:2]):
            block = sub[sub.endpoint == endpoint]
            a, b = e*5 - .36, e*5 + 3.36
            lo = block.clipped_radius_m.min()
            ax.hlines(lo,a,b,color="#4e534f",lw=.7,ls="--",zorder=1)
            for i,model in enumerate(ALL_MODELS):
                for j,(mechanism,color) in enumerate(colors.items()):
                    row = block[(block.model==model)&(block.mechanism==mechanism)].iloc[0]
                    symbol = {"crossing":"o","right_boundary":"^","no_segment_at_lower_bound":"v"}[row.status]
                    ax.scatter(e*5+i+(j-.5)*.26,row.clipped_radius_m,marker=symbol,s=25,
                               facecolor=color if j==0 else "white",edgecolor=color,linewidth=1.0,
                               clip_on=False,zorder=4)
            ax.text(e*5+1.5,65,("联系","招募")[e],ha="center",fontsize=7)
        ax.set(xticks=[0,1,2,3,5,6,7,8],xticklabels=labels+labels,
               xlim=(-.55,8.55),ylim=(60,154),yticks=[70,90,110,130,150])
        ax.tick_params(axis="x",labelsize=7)
        ax.set_title(title,loc="left",fontsize=9)
        ax.grid(axis="y",color=".9",lw=.5)
    axes[0].set_ylabel("双声型功能半径（m）")
    legend = [Line2D([],[],ls="",marker="o",mfc=c if j==0 else "white",mec=c,
               label=l) for j,(c,l) in enumerate(zip(colors.values(),("共享门控","联合扰动")))]
    legend += [Line2D([],[],ls="",marker="^",color="#53605e",label="达到 150 m 边界"),
               Line2D([],[],ls="--",color="#4e534f",lw=.8,label="共同达标段端点")]
    fig.legend(handles=legend,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.51,.99))
    output.mkdir(exist_ok=True, parents=True)
    save(fig,output,"result_q4_model_support")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output",type=Path,default=HERE/"results")
    args.add_argument("--smoke",action="store_true")
    args.add_argument("--figures",type=Path,default=HERE/"figures")
    args.add_argument("--no-figures",action="store_true")
    opts = args.parse_args()
    _, playback, data = read_inputs()
    means = summary_cells(data)[1]
    if opts.smoke:
        keep = np.arange(len(DISTANCES)) != 2
        f = fit_primary(means[:,:,keep],DISTANCES[keep])
        assert mixed_components(f,[DISTANCES[2]]).shape == (3,3,1)
        print("Q2 real-data held-distance mixture ran.")
        return
    held,errors = evaluate_distances(means)
    fit,curves,ranges = propagation_results(means)
    timing,timing_scores = onset_results(playback)
    details,robust = scenario_results(fit,playback)
    tables={"q2v5_held":held,"q2v5_scores":errors,"q2v5_curves":curves,"q2v5_ranges":ranges,
            "q3v5_probabilities":timing,"q3v5_scores":timing_scores,"q4v5_scenarios":details,
            "q4v5_joint_support":robust}
    opts.output.mkdir(exist_ok=True,parents=True)
    for name, table in tables.items():
        table.to_csv(opts.output/(name+".csv"),index=False,float_format="%.12g")
    if not opts.no_figures:
        plot_support(details,opts.figures)
    print({name:len(table) for name, table in tables.items()})


if __name__=="__main__":
    main()
