"""Q2. Fit order-invariant propagation curves and audit what the anonymous rows identify."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "第三版实验"))
from models_acoustic import BLOCK_SAMPLES, acoustic_range, block_means, evaluate_nodes
from models_acoustic import monotone_nodes, smooth_fit, smooth_value
from penguin_data import CALLS, DISTANCES, METRICS, SEED, read_inputs
from propagation import basis, fit_curves

MODELS = ("pava", "spline", "parametric")
CRITERIA = ("mean", "all")
THRESHOLDS = (.5, .6, .7)


def summary_cells(values):
    """Sum on sorted valid values so results do not depend even numerically on row order."""
    slots = values.shape[2]
    values = np.sort(values, axis=2)
    valid = np.isfinite(values).sum(axis=2)
    total = np.nansum(values, axis=2)
    if np.any(valid == 0):
        raise ValueError("No observed correlations at a measured node")
    missing = slots - valid
    return valid, total / valid, (total - missing) / slots, (total + missing) / slots


def fit_primary(means=None, distances=DISTANCES):
    if means is None:
        means = summary_cells(read_inputs()[2])[1]
    return {
        "distances": distances,
        "pava": monotone_nodes(means, distances),
        "spline": fit_curves(means, distances),
        "parametric": np.asarray([[smooth_fit(distances, row) for row in call] for call in means])
    }


def components(fits, distances, model):
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    if distances.min() < DISTANCES[0] or distances.max() > DISTANCES[-1]:
        raise ValueError("No extrapolation outside the measured 4–256 m domain")
    coef = fits[model]
    if model == "pava":
        return np.array([[evaluate_nodes(row, distances, fits["distances"]) for row in call] for call in coef])
    if model == "spline":
        return coef @ basis(distances).T
    if model == "parametric":
        return np.array([[smooth_value(row, distances) for row in call] for call in coef])
    raise KeyError(model)


def aggregate(values, criterion):
    if criterion == "mean":
        return np.mean(values, axis=1)
    if criterion == "all":
        return np.min(values, axis=1)
    raise KeyError(criterion)


def predict(fits, distances, model="pava", criterion="mean"):
    return aggregate(components(fits, distances, model), criterion)


def invert(fn, threshold, nodes=None):
    """Return a domain-capped bound if the curve does not cross. Keep plateau right edges."""
    y = fn(DISTANCES) if nodes is None else nodes
    feasible = np.flatnonzero(y >= threshold)
    if not feasible.size:
        return DISTANCES[0], "below_4m", np.nan, DISTANCES[0]
    if feasible[-1] == len(DISTANCES) - 1:
        return DISTANCES[-1], "right_boundary", DISTANCES[-1], np.nan
    i = feasible[-1]
    lo, hi = DISTANCES[i:i + 2]
    if y[i] == threshold:
        # The criterion may switch component inside an interval, extending a tied plateau.
        left, right = lo, hi
        for _ in range(45):
            mid = (left + right) / 2
            if fn([mid])[0] >= threshold:
                left = mid
            else:
                right = mid
        radius = left
    else:
        radius = brentq(lambda d: fn([d])[0] - threshold, lo, hi)
    return radius, "crossing", lo, hi


def primary_tables(fits, valid, means, low, high):
    rows = [(call, metric, d, int(valid[k, j, i]), means[k,j,i], fits["pava"][k,j,i],
             low[k,j,i], high[k,j,i])
            for k,call in enumerate(CALLS) for j,metric in enumerate(METRICS)
            for i,d in enumerate(DISTANCES)]
    cells = pd.DataFrame(rows, columns=["call","representation","distance_m","n_valid",
                                        "mean","pava","complete_mean_min","complete_mean_max"])
    thresholds = np.round(np.arange(.45,.751,.005),3)
    rows = []
    for model in MODELS:
        for mode in CRITERIA:
            values = predict(fits,DISTANCES, model, mode)
            for k,call in enumerate(CALLS):
                def fn(d):
                    return predict(fits,d,model,mode)[k]
                for tau in thresholds:
                    r, status, left, right = invert(fn,tau,values[k])
                    bottleneck = METRICS[components(fits,[r],model)[k,:,0].argmin()]
                    rows.append((model,mode,call,tau,r,status,left,right,bottleneck))
    ranges = pd.DataFrame(rows, columns=["model","criterion","call","threshold","radius_m","status",
                                        "last_passing_node_m","next_failing_node_m","lowest_representation"])
    # Every conservative crossing must be no further than the same model's average crossing.
    pivot = ranges.pivot(index=["call","model","threshold"],columns="criterion",values="radius_m")
    assert (pivot["all"] <= pivot["mean"]+1e-6).all()
    return cells, ranges


def completion_bounds(means, low, high):
    rows = []
    pooled = []
    for arr in (low, means, high):
        pooled.append(dict(pava=monotone_nodes(arr), distances=DISTANCES))
    for k,call in enumerate(CALLS):
        for mode in CRITERIA:
            for tau in THRESHOLDS:
                result = [invert(lambda d: predict(fit,d,"pava",mode)[k],tau)[:2] for fit in pooled]
                (left,ls),(mid,ms),(right,rs) = result
                assert left <= mid+1e-8 and mid <= right+1e-8
                rows.append((call,mode,tau,left,ls,mid,ms,right,rs))
    return pd.DataFrame(rows,columns=["call","criterion","threshold","completion_lower_m",
        "lower_status","observed_point_m","point_status","completion_upper_m","upper_status"])


def distance_validation(means):
    rows=[]
    for i in range(1,len(DISTANCES)-1):
        remaining = np.arange(len(DISTANCES)) != i
        training, query = DISTANCES[remaining], DISTANCES[i]
        fit = fit_primary(means[:,:,remaining], training)
        for mode in CRITERIA:
            observed = aggregate(means, mode)
            predicted = {model: predict(fit,query,model,mode)[:,0] for model in MODELS}
            for k,call in enumerate(CALLS):
                for model in MODELS:
                    rows.append((call,query,mode,model,observed[k,i],predicted[model][k]))
                if mode=="mean":
                    direct = smooth_fit(training,observed[k,remaining])
                    rows.append((call,query,mode,"direct_composite_sigmoid",observed[k,i],
                                 smooth_value(direct,query)))
    table = pd.DataFrame(rows,columns=["call","held_distance_m","criterion","model","observed","predicted"])
    table["error"]=table.predicted-table.observed
    scores=[]
    for (mode,model),chunk in table.groupby(["criterion","model"]):
        for call,sub in [("pooled",chunk), *chunk.groupby("call")]:
            e=sub.error.to_numpy()
            scores.append((mode,model,call,len(e),np.mean(np.abs(e)),np.sqrt(np.mean(e**2))))
    return table,pd.DataFrame(scores,columns=["criterion","model","call","n","mae","rmse"])


def position_diagnostic(correlation, pooled):
    """Enumerate exactly the prior 27 positional re-draws, preserving multiplicity."""
    records=[]
    for name,ix in (("original",np.arange(300)),("shift_50",np.roll(np.arange(300),50)),
                    ("shared_permutation",np.random.default_rng(SEED).permutation(300))):
        reordered=correlation[:,:,ix,:]
        np.testing.assert_array_equal(summary_cells(reordered)[1],pooled)
        blocks=block_means(reordered)[0]
        draw_stack = np.stack([np.take(blocks,s,axis=2).mean(axis=2) for s in BLOCK_SAMPLES])
        nodes = monotone_nodes(np.concatenate((blocks.mean(axis=2)[None,...], draw_stack))).mean(axis=2)
        for k,call in enumerate(CALLS):
            for tau in THRESHOLDS:
                estimates = [acoustic_range(row,tau) for row in nodes[:,k]]
                interval = np.quantile([x[0] for x in estimates[1:]],[.025,.975])
                records.append((name,call,tau,estimates[0][0],estimates[0][1],*interval,
                          sum(x[1]=="below_4m" for x in estimates[1:]),
                          sum(x[1]=="right_boundary" for x in estimates[1:])))
    return pd.DataFrame(records,columns=["partition","call","threshold","equal_block_point_m","point_status",
        "positional_q025_m","positional_q975_m","draws_below_4m","draws_right_boundary"])


def plot_results(ranges, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style("en","report")
    matplotlib.rcParams.update({"figure.constrained_layout.use":False,"figure.autolayout":False,
        "font.sans-serif":["Arial Unicode MS","Noto Sans CJK SC","Hiragino Sans GB","DejaVu Sans"]})
    fig,axs=plt.subplots(1,2,figsize=(6.55,3.0),sharey=True,layout="none")
    fig.subplots_adjust(left=.1,right=.98,bottom=.2,top=.75,wspace=.13)
    for ax,mode,title in zip(axs,CRITERIA,["a　三项均值达标","b　三项分别达标"]):
        for call,label,color,style in zip(CALLS,["平直型","调制型","双声型"],
                 ["#405267","#007992","#AC481F"],["-", "--", "-."]):
            sub=ranges.loc[(ranges.model=="pava")&(ranges.criterion==mode)&(ranges.call==call)]
            observed=sub.status.eq("crossing")
            ax.plot(sub.threshold,sub.radius_m.where(observed),label=label,color=color,linestyle=style,lw=1.8)
            below=sub.loc[sub.status=="below_4m"]
            if not below.empty:
                first=below.iloc[0]
                ax.scatter(first.threshold,4.0,marker="v",s=25,color=color,zorder=4)
        ax.set_title(title,loc="left",fontsize=9)
        ax.set(xlabel="保真度阈值",xticks=[.45,.50,.55,.60,.65,.70,.75],xlim=(.448,.752),ylim=(0,205))
        ax.grid(axis="y",alpha=.23)
    axs[0].set_ylabel("阈值传播距离（m）")
    axs[1].text(.99,.97,"▼  首次低于 4 m 边界",ha="right",va="top",transform=axs[1].transAxes,fontsize=7)
    fig.legend(*axs[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,
               bbox_to_anchor=(.52,.99))
    output.mkdir(parents=True,exist_ok=True)
    save(fig, output, "result_q2_criteria")


def run(output=HERE/"results", figures=HERE/"figures", smoke=False):
    corr=read_inputs()[2]
    valid,means,low,high=summary_cells(corr)
    fit=fit_primary(means)
    assert (np.diff(components(fit,DISTANCES,"pava"),axis=-1)<=1e-12).all()
    if smoke:
        print("tv t=.70",invert(lambda d:predict(fit,d)[2],.7)[:2])
        return
    cells,ranges=primary_tables(fit,valid,means,low,high)
    diagnostic=position_diagnostic(corr,means)
    held,scores=distance_validation(means)
    grid=np.exp2(np.linspace(2,8,241))
    curves=[]
    for model in MODELS:
        raw=components(fit,grid,model)
        assert np.diff(raw,axis=-1).max()<1e-8
        for k,call in enumerate(CALLS):
            series=[*raw[k],raw[k].mean(axis=0),raw[k].min(axis=0)]
            for label,v in zip([*METRICS,"mean","all"],series):
                curves.extend((model,call,label,d,f) for d,f in zip(grid,v))
    curves=pd.DataFrame(curves,columns=["model","call","representation_or_criterion","distance_m","fidelity"])
    output.mkdir(parents=True,exist_ok=True)
    for name,table in (("nodes",cells),("ranges",ranges),("completion_bounds",completion_bounds(means,low,high)),
                      ("position_diagnostic",diagnostic),("held_distances",held),("scores",scores),("curves",curves)):
        table.to_csv(output/f"q2_{name}.csv",index=False,float_format="%.12g")
    if figures is not None:
        plot_results(ranges,figures)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--figures",type=Path,default=HERE/"figures")
    parser.add_argument("--no-figures",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.output,None if args.no_figures else args.figures,args.smoke)
