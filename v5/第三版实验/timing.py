"""Compare right-censored Cox risks with the paired, two-part playback model."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from analyze_playback import SCORE_TIMES, fit_timing, representative_design
from models_playback import design, time_cdf
from penguin_data import (
    BEHAVIOURS, HORIZON, REPRESENTATIVE_N, SEED, STATE_NAMES, STATES,
    context_weights, event_flags, input_path, latency_values, state_index,
)

RIDGE = 4.0
N_BOOT = 5000


def fit_cox(x, event, latency):
    """Breslow tied-event partial likelihood with one fixed ridge penalty."""
    t = np.where(event, latency, HORIZON)
    times, counts = np.unique(t[event], return_counts=True)
    if not len(times):
        raise ValueError("A training fold has no events")
    at_risk = (t[None, :] >= times[:, None]).astype(float)
    event_features = x[event].sum(axis=0)

    def objective(beta):
        eta = x @ beta
        # Common shift cancels exactly between risk sums and event scores.
        shift = eta.max()
        relative = np.exp(eta - shift)
        denominators = at_risk @ relative
        log_normalizer = np.log(denominators) + shift
        value = counts @ log_normalizer - event_features @ beta + RIDGE * (beta @ beta) / 2
        gradient = x.T @ (relative * (at_risk.T @ (counts / denominators)))
        return value, gradient - event_features + RIDGE * beta

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 500})
    if not fit.success or np.max(np.abs(objective(fit.x)[1])) > 0.001:
        raise RuntimeError(f"Cox partial likelihood failed: {fit.message}")
    increments = counts / (at_risk @ np.exp(x @ fit.x))
    return fit.x, times, np.r_[0., np.cumsum(increments)]


def risk_by_time(model, x, times):
    beta, jumps, cumulative = model
    indices = np.searchsorted(jumps, times, side="right")
    return -np.expm1(-np.exp(x @ beta)[..., None] * cumulative[indices])


def data_inputs():
    data = pd.read_csv(input_path(2))
    oof = pd.read_csv(PROJECT / "results/q3_held_out_probabilities.csv")
    flags, latency = event_flags(data), latency_values(data)
    for original, cached in (("expID", "expID"), ("Type", "type"), ("CallType", "stimulus")):
        np.testing.assert_array_equal(data[original], oof[cached])
    np.testing.assert_array_equal(state_index(flags), oof.state)
    assert np.isfinite(latency[flags]).all() and np.all((latency[flags] >= 0) & (latency[flags] <= HORIZON))
    assert data.groupby("expID").Type.nunique().eq(2).all() and data.groupby("expID").size().eq(2).all()
    x = design(data.CallType, data.Context, data.Distance, data.Npenguins)
    marginal = oof[[f"p_{s}" for s in STATE_NAMES]].to_numpy() @ STATES
    return data, x, flags, latency, marginal


def predict_folds(data, x, flags, latency, marginal, smoke):
    blocks = []
    groups = data.expID.to_numpy()
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=groups), 1):
        assert not np.intersect1d(groups[train], groups[test]).size
        conditional = fit_timing(x[train], flags[train], latency[train])
        for j, behaviour in enumerate(BEHAVIOURS):
            fitted = fit_cox(x[train, 1:], flags[train, j], latency[train, j])
            cox = risk_by_time(fitted, x[test, 1:], SCORE_TIMES)
            reference = marginal[test, j, None] * time_cdf(conditional[j], x[test, None, :], SCORE_TIMES)
            observed = flags[test, j, None] & (latency[test, j, None] <= SCORE_TIMES)
            assert np.isfinite(cox).all() and np.all((cox >= 0) & (cox <= 1))
            assert np.all(np.diff(cox, axis=1) >= 0)
            np.testing.assert_allclose(reference[:, -1], marginal[test, j], atol=2e-10)
            for k, t in enumerate(SCORE_TIMES):
                blocks.append(pd.DataFrame(dict(row=test+1, expID=groups[test], fold=fold,
                    behaviour=behaviour, time_s=t, observed=observed[:, k].astype(int),
                    cox=cox[:, k], two_part=reference[:, k])))
        if smoke:
            break
    return pd.concat(blocks, ignore_index=True).sort_values(["row","behaviour","time_s"])


def score(frame):
    indicators = ["behaviour", "time_s"]
    df = frame.assign(cox_loss=(frame.cox-frame.observed)**2, reference_loss=(frame.two_part-frame.observed)**2)
    pair = df.groupby(["expID",*indicators]).agg(
        cox=("cox_loss","mean"), reference=("reference_loss","mean")).reset_index()
    keys = [(b,t) for b in BEHAVIOURS for t in SCORE_TIMES]
    by_model = [pair.pivot(index="expID", columns=indicators, values=name).reindex(columns=keys).to_numpy()
                for name in ("cox","reference")]
    n = len(by_model[0])
    weights = np.random.default_rng(SEED+303).multinomial(n, np.full(n, 1/n), size=N_BOOT)/n
    extended = [np.column_stack([m, m.mean(axis=1)]) for m in by_model]
    cox, reference = [m.mean(axis=0) for m in extended]
    lower, upper = np.quantile(weights @ (extended[0]-extended[1]), [.025,.975], axis=0)
    rows = [(*key,n,cox[i],reference[i],cox[i]-reference[i],lower[i],upper[i])
            for i,key in enumerate(keys+[("all_grid_mean",np.nan)])]
    out = pd.DataFrame(rows, columns=[*indicators,"pairs","cox_brier","two_part_brier","delta","lower","upper"])
    previous = pd.read_csv(PROJECT/"results/q3_timing_validation.csv")
    valid = out.merge(previous, on=indicators,validate="one_to_one")
    assert len(valid) == len(keys)
    np.testing.assert_allclose(valid.two_part_brier, valid.brier, atol=1e-8, rtol=0)
    return out


def scenario_results(data, x, flags, latency):
    stimuli, distances, grid = representative_design()
    weights = context_weights(data)
    records = []
    for j, behaviour in enumerate(BEHAVIOURS):
        fit = fit_cox(x[:,1:],flags[:,j],latency[:,j])
        times = np.unique(np.r_[0.,fit[1],HORIZON])
        risks = (risk_by_time(fit,grid[:,:,1:],times)*weights[None,:,None]).sum(axis=1)
        for i,(stimulus,distance) in enumerate(zip(stimuli,distances)):
            selected = (data.CallType.eq(stimulus) & data.Distance.eq(distance)).to_numpy()
            n = int(selected.sum())
            events = int(flags[selected,j].sum())
            curve = risks[i]
            conditional = times[np.flatnonzero(curve >= .5*curve[-1])[0]]
            unconditional = times[np.flatnonzero(curve >= .5)[0]] if curve[-1] >= .5 else np.nan
            records.append((stimulus,distance,REPRESENTATIVE_N,behaviour,n,events,
                            "Estimated" if events else ("No response" if n else "No trials"),
                            curve[-1],conditional,unconditional))
    return pd.DataFrame(records,columns=["stimulus","distance_m","group_size","behaviour",
        "trials_at_distance","events_at_distance","sample_status","cox_p60",
        "cox_conditional_median_projection_s","cox_unconditional_median_in_window_s"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    data, x, flags, latency, marginal = data_inputs()
    # Exact intercept-only test with tied events at second zero and a final censor.
    test_model = fit_cox(np.zeros((4,1)),np.array([True,True,True,False]),np.array([0.,0.,2.,np.nan]))
    np.testing.assert_allclose(test_model[2],[0.,.5,1.])
    frame = predict_folds(data, x, flags, latency, marginal, args.smoke)
    print("test rows:",frame.row.nunique(),"complete pairs:",frame.expID.nunique(),"predictions:",len(frame))
    if args.smoke:
        return
    assert not frame.duplicated(["row","behaviour","time_s"]).any() and len(frame)==len(data)*12
    audit = pd.DataFrame(dict(behaviour=BEHAVIOURS, events=flags.sum(axis=0), censored=(~flags).sum(axis=0),
        events_at_zero=(flags & (latency==0)).sum(axis=0),
        non_event_positive_latency=(~flags & (latency>0)).sum(axis=0)))
    results = {"oof":frame, "scores":score(frame), "scenarios":scenario_results(data,x,flags,latency), "audit":audit}
    args.output.mkdir(parents=True,exist_ok=True)
    for name, table in results.items():
        table.to_csv(args.output/f"q3v3_{name}.csv",index=False,float_format="%.12g")
    print(results["scores"].to_string(index=False))


if __name__=="__main__":
    main()
