"""Coherent playback event, intensity, and group-first-onset predictions."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "v6"
sys.path[:0] = [str(BASE / "第六版实验"), str(BASE)]
from penguin_data import CALLS, CONTEXTS, METRICS, STATES, STATE_NAMES, STIMULI, event_flags, input_path, latency_values, state_index
from models_playback import INTERACTIONS, LAMBDAS, design, time_cdf
from q3_timing import fit_clock, state_fit, state_logp
from behavior import level_fit, level_predict

RESULTS = HERE / "results"
V6 = BASE / "第六版实验" / "results"
TIMES = np.array([3., 10., 30., 60.])
KNOTS = np.array([0., 3., 10., 30., 60.])
NAMES = ("HeadsUp", "Vocal", "Approach")
FIELDS = ("HeadsUpDuration", "VocalNcalls", "ApproachDuration", "VocalRelNCalls")


def save(frame, name):
    frame.to_csv(RESULTS / f"q3_{name}.csv", index=False, float_format="%.12g")


def design_frame(frame):
    return design(frame.CallType, frame.Context, frame.Distance, frame.Npenguins)


def fit_state(x, before, y, penalty, model):
    if model == "shared":
        return state_fit(x, y, penalty)
    # One additional slope only on the vocal main effect of the eight-state model.
    n = x.shape[1] * 3
    penalties = np.full(n + INTERACTIONS.shape[1] + 1, float(penalty))
    penalties[:3] *= 0.1
    pre = np.log1p(before)
    def objective(beta):
        score = (x @ beta[:n].reshape(-1, 3)) @ STATES.T + INTERACTIONS @ beta[n:-1]
        score += beta[-1] * pre[:, None] * STATES[:, 1]
        lp = score - logsumexp(score, axis=1, keepdims=True)
        probs = np.exp(lp)
        residual = probs @ STATES - STATES[y]
        grad = np.r_[(x.T @ residual).ravel(), (probs @ INTERACTIONS - INTERACTIONS[y]).sum(axis=0), pre @ residual[:, 1]]
        return -lp[np.arange(len(y)), y].sum() + .5 * np.dot(penalties * beta, beta), grad + penalties * beta
    fitted = minimize(objective, np.zeros_like(penalties), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 500})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x


def state_probs(beta, x, before, model):
    if model == "shared":
        return np.exp(state_logp(beta, x))
    lp = state_logp(beta[:-1], x) + beta[-1] * np.log1p(before)[:, None] * STATES[:, 1]
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def inner_splits(rows, groups):
    return [(rows[a], rows[b]) for a, b in GroupKFold(4).split(rows, groups=groups[rows])]


def tune_states(x, before, y, rows, splits):
    scores = []
    fits = {}
    for model in ("shared", "baseline"):
        for lam in LAMBDAS:
            loss = []
            for a, b in splits:
                beta = fit_state(x[a], before[a], y[a], lam, model)
                probs = state_probs(beta, x[b], before[b], model)
                loss.extend(-np.log(probs[np.arange(len(b)), y[b]]))
            scores.append((model, lam, np.mean(loss)))
        _, lam, score = min((r for r in scores if r[0] == model), key=lambda r: r[2])
        fits[model] = (model, fit_state(x[rows], before[rows], y[rows], lam, model), lam, score)
    fits["selected"] = min(fits.values(), key=lambda r: r[-1])
    return fits, scores


def nonnet_branch(data):
    before = data.VocalNcallsBefore.to_numpy()
    total = data.VocalNcalls.to_numpy()
    eligible = (before > 0) & (total <= before)
    extra = eligible & (total > 0)
    p = (extra.sum() + .5) / (eligible.sum() + 1.)
    # Excess beyond the first call among 1..B has B-1 remaining slots.
    share = ((total[extra] - 1).sum() + .5) / ((before[extra] - 1).sum() + 1.)
    return p, share


def coherent_parts(net_states, before, branch):
    net = net_states @ STATES[:, 1]
    p_extra = (1 - net) * (before > 0) * branch[0]
    onset_states = net_states.copy()
    for state in (0, 1, 4, 5):
        mass = net_states[:, state] * (before > 0) * branch[0]
        onset_states[:, state] -= mass
        onset_states[:, state + 2] += mass
    onset_prob = onset_states @ STATES
    assert np.allclose(onset_prob[:, 1], net + p_extra)
    assert np.all(net <= onset_prob[:, 1] + 1e-12)
    return onset_states, onset_prob, net, p_extra


def grid(frame, fits):
    mix = frame.groupby(["Context", "VocalNcallsBefore"]).size().rename("n").reset_index()
    before = mix.VocalNcallsBefore.to_numpy()
    weights = mix.n.to_numpy() / len(frame)
    branch = nonnet_branch(frame)
    records = []
    for call in STIMULI:
        for distance in range(20, 151):
            x = design(call, mix.Context, distance, 8)
            for name, (model, beta, lam, _) in fits.items():
                p = state_probs(beta, x, before, model)
                _, onset, net, _ = coherent_parts(p, before, branch)
                row = dict(model=name, call=call, distance_m=distance, group_size=8,
                           fitted_model=model, penalty=lam, p_any_vocal=weights @ onset[:, 1],
                           p_net_vocal=weights @ net, p_approach=weights @ onset[:, 2], p_heads=weights @ onset[:, 0])
                records.append({**row, **dict(zip(["p_" + s for s in STATE_NAMES], weights @ p))})
    out = pd.DataFrame(records)
    assert np.allclose(out[["p_" + s for s in STATE_NAMES]].sum(axis=1), 1)
    save(out, "state_grid")


def fit_opportunity(times, size, rho):
    scale = (size / 8.) ** rho
    # Estimate conditional 8-bird CDF at each scoring time by Brier loss.
    knots = []
    for t in KNOTS[:-1]:
        event = (times <= t).astype(float)
        def risk(p):
            return np.mean((1. - (1. - p) ** scale - event) ** 2)
        fit = minimize_scalar(risk, bounds=(0., 1.), method="bounded", options={"xatol": 1e-11})
        knots.append(min((0., fit.x, 1.), key=risk))
    cdf = np.r_[knots, 1.]
    assert np.all(np.diff(cdf) >= -1e-7)
    return np.maximum.accumulate(cdf)


def clock_fit(x, size, times, model, parameter):
    if model == "opportunity":
        return fit_opportunity(times, size, parameter)
    return fit_clock(x, times, parameter)


def clock_predict(x, size, times, model, parameter, fit):
    if model == "opportunity":
        p = np.interp(times, KNOTS, fit)
        return 1. - (1. - p) ** ((size / 8.) ** parameter)[:, None]
    return time_cdf(fit, x[:, None, :], times)


def tune_clocks(x, size, times, rows, splits):
    fits, tuning = {}, []
    for j, label in enumerate(NAMES):
        hit = np.isfinite(times[:, j])
        candidates = [("opportunity", r) for r in (0., .5, 1.)] + [("regression", k) for k in (1., 4., 16.)]
        scores = []
        for model, parameter in candidates:
            loss = []
            for a, b in splits:
                a, b = a[hit[a]], b[hit[b]]
                if not len(b):
                    continue
                fit = clock_fit(x[a], size[a], times[a, j], model, parameter)
                p = clock_predict(x[b], size[b], TIMES[:3], model, parameter, fit)
                loss.extend(np.mean((p - (times[b, j, None] <= TIMES[:3])) ** 2, axis=1))
            scores.append((np.mean(loss), model, parameter))
        for key, subset in (
            ("selected", scores), ("opportunity", [r for r in scores if r[1] == "opportunity"]),
            ("constant", [r for r in scores if r[1:] == ("opportunity", 0.)]),
        ):
            _, model, par = min(subset)
            ix = rows[hit[rows]]
            fits[(j, key)] = (model, par, clock_fit(x[ix], size[ix], times[ix, j], model, par))
        tuning.extend((label, model, par, s, (s, model, par) == min(scores)) for s, model, par in scores)
    return fits, tuning


def fit_positive(x, target, field, name):
    if name == "mean":
        return target.mean()
    if name == "ridge":
        return level_fit(x, target, field, 4.)
    # Shifted NB, fixed dispersion r=1, log link for the mean excess.
    y = target - 1
    def objective(beta):
        eta = x @ beta
        mu = np.exp(eta)
        g = x.T @ ((mu - y) / (1. + mu))
        g[1:] += 4 * beta[1:]
        return np.sum((1 + y) * np.logaddexp(eta, 0.) - y * eta) + 2 * (beta[1:] @ beta[1:]), g
    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   bounds=[(-8., 8.)] * x.shape[1], options={"maxiter": 300, "ftol": 1e-11})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def positive_predict(x, field, name, fit):
    if name == "mean":
        return np.full(len(x), fit)
    return level_predict(x, fit, field)


def tune_levels(x, frame, train, splits):
    fits, details = {}, []
    for field in ("HeadsUpDuration", "VocalRelNCalls", "ApproachDuration"):
        y = frame[field].to_numpy()
        candidates = ["mean", "ridge"] + (["nb"] if field == "VocalRelNCalls" else [])
        choices = []
        for model in candidates:
            loss = []
            for a, b in splits:
                a, b = a[y[a] > 0], b[y[b] > 0]
                if len(b):
                    beta = fit_positive(x[a], y[a], field, model)
                    loss.extend((positive_predict(x[b], field, model, beta) - y[b]) ** 2)
            choices.append((np.mean(loss), model))
        _, model = min(choices)
        ix = train[y[train] > 0]
        fits[field] = (model, fit_positive(x[ix], y[ix], field, model))
        details.extend((field, m, s, (s, m) == min(choices)) for s, m in choices)
    return fits, details


def positive_means(x, fits):
    return {field: positive_predict(x, field, model, fit) for field, (model, fit) in fits.items()}


def mean_intensity(p, pnet, pextra, before, branch, pos):
    enet = pnet * pos["VocalRelNCalls"]
    return {
        "HeadsUpDuration": p[:, 0] * pos["HeadsUpDuration"],
        "ApproachDuration": p[:, 2] * pos["ApproachDuration"],
        "VocalRelNCalls": enet,
        "VocalNcalls": enet + pnet * before + pextra * (1 + (before - 1) * branch[1]),
    }


def parse():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--grid-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path(2))
    groups = frame.expID.to_numpy()
    assert frame.groupby("expID").size().eq(2).all() and frame.groupby("expID").Type.nunique().eq(2).all()
    before = frame.VocalNcallsBefore.to_numpy()
    flags = event_flags(frame)
    assert np.array_equal(np.maximum(0, frame.VocalNcalls - before), frame.VocalRelNCalls)
    y = state_index(flags)
    x = design_frame(frame)
    all_rows = np.arange(len(frame))
    fits, full_state_tuning = tune_states(x, before, y, all_rows, inner_splits(all_rows, groups))
    grid(frame, fits)
    print("Grid ready:", fits["selected"][0], "penalty", fits["selected"][2], flush=True)
    if not args.grid_only:
        run_full(frame, groups, before, flags, y, x, fits, full_state_tuning)


def standardized_scenarios(frame, fitted, clocks, levels):
    mix = frame.groupby(["Context", "VocalNcallsBefore"]).size().rename("n").reset_index()
    before = mix.VocalNcallsBefore.to_numpy()
    mix_weights = mix.n.to_numpy() / len(frame)
    branch = nonnet_branch(frame)
    model, beta, _, _ = fitted
    rows = []
    for call in STIMULI:
        for d in (50, 100, 150):
            n = np.full(len(mix), 8)
            x = design(call, mix.Context, d, n)
            states = state_probs(beta, x, before, model)
            _, p, net, extra = coherent_parts(states, before, branch)
            intensity = mean_intensity(p, net, extra, before, branch,
                                       positive_means(np.column_stack((x, np.log1p(before))), levels))
            at = frame.loc[frame.CallType.eq(call) & frame.Distance.eq(d)]
            for context in ("standardized", *CONTEXTS):
                if context == "standardized":
                    weights = mix_weights
                    observed_cell = at
                else:
                    weights = mix_weights * mix.Context.eq(context).to_numpy()
                    weights = weights / weights.sum()
                    observed_cell = at.loc[at.Context.eq(context)]
                for j, name in enumerate(NAMES):
                    prob = weights @ p[:, j]
                    clock, parameter, fit = clocks[(j, "selected")]
                    def cdf(t):
                        cond = clock_predict(x, n, np.atleast_1d(t), clock, parameter, fit)
                        return (weights * p[:, j]) @ cond
                    low, high = 0., 60.
                    for _ in range(32):
                        middle = (low + high) / 2
                        if cdf(middle)[0] < prob / 2:
                            low = middle
                        else:
                            high = middle
                    target = NAMES[j] + "YesNo" if j != 1 else "VocalNcalls"
                    event_n = observed_cell[target].gt(0).sum()
                    row = dict(call=call, distance_m=d, group_size=8, behavior=name, context=context,
                               trials_at_distance=len(observed_cell), events_at_distance=event_n, exact_size_trials=int(observed_cell.Npenguins.eq(8).sum()),
                               p_event=prob, conditional_median_s=(low + high) / 2 if event_n else np.nan,
                               p_net_vocal=weights @ net, mean_net_count_all=weights @ intensity["VocalRelNCalls"],
                               positive_clock=clock, parameter=parameter)
                    row["mean_recorded_level"] = weights @ intensity[FIELDS[j]]
                    row.update({f"cdf_{int(t)}s": float(z) for t, z in zip(TIMES, cdf(TIMES))})
                    rows.append(row)
    table = pd.DataFrame(rows)
    save(table.loc[table.context.eq("standardized")], "scenarios")
    save(table.loc[table.context.ne("standardized")], "context_scenarios")


def run_full(frame, groups, before, flags, y, x, full_fits, full_tuning):
    size = frame.Npenguins.to_numpy()
    times = latency_values(frame)
    anyflags = flags.copy()
    anyflags[:, 1] = frame.VocalNcalls.gt(0)
    times[~anyflags] = np.nan  # Five unconfirmed approach latencies do not define confirmed events.
    xa = np.column_stack((x, np.log1p(before)))
    idx = np.arange(len(frame))
    folds = list(GroupKFold(5).split(idx, groups=groups))
    level_tune, clock_tune, state_tune = [], [], [("full", *r) for r in full_tuning]
    state_oof, clock_oof, level_oof = [], [], []
    old_clock = pd.read_csv(V6 / "q3v6_timing_oof.csv")
    q = old_clock.pivot(index="source_csv_line", columns=["behaviour", "time_s"], values="shared").sort_index()
    for fold, (a, b) in enumerate(folds, 1):
        assert not set(groups[a]).intersection(groups[b])
        splits = inner_splits(a, groups)
        assert all(not set(groups[i]).intersection(groups[k]) for i, k in splits)
        fits, scores = tune_states(x, before, y, a, splits)
        state_tune.extend((fold, *r) for r in scores)
        clocks, tune = tune_clocks(x, size, times, a, splits)
        clock_tune.extend((fold, *r) for r in tune)
        levels, tune = tune_levels(xa, frame, a, splits)
        level_tune.extend((fold, *r) for r in tune)
        pos = positive_means(xa[b], levels)
        branch = nonnet_branch(frame.iloc[a])
        for version, (model, beta, lam, _) in fits.items():
            states = state_probs(beta, x[b], before[b], model)
            onset_states, p, net, extra = coherent_parts(states, before[b], branch)
            state_oof.append(pd.DataFrame(dict(source_csv_line=b+2, expID=groups[b], fold=fold, model=version,
                          fitted_model=model, penalty=lam, source_state=y[b], recorded_state=state_index(anyflags[b]),
                          net_nll=-np.log(states[np.arange(len(b)), y[b]]),
                          recorded_nll=-np.log(onset_states[np.arange(len(b)), state_index(anyflags[b])]),
                          p_heads=p[:, 0], p_any_vocal=p[:, 1], p_net_vocal=net, p_approach=p[:, 2],
                          p_extra_if_eligible=branch[0], share_excess_if_extra=branch[1],
                          **{"p_"+s: states[:, k] for k, s in enumerate(STATE_NAMES)})))
            marg = mean_intensity(p, net, extra, before[b], branch, pos)
            for field in FIELDS:
                level_oof.append(pd.DataFrame(dict(source_csv_line=b+2, expID=groups[b], fold=fold, model=version,
                              field=field, observed=frame.iloc[b][field].values, predicted=marg[field])))
            if version != "selected":
                continue
            for j, name in enumerate(NAMES):
                for clk in ("selected", "opportunity", "constant", "v6_conditional"):
                    if clk == "v6_conditional":
                        old = q.loc[b+2, name].to_numpy()
                        cdf = old / old[:, [-1]]
                    else:
                        cdf = clock_predict(x[b], size[b], TIMES, *clocks[(j, clk)])
                    assert np.allclose(cdf[:, -1], 1.)
                    for k, t in enumerate(TIMES):
                        clock_oof.append(pd.DataFrame(dict(source_csv_line=b+2, expID=groups[b], fold=fold, behaviour=name,
                            time_s=t, model=clk, present=anyflags[b, j], observed=(anyflags[b, j] & (times[b, j] <= t)),
                            conditional_cdf=cdf[:, k], risk=p[:, j]*cdf[:, k])))
    state = pd.concat(state_oof, ignore_index=True)
    clock = pd.concat(clock_oof, ignore_index=True)
    levels = pd.concat(level_oof, ignore_index=True)
    assert state.groupby("model").source_csv_line.nunique().eq(278).all()
    assert not state.duplicated(["model","source_csv_line"]).any()
    assert not clock.duplicated(["model","behaviour","time_s","source_csv_line"]).any()
    assert not levels.duplicated(["model","field","source_csv_line"]).any()
    save(state, "state_oof")
    save(clock, "timing_oof")
    save(levels, "intensity_oof")
    all_splits = inner_splits(idx, groups)
    clk, selected = tune_clocks(x, size, times, idx, all_splits)
    clock_tune.extend(("full", *r) for r in selected)
    lev, selected = tune_levels(xa, frame, idx, all_splits)
    level_tune.extend(("full", *r) for r in selected)
    standardized_scenarios(frame, full_fits["selected"], clk, lev)
    save(pd.DataFrame(state_tune, columns=["fold", "model", "penalty", "inner_log_loss"]), "state_tuning")
    save(pd.DataFrame(clock_tune, columns=["fold", "behaviour", "clock", "parameter", "inner_conditional_brier", "selected"]), "clock_tuning")
    save(pd.DataFrame(level_tune, columns=["fold", "field", "model", "inner_positive_mse", "selected"]), "level_tuning")
    summarize(frame, state, clock, levels, old_clock)
    curvature_test(frame)


def aggregate(metric, table, ref, candidates, draws, ids):
    by_pair = table.groupby("expID")
    counts = by_pair.size().reindex(ids, fill_value=0).to_numpy()
    totals = by_pair[candidates].sum().reindex(ids, fill_value=0).to_numpy()
    point = totals.sum(axis=0) / counts.sum()
    weighted_counts = draws @ counts
    valid = weighted_counts > 0
    boot = (draws[valid] @ totals) / weighted_counts[valid, None]
    if metric == "RMSE":
        point, boot = np.sqrt(point), np.sqrt(boot)
    i = candidates.index(ref)
    for j, name in enumerate(candidates):
        lower, upper = np.quantile(boot[:, j] - boot[:, i], [.025, .975])
        yield (metric, name, ref, len(table), np.count_nonzero(counts), point[j], point[j] - point[i], lower, upper)


def summarize(frame, state, clock, levels, old_clock):
    ids = np.sort(frame.expID.unique())
    draws = np.random.default_rng(2026112703).multinomial(len(ids), np.ones(len(ids))/len(ids), 5000)
    scores, calibration = [], []
    old = pd.read_csv(V6 / "q3v6_state_oof.csv").query("model=='shared'")
    old_net = old.query("endpoint=='net_response'").set_index("source_csv_line").sort_index()
    old_any = old.query("endpoint=='recorded_onset'").set_index("source_csv_line").sort_index()
    s = state.pivot(index=["source_csv_line", "expID", "fold"], columns="model")
    assert np.array_equal(s.index.get_level_values("fold"), old_net.fold)
    np.testing.assert_allclose(s.net_nll.shared.values, old_net.log_loss.values, rtol=3e-7, atol=2e-7)
    for nm, prev in [("net_nll",old_net), ("recorded_nll",old_any)]:
        tab = s[nm].copy()
        tab["v6"] = prev.log_loss.to_numpy()
        scores.extend((nm, *r) for r in aggregate("NLL", tab.reset_index(), "v6", ["v6","shared","baseline","selected"], draws, ids))
    names = ["p_"+a for a in STATE_NAMES]
    old_n, old_a = old_net[names].to_numpy() @ STATES, old_any[names].to_numpy() @ STATES
    events = [("p_heads", old_a[:, 0], frame.HeadsUpYesNo), ("p_net_vocal", old_n[:, 1], frame.VocalYesNo),
              ("p_any_vocal", old_a[:, 1], frame.VocalNcalls.gt(0)), ("p_approach", old_a[:, 2], frame.ApproachYesNo)]
    for label, prev, y in events:
        probs = s[label].copy()
        probs["v6"] = prev
        for model in probs.columns:
            # Uniform, fixed forecast bins preserve comparability across candidates.
            bins = pd.cut(probs[model], [-.001,.2,.4,.6,.8,1.001], labels=False)
            tab = pd.DataFrame(dict(bin=bins.to_numpy(), obs=y.values, prob=probs[model].values))
            for b, group in tab.groupby("bin"):
                calibration.append((label, model, b, len(group), group.obs.mean(), group.prob.mean()))
        errors = probs.sub(y.to_numpy().astype(float), axis=0).pow(2).reset_index()
        for subset, ix in [("all",np.ones(len(frame), dtype=bool)),("before_positive",frame.VocalNcallsBefore.gt(0).values)]:
            scores.extend((label+"_"+subset, *r) for r in aggregate("Brier", errors[ix], "v6", ["v6","shared","baseline","selected"], draws, ids))
    counts = pd.DataFrame({"model":["v6","shared","baseline","selected"],
              "net_exceeds_any":[int(np.sum(old_n[:, 1]>old_a[:, 1]+1e-10))] +
              [int(np.sum(s.p_net_vocal[k]>s.p_any_vocal[k]+1e-10)) for k in ("shared","baseline","selected")],
              "n_stimuli":len(frame)})
    save(counts,"coherence")
    save(pd.DataFrame(calibration, columns=["event","model","bin_id","n","observed_fraction","mean_probability"]),"calibration")
    cw = clock.pivot(index=["source_csv_line","expID","fold","behaviour","time_s","observed","present"],
                     columns="model",values="risk").reset_index()
    old_cols = ["source_csv_line","expID","fold","behaviour","time_s"]
    cw = cw.merge(old_clock[old_cols+["shared","observed"]].rename(columns={"shared":"v6","observed":"v6_obs"}),
                  on=old_cols, validate="one_to_one")
    assert np.all(cw.observed==cw.v6_obs)
    cms = ["v6","constant","opportunity","v6_conditional","selected"]
    for name, group in [("clock_all",cw), *((f"clock_{n}",g) for n,g in cw.groupby("behaviour"))]:
        group = group.copy(); group[cms] = group[cms].sub(group.observed.astype(int), axis=0).pow(2)
        scores.extend((name,*r) for r in aggregate("Brier",group,"v6",cms,draws,ids))
    # Conditional timing comparisons isolate group opportunities from occurrence modeling.
    cond = clock[(clock.present) & (clock.time_s<60)].pivot(index=old_cols+["observed"],columns="model",values="conditional_cdf").reset_index()
    cm = ["constant","opportunity","v6_conditional","selected"]
    for key, group in [("conditional_clock_all",cond), *((f"conditional_clock_{n}",g) for n,g in cond.groupby("behaviour"))]:
        group = group.copy(); group[cm] = group[cm].sub(group.observed.astype(int),axis=0).pow(2)
        scores.extend((key,*r) for r in aggregate("Brier",group,"constant",cm,draws,ids))
    old_level = pd.read_csv(V6/"q3v6_intensity_oof.csv").query("model=='shared'")
    keys = ["source_csv_line","expID","fold","field","observed"]
    levels = levels.pivot(index=keys, columns="model", values="predicted").reset_index()
    levels = levels.merge(old_level[keys+["unconditional"]].rename(columns={"unconditional":"v6"}), on=keys,validate="one_to_one")
    for field, g in levels.groupby("field"):
        ms = ["v6","shared","baseline","selected"]
        g = g.copy(); g[ms] = g[ms].sub(g.observed,axis=0).pow(2)
        scores.extend((field,*r) for r in aggregate("RMSE",g,"v6",ms,draws,ids))
    save(pd.DataFrame(scores,columns=["target","metric","model","reference","n","pairs","score","difference","lower","upper"]),"scores")


def curvature_test(frame):
    from cue_transfer import evidence_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    groups = frame.expID.to_numpy(); types = frame.CallType.to_numpy()
    logd = np.log2(frame.Distance.to_numpy()/50)
    x = np.column_stack((design_frame(frame)[:,1:], *[(types==k)*logd for k in CALLS]))
    # Choose the curve entirely by reservoir validation, before looking at playback responses.
    acoustic_scores = pd.read_csv(RESULTS / "q2v7_scores.csv").query("criterion=='mean' and threshold==0.5")
    curve_family = acoustic_scores.loc[acoustic_scores.rmse.idxmin(), "model"]
    pars = pd.read_csv(RESULTS / "q2v7_parameters.csv")
    pars = pars.loc[pars.model.eq(curve_family)].set_index(["call", "representation"])
    weights = pd.read_csv(V6 / "q2v6_weights.csv").set_index("criterion")
    curves = {k: np.zeros(len(frame)) for k in ("mean", "class_shrunk")}
    for call in CALLS:
        at = types == call
        d = frame.Distance.to_numpy()[at]
        params = pars.loc[call].loc[list(METRICS)]
        a, b, c = (params[k].to_numpy()[:, None] for k in ("a", "b_m", "c"))
        fidelity = a / (1. + (d[None, :] / b) ** c)
        for criterion in curves:
            w = weights.loc[criterion, list(METRICS)].to_numpy(dtype=float)
            curves[criterion][at] = w @ fidelity
    candidates = {"linear":None,"distance_quad":logd**2, "reservoir_mean":curves["mean"],
                  "reservoir_weighted":curves["class_shrunk"]}
    scores, oof = [], []
    ids = np.unique(groups)
    draws = np.random.default_rng(2026112704).multinomial(len(ids),np.ones(len(ids))/len(ids),5000)
    for label in ("VocalYesNo","ApproachYesNo"):
        y = frame[label].to_numpy()
        for fold,(a,b) in enumerate(GroupKFold(5).split(x,groups=groups),1):
            predictions = dict(source_csv_line=b+2, expID=groups[b], fold=fold, target=label, observed=y[b])
            for name, curve in candidates.items():
                def predict(ia,ib,c):
                    A,B=evidence_features(x, types, logd, curve, ia, ib)
                    scaler=StandardScaler().fit(A)
                    fit=LogisticRegression(C=c,solver="lbfgs",max_iter=1000,tol=1e-9).fit(scaler.transform(A),y[ia])
                    return fit.predict_proba(scaler.transform(B))[:,1]
                splits=inner_splits(a,groups)
                settings=[(np.mean(np.concatenate([(predict(i,k,c)-y[k])**2 for i,k in splits])),c) for c in (.1,1.,10.)]
                _,regularizer=min(settings)
                predictions[name]=predict(a,b,regularizer)
            oof.append(pd.DataFrame(predictions))
    table=pd.concat(oof,ignore_index=True);save(table,"curvature_oof")
    for label,group in table.groupby("target"):
        group=group.copy();names=list(candidates);group[names]=group[names].sub(group.observed,axis=0).pow(2)
        for ref in ("linear","distance_quad"):
            scores.extend((label,*r) for r in aggregate("Brier",group,ref,names,draws,ids))
    save(pd.DataFrame(scores,columns=["target","metric","model","reference","n","pairs","score","difference","lower","upper"]),"curvature_scores")


if __name__ == "__main__":
    main()
