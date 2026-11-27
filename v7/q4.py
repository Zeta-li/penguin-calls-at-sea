"""Separate acoustic loss from a vigilance-limited response clock using two field experiments."""
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
V6 = HERE.parent / "v6"
OUT = HERE / "results"
sys.path.insert(0, str(V6))
from models_effectiveness import SCENARIOS, functional_radius
from penguin_data import CALLS, STATES, input_path

NOISE = HERE / "data" / "noise_stony_point"
HASHES = (
    "9dd355a9d5270be156e88b18319fa6f798efc33d70b8900a0b28ea4b851c5d2f",
    "708d90aa1d56a59e39f2b8193cf43592dc627ac0985cafac513f10e07096c303",
    "39319feda43812ee821c7178c05dc24b3d1b1912a0a7f46eb56b63ff01e73bf1",
)
SEED = 20261127


def write(name, rows):
    pd.DataFrame(rows).to_csv(OUT / f"q4v7_{name}.csv", index=False, float_format="%.12g")


def lost_fraction(before, during):
    return (np.mean(during) - np.mean(before)) / (1 - np.mean(before))


def shore():
    tables = []
    for number, expected in enumerate(HASHES, 1):
        path = NOISE / f"DataTable_analysis{number}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        tables.append(pd.read_csv(path, sep=";", encoding="utf-8-sig"))
    group, score, scans = tables
    # Require the same pre/during/post individual set, including explicit missing values.
    counts = scans.filter(regex="_count$")
    assert counts.isna().all(axis=1).equals(counts.isna().any(axis=1))
    scans["fraction"] = scans.vigilance_count / counts.sum(axis=1, min_count=1)
    individual = scans.pivot(index=["video", "noise", "individual"], columns="time",
                             values="fraction")
    assert individual.isna().all(axis=1).equals(individual.isna().any(axis=1))
    videos = individual.groupby(["video", "noise"]).mean().reset_index()
    videos["n_enrolled"] = individual.groupby(["video", "noise"]).size().to_numpy()
    videos["n_scored"] = individual.groupby(["video", "noise"]).Playback.count().to_numpy()
    labels = group.loc[group.time == "t0"].set_index("video").noise
    score_labels = score.groupby("video").noise.first()
    assert set(videos.video) == set(labels.index) == set(score_labels.index)
    videos["noise_avoidance_table"] = videos.video.map(labels)
    videos["noise_score_table"] = videos.video.map(score_labels)
    write("shore_videos", videos)
    observed = videos.dropna(subset=["Playback"]).copy()
    # Equal weight for each experimental video, then for individuals within video.
    rng = np.random.default_rng(SEED)
    summary = []
    for level, block in [("pooled", observed), *observed.groupby("noise")]:
        pre, play, post = block[["Pre_playback", "Playback", "Post_playback"]].to_numpy().T
        idx = rng.integers(len(block), size=(5000, len(block)))
        samples = (play[idx].mean(axis=1)-pre[idx].mean(axis=1))/(1-pre[idx].mean(axis=1))
        lo, hi = np.quantile(samples, [.025, .975])
        summary.append(dict(level=level, videos=len(block), individuals=block.n_scored.sum(),
                            pre=pre.mean(), during=play.mean(), post=post.mean(),
                            lost=lost_fraction(pre, play), lo_video=lo, hi_video=hi))
    write("shore_budget", summary)
    oof = []
    for row in observed.itertuples():
        train = observed[observed.video != row.video]
        group_train = train[train.noise == row.noise]
        for name, subset in [("unchanged", train), ("pooled", train), ("by_level", group_train)]:
            lost = 0 if name == "unchanged" else lost_fraction(subset.Pre_playback, subset.Playback)
            pred = np.clip(row.Pre_playback + (1-row.Pre_playback)*lost, 0., 1.)
            oof.append(dict(video=row.video, level=row.noise, model=name,
                            observed=row.Playback, predicted=pred))
    oof = pd.DataFrame(oof)
    write("shore_oof", oof)
    oof["sq"] = (oof.predicted-oof.observed)**2
    write("shore_scores", oof.groupby("model", as_index=False).agg(
        n=("sq","size"), mse=("sq","mean")).assign(rmse=lambda d: np.sqrt(d.mse)))
    # Missing complete individual time budgets may differ systematically from scored individuals.
    w = videos.n_scored / videos.n_enrolled
    pre_min = (w * videos.Pre_playback.fillna(0.)).mean()
    during_min = (w * videos.Playback.fillna(0.)).mean()
    unscored = (1 - w).mean()
    low = (during_min - pre_min-unscored)/(1-pre_min-unscored)
    high = (during_min+unscored - pre_min)/(1-pre_min)
    write("shore_missing", [dict(videos=len(videos), birds=videos.n_enrolled.sum(),
                                 scored=videos.n_scored.sum(), missing_video_weight=unscored,
                                 min_loss=low, max_loss=high)])
    return summary[0]


class EventClock:
    """Empirical interval likelihood among positive trials; retain completion bounds."""
    def __init__(self, times):
        self.lo, self.hi = times[["lo", "hi"]].to_numpy().T
        exact = self.lo == self.hi
        self.support = np.unique(self.hi)
        at = self.support[None, :]
        # If playback already had calls, net crossing is at a later call than first onset.
        accept = np.where(exact[:, None], at == self.lo[:, None],
                          (at > self.lo[:, None]) & (at <= self.hi[:, None]))
        assert np.all(accept.any(axis=1))
        mass = np.full(len(self.support), 1 / len(self.support))
        for _ in range(10000):
            updated = mass * (accept / (accept @ mass)[:, None]).mean(axis=0)
            if np.max(np.abs(updated - mass)) < 1e-12:
                mass = updated
                break
            mass = updated
        else:
            raise RuntimeError("Interval likelihood did not converge")
        self.mass = mass / mass.sum()

        def nll(log_scale):
            scale = np.exp(log_scale)
            normalizer = -np.expm1(-60 / scale)
            density = np.exp(-self.lo / scale) / (scale * normalizer)
            interval = (np.exp(-self.lo / scale) *
                        -np.expm1(-(self.hi - self.lo) / scale) / normalizer)
            return -np.log(np.where(exact, density, interval)).sum()

        fit = minimize_scalar(nll, bounds=(np.log(.1), np.log(10000)), method="bounded")
        if not fit.success:
            raise RuntimeError("Truncated exponential clock did not converge")
        self.scale = np.exp(fit.x)

    def exponential(self, t):
        return -np.expm1(-t / self.scale) / -np.expm1(-60 / self.scale)

    def at(self, t):
        low = (self.hi <= t).mean()
        high = np.where(self.lo == self.hi, self.lo <= t, self.lo < t).mean()
        central = self.mass[self.support <= t].sum()
        # The fitted distribution can redistribute censored observations only inside their intervals.
        assert low-1e-8 <= central <= high+1e-8
        return np.clip([low, central, high], 0., 1.)


def sea_clocks():
    df = pd.read_csv(input_path(2))
    net = df.loc[df.VocalNcalls > df.VocalNcallsBefore].copy()
    approach = df.loc[df.ApproachYesNo == 1].copy()
    records = {
        "contact": pd.DataFrame(dict(pair=net.expID, distance=net.Distance, role=net.Type, lo=net.VocalLatency,
                                    hi=np.where(net.VocalNcallsBefore.eq(0), net.VocalLatency, 60.))),
        "recruitment": pd.DataFrame(dict(pair=approach.expID, distance=approach.Distance, role=approach.Type, lo=approach.ApproachLatency,
                                        hi=approach.ApproachLatency)),
    }
    folds = list(GroupKFold(n_splits=5).split(df, groups=df.expID))
    oof, summary = [], []
    for task, times in records.items():
        assert times.notna().all().all()
        assert times.lo.between(0., 60.).all() and times.hi.ge(times.lo).all()
        clock = EventClock(times)
        for t in range(61):
            lo, est, hi = clock.at(t)
            summary.append(dict(endpoint=task, time_s=t, n=len(times),
                                intervals=(times.lo != times.hi).sum(), cdf_low=lo,
                                cdf=est, cdf_high=hi, exponential=clock.exponential(t)))
        for fold, (train, test) in enumerate(folds):
            tr = times[times.pair.isin(df.expID.iloc[train])]
            te = times[times.pair.isin(df.expID.iloc[test])]
            fit = EventClock(tr)
            for t in [10., 30., 45.]:
                # Exclude interval-censored test labels unresolved at the scoring time.
                known = (te.hi <= t) | (te.lo > t) | ((te.hi != te.lo) & (te.lo == t))
                y = (te.loc[known, "hi"] <= t).astype(float)
                for model, pred in [("empirical_interval", fit.at(t)[1]), ("uniform", t/60),
                                    ("truncated_exponential", fit.exponential(t))]:
                    for ix, event in y.items():
                        oof.append(dict(endpoint=task, fold=fold, pair=te.loc[ix, "pair"],
                                        t=t, model=model, y=event, prediction=pred,
                                        loss=(pred-event)**2))
    oof = pd.DataFrame(oof)
    write("clock_oof", oof)
    write("clock_scores", oof.groupby(["endpoint","model","t"], as_index=False).agg(
        scored=("loss","size"), brier=("loss","mean")))
    write("clocks", summary)
    clocks = {k: EventClock(v) for k, v in records.items()}
    for task, block in records.items():
        clocks[task + "_inside"] = EventClock(block[block.distance.between(20, 150)])
    return clocks


def state_inputs():
    params = pd.read_csv(OUT / "q2v7_parameters.csv").query("model=='v6_parametric'")
    states = pd.read_csv(OUT / "q3_state_grid.csv").query("model=='selected' and group_size==8")
    keys = [f"p_{s:03b}" for s in range(8)]
    distance = np.arange(20., 151.)
    p, fidelity = {}, {}
    for call in ("control", *CALLS):
        part = states[states.call == call].set_index("distance_m").loc[distance]
        p[call] = part[keys].to_numpy()
        np.testing.assert_allclose(p[call].sum(axis=1), 1., atol=1e-9)
        np.testing.assert_allclose(p[call] @ STATES, part[["p_heads","p_net_vocal","p_approach"]], atol=1e-9)
        if call != "control":
            par = params[params.call == call]
            waves = [r.a / (1 + (distance / r.b_m)**r.c) for r in par.itertuples()]
            assert len(waves) == 3
            fidelity[call] = np.mean(waves, axis=0)
    return distance, p, fidelity


def scenarios(budget, clocks):
    distance, p, fidelity = state_inputs()
    rows, people, critical = [], [], []
    c = p["control"]
    pos = {int(d): i for i, d in enumerate(distance)}
    any_function = (STATES[:,1] + STATES[:,2]) > 0
    sensitivity = []
    for tau_name, tau, loss in SCENARIOS:
        for call in CALLS:
            f = fidelity[call]
            h = np.minimum(1., expit(((1-loss)*f-tau)/.05) / expit((f-.5)/.05))
            gated = c + h[:,None]*(p[call]-c)
            delta = (gated-c) @ STATES[:,1:]
            # Complete conservative V6 and V7 comparison at matching offshore conditions.
            for eta in [0., .25, .50, .75, 1.]:
                active = 60*(1-eta*budget["lost"])
                v, a = clocks["contact"].at(active), clocks["recruitment"].at(active)[1]
                for weight, label in [(1., "contact"), (0., "recruitment"), (.5, "equal_priority"),
                                      (.25, "recruitment_priority"), (.75, "contact_priority")]:
                    for bound, qv in zip(["cdf_lower","interval_mle","cdf_upper"], v):
                        # For positive excess the completion bounds propagate in the same order.
                        d = weight*delta[:,0]*qv+(1-weight)*delta[:,1]*a
                        r, status, later = functional_radius(distance, d, .05)
                        rows.append(dict(acoustic=tau_name, call=call, eta=eta, clock=bound,
                                         endpoint=label, vocal_weight=weight, active_s=active,
                                         retention_v=qv, retention_a=a, radius_clipped_m=r,
                                         status=status, later_segment=later, excess_100m=d[pos[100]]))
                # At least one functional response: default conditional independence of
                # thinning given 60 s state, with Frechet bounds preserving both marginals.
                pv = v[1] * STATES[:,1]
                pa = a * STATES[:,2]
                pu = pv + pa - pv*pa
                for d in [50, 100, 150]:
                    j = pos[d]
                    for role, prob in [("natural",gated[j]), ("waves",c[j])]:
                        p_fun = prob @ pu
                        people.append(dict(acoustic=tau_name, call=call, eta=eta, distance_m=d,
                            stimulus=role, functional_any=p_fun,
                            union_low=prob @ np.maximum(pv, pa), union_high=prob @ np.minimum(1.,pv+pa),
                            p_contact=prob @ pv, p_recruitment=prob @ pa,
                            expected_min=p_fun, expected_half=4.5*p_fun, expected_max=8*p_fun))
                if eta == 0:
                    assert np.allclose(v, 1.) and abs(a-1.) < 1e-9
                    np.testing.assert_allclose(gated @ pu, gated @ any_function, atol=1e-10)
            # Diagnose which assumptions matter at full shore-to-sea transfer.
            core_time = 60 * (1-budget["lost"])
            for name, active, method in [
                ("central", core_time, "pooled"),
                ("shore_low_loss", 60*(1-budget["lo_video"]), "pooled"),
                ("shore_high_loss", 60*(1-budget["hi_video"]), "pooled"),
                ("inside_20_150m", core_time, "inside"),
                ("smooth_exponential", core_time, "exponential"),
                ("time_proportional", core_time, "uniform"),
            ]:
                for k, task in enumerate(["contact","recruitment"]):
                    if method == "uniform":
                        retain = active/60
                    elif method == "exponential":
                        retain = clocks[task].exponential(active)
                    else:
                        source = clocks[task + "_inside"] if method == "inside" else clocks[task]
                        retain = source.at(active)[1]
                    d = delta[:,k]*retain
                    r, status, _ = functional_radius(distance,d,.05)
                    pn, pc = ((gated @ STATES[:,1:])[:,k], (c @ STATES[:,1:])[:,k])
                    # All observed confirmed approaches were to natural calls. Allow
                    # complete uncertainty over the unobserved control time retention.
                    uncalibrated_low = pn*retain-pc
                    uncalibrated_high = pn*retain
                    j = pos[100]
                    sensitivity.append(dict(acoustic=tau_name, call=call, endpoint=task, clock=name,
                        active_s=active, time_retained=retain, radius_clipped_m=r, status=status,
                        excess_100m=d[j], control_free_low100=uncalibrated_low[j],
                        control_free_high100=uncalibrated_high[j]))
            # The break-even transport needed to drop under a 5-point net benefit at
            # a fixed distance uses monotonicity of the two measured conditional clocks.
            for d in [50,100,150]:
                j = pos[d]
                for k, task in enumerate(["contact","recruitment"]):
                    base = delta[j,k]
                    def holds(eta):
                        return base*clocks[task].at(60*(1-eta*budget["lost"]))[1] >= .05
                    if not holds(0):
                        crossing, state = 0., "below_with_zero_transfer"
                    elif holds(1):
                        crossing, state = 1., "retains_at_full_transfer"
                    else:
                        lo, hi = 0., 1.
                        for _ in range(45):
                            mid = (lo+hi)/2
                            if holds(mid):
                                lo = mid
                            else:
                                hi = mid
                        crossing, state = hi, "crosses_within_range"
                    critical.append(dict(acoustic=tau_name, call=call, distance_m=d, endpoint=task,
                                         eta_critical_clipped=crossing, status=state))
    write("radii", rows)
    write("responders", people)
    write("critical", critical)
    write("sensitivity", sensitivity)
    # Verify that removing the new clock and retaining Q6 behaviour reproduces its
    # published primary results. 1 m (Q7) vs 0.25 m (Q6) grids imply interpolation error.
    ref = pd.read_csv(V6/"第六版实验/results/q4v6_primary_specifications.csv")
    ref = ref.query("criterion=='mean' and mechanism=='shared' and model=='parametric'").copy()
    ref["endpoint"] = ref.endpoint.map({"contact_net_vocal":"contact","recruitment_approach":"recruitment"})
    ref = ref.dropna(subset=["endpoint"]).rename(columns={"scenario":"acoustic","status":"old_status"})
    base = pd.DataFrame(rows).query("eta==0 and clock=='interval_mle'")
    check = ref.merge(base,on=["acoustic","call","endpoint"],validate="one_to_one")
    check["radius_difference"] = check.radius_clipped_m-check.clipped_m
    assert len(check)==42
    # A new Q3 selected model may change radii; enforce numerical identity only if
    # the chosen model is Q6's shared design and ridge strength.
    chosen = pd.read_csv(OUT/"q3_state_grid.csv").query("model=='selected'")
    if chosen.fitted_model.eq("shared").all() and chosen.penalty.eq(2.).all():
        assert check.radius_difference.abs().max() < .1
        assert (check.status==check.old_status).all()
    write("v6_comparison",check[["acoustic","call","endpoint","clipped_m","radius_clipped_m",
                              "radius_difference","old_status","status","index100","excess_100m"]])


def main():
    OUT.mkdir(exist_ok=True)
    budget = shore()
    clocks = sea_clocks()
    scenarios(budget, clocks)


if __name__ == "__main__":
    main()
