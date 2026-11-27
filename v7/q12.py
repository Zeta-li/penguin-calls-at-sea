"""Grouped validation of source-preserving frequency augmentation and pooled propagation."""
from functools import lru_cache
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "v6"
sys.path[:0] = [str(BASE), str(BASE / "第四版实验"), str(BASE / "第五版实验")]
import classical
from models_acoustic import smooth_fit, smooth_value
from penguin_data import CALLS, DISTANCES, METRICS, read_inputs
from propagation_v4 import invert, summary_cells

OUT = HERE / "results"
POOLING = (0., .003, .03, .3)
THRESHOLDS = (.5, .6, .7)
NAMES = np.array(CALLS)
COVERAGES = (1., .8, .6, .4)


def save(name, records):
    pd.DataFrame(records).to_csv(OUT / (name + ".csv"), index=False, float_format="%.12g")


def joint_curves(separate, distance, observed, strength):
    """Shrink log slopes by representation, allowing each call its own height and midpoint."""
    if strength == 0:
        return separate
    result = separate.copy()
    for j in range(len(METRICS)):
        start = separate[:, j].copy()
        start[:, 2] = np.log(start[:, 2])
        def residual(theta):
            a, log_b, log_c = theta.reshape(3, 3).T
            fitted = a[:, None] / (1 + np.exp(np.exp(log_c[:, None]) *
                                                (np.log(distance)[None, :] - log_b[:, None])))
            return np.r_[(fitted - observed[:, j]).ravel(),
                         np.sqrt(strength) * (log_c - log_c.mean())]
        fit = least_squares(residual, start.ravel(), bounds=(
            np.tile([0., 0., np.log(.05)], 3), np.tile([1., np.log(2048.), np.log(6.)], 3)),
            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if not fit.success:
            raise RuntimeError(f"Partial pooling failed for {METRICS[j]}: {fit.message}")
        coefs = fit.x.reshape(3, 3)
        coefs[:, 2] = np.exp(coefs[:, 2])
        result[:, j] = coefs
    return result


def parts(coefs, distance):
    return np.array([[smooth_value(c, np.atleast_1d(distance)) for c in row] for row in coefs])


def radius(coefs, call, criterion, threshold):
    def predicted(distance):
        values = parts(coefs[call:call+1], distance)[0]
        return (values.mean(axis=0) if criterion == "mean" else values.min(axis=0))
    return invert(predicted, threshold)[:2]


def run_q2(raw):
    means = summary_cells(raw)[1]
    indices = tuple(range(len(DISTANCES)))
    interior = indices[1:-1]

    @lru_cache(None)
    def unpooled(kept):
        idx = list(kept)
        return np.array([[smooth_fit(DISTANCES[idx], row[idx]) for row in call] for call in means])

    @lru_cache(None)
    def fit(kept, strength):
        return joint_curves(unpooled(kept), DISTANCES[list(kept)], means[:, :, list(kept)], strength)

    tuning, held, parameters, ranges, nodes = [], [], [], [], []
    def tune(kept, outer):
        candidates = []
        for strength in POOLING:
            errors = []
            for h in interior:
                if h not in kept:
                    continue
                training = tuple(i for i in kept if i != h)
                pred = parts(fit(training, strength), DISTANCES[h])[:, :, 0]
                errors.extend(pred.mean(axis=1) - means[:, :, h].mean(axis=1))
            rmse = np.sqrt(np.mean(np.square(errors)))
            candidates.append((rmse, strength))
            tuning.append(dict(outer=outer, pooling=strength, rmse_mean_fidelity=rmse, n=len(errors)))
        return min(candidates)[1]

    model_folds = {"v6_parametric": [], "nested_pool": []}
    for h in interior:
        keep = tuple(i for i in indices if i != h)
        best = tune(keep, f"{DISTANCES[h]:g}")
        for model, strength in (("v6_parametric", 0.), ("nested_pool", best)):
            curve = fit(keep, strength)
            model_folds[model].append(curve)
            yhat = parts(curve, DISTANCES[h])[:, :, 0]
            for criterion, reduce in (("mean", np.mean), ("all", np.min)):
                for call, truth, pred in zip(CALLS, reduce(means[:, :, h], axis=1), reduce(yhat, axis=1)):
                    held.append(dict(model=model, pooling=strength, criterion=criterion, call=call,
                                     held_distance_m=DISTANCES[h], observed=truth, predicted=pred))
    best_full = tune(indices, "full")
    for model, strength in (("v6_parametric", 0.), ("nested_pool", best_full)):
        coef = fit(indices, strength)
        fitted = parts(coef, DISTANCES)
        for k, call in enumerate(CALLS):
            for j, rep in enumerate(METRICS):
                a, log_b, c = coef[k, j]
                parameters.append(dict(model=model, call=call, representation=rep, a=a,
                                       b_m=np.exp(log_b), c=c, pooling=strength))
            for criterion, reduce in (("mean", np.mean), ("all", np.min)):
                obs, est = reduce(means[k], axis=0), reduce(fitted[k], axis=0)
                for d, y, yhat in zip(DISTANCES, obs, est):
                    nodes.append(dict(model=model, call=call, criterion=criterion, distance_m=d,
                                      measured=y, fitted=yhat))
                for tau in THRESHOLDS:
                    end, status = radius(coef, k, criterion, tau)
                    omitted = [radius(f, k, criterion, tau)[0] for f in model_folds[model]]
                    ok = np.flatnonzero(obs >= tau)
                    i = ok[-1] if len(ok) else -1
                    ranges.append(dict(model=model, call=call, criterion=criterion, threshold=tau,
                       radius_m=end, status=status, last_observed_pass_m=DISTANCES[i] if i>=0 else np.nan,
                       next_observed_fail_m=DISTANCES[i+1] if i+1<len(DISTANCES) else np.nan,
                       passes_at_4m=bool(obs[0]>=tau), reentries=int(np.sum(np.diff((obs>=tau).astype(int))>0)),
                       omit_one_min_clipped_m=min(omitted), omit_one_max_clipped_m=max(omitted)))
    held = pd.DataFrame(held)
    scores = []
    for (model,criterion), g in held.groupby(["model", "criterion"]):
        errors = (g.predicted-g.observed).to_numpy()
        for tau in THRESHOLDS:
            near = abs(g.observed.to_numpy()-tau)<=.05
            agrees = (g.predicted>=tau)==(g.observed>=tau)
            scores.append(dict(model=model, criterion=criterion, threshold=tau, n=len(g),
                rmse=np.sqrt(np.mean(errors**2)), agreements=int(agrees.sum()),
                false_pass=int(((g.predicted>=tau)&(g.observed<tau)).sum()),
                false_fail=int(((g.predicted<tau)&(g.observed>=tau)).sum()),
                n_near=near.sum(), near_mae=np.abs(errors[near]).mean() if near.any() else np.nan))
    old = pd.read_csv(BASE/"第六版实验/results/q2v6_held.csv").query(
        "model=='parametric' and criterion in ['all','mean']").sort_values(["criterion","call","held_distance_m"])
    ours = held[held.model=="v6_parametric"].sort_values(["criterion","call","held_distance_m"])
    np.testing.assert_allclose(old.predicted, ours.predicted, atol=1e-9)
    for name, table in (("parameters",parameters),("ranges",ranges),("nodes",nodes),("tuning",tuning),
                        ("held",held),("scores",scores)):
        save("q2v7_"+name, table)


def acoustic_views(data):
    # A common pitch displacement leaves every frequency ratio and quartile order unchanged.
    frequency = ["F01", "Fpeak", "Q25", "Q50", "Q75"]
    views = {}
    for semitones in (0., -.5, .5, -1., 1.):
        shifted = data.copy()
        shifted[frequency] = data[frequency] * 2**(semitones/12)
        all_features = classical.extract(shifted)["acoustic"]
        views[semitones] = {
            "common": all_features[:, :6],
            "harmonic": all_features[:, :8],
            "shape": all_features[:, [0,1,2,3,4,5,8,9,10,11]],
            "both": all_features,
        }
    return views


def tree_prediction(views, y, train, test, representation, leaf, shift):
    replicas = [0., -shift, shift] if shift else [0.]
    x = np.concatenate([views[s][representation][train] for s in replicas])
    labels = np.tile(y[train], len(replicas))
    tree = ExtraTreesClassifier(n_estimators=128, max_features=1., n_jobs=1,
                                random_state=classical.SEED, min_samples_leaf=leaf*len(replicas))
    tree.fit(x, labels)
    return .98*tree.predict_proba(views[0.][representation][test]) + .02*classical.class_prior(y[train])


def labels_margin(p, priors, rule):
    scores = p / priors if rule=="balanced" else p.copy()
    scores = scores / scores.sum(axis=1, keepdims=True)
    order = np.sort(scores, axis=1)
    return scores.argmax(axis=1), order[:, -1]-order[:, -2]


def class_metrics(truth, decision, probabilities):
    counts = np.bincount(truth, minlength=3)
    hits = np.bincount(truth[truth==decision], minlength=3)
    recall = np.divide(hits,counts,out=np.full(3,np.nan),where=counts>0)
    return dict(n=len(truth), accuracy=(truth==decision).mean(), balanced_accuracy=recall.mean(),
                log_loss=-np.log(probabilities[np.arange(len(truth)),truth]).mean(),
                flat_recall=recall[0], mod_recall=recall[1], tv_recall=recall[2],
                tv_hits=hits[2], tv_true=counts[2], tv_predicted=int((decision==2).sum()))


def q1_fold(views, y, days, train, test, validation, held, old, predictions, tuning, thresholds):
    # The model and the review threshold see only training-date predictions in this outer fold.
    folds = list(GroupKFold(4).split(train, y[train], days[train]))
    priors = classical.class_prior(y[train])
    inner_priors = np.empty((len(train), 3))
    for sub,val in folds:
        assert not set(days[train[sub]]) & set(days[train[val]])
        inner_priors[val] = classical.class_prior(y[train[sub]])
    assert not set(days[train]) & set(days[test])
    @lru_cache(None)
    def inner(rep, leaf, shift):
        p = np.empty((len(train),3))
        for sub,val in folds:
            p[val] = tree_prediction(views,y,train[sub],train[val],rep,leaf,shift)
        return p
    @lru_cache(None)
    def outer(rep, leaf, shift):
        return tree_prediction(views,y,train,test,rep,leaf,shift)

    specs = { "fixed_"+rep:(rep,5,0.,"balanced") for rep in ("common","harmonic","shape","both") }
    specs.update({"fixed_shift_half":("both",5,.5,"balanced"),"fixed_shift_one":("both",5,1.,"balanced")})
    lookup=old.set_index("row").loc[test+1]
    assert len(set(lookup.candidate))==1 and len(set(lookup.decision_rule))==1
    _, features, size = lookup.iloc[0].candidate.split("_")
    specs["v6_tree"] = ("both" if features=="acoustic" else "common",int(size),0.,
                        lookup.iloc[0].decision_rule)
    candidates=[]
    for shift in (0.,.5,1.):
        for rule in ("ordinary","balanced"):
            p=inner("both",5,shift)
            pred, _ = labels_margin(p, inner_priors, rule)
            metric = class_metrics(y[train],pred,p)
            candidates.append((-round(metric["balanced_accuracy"],12),metric["log_loss"],shift,rule))
            tuning.append(dict(validation=validation,held=held,shift=shift,rule=rule,**metric))
    for name, choices in (("inner_plain",[x for x in candidates if x[2]==0.]),("inner_aug",candidates)):
        _,_,shift,rule = min(choices)
        specs[name] = ("both",5,shift,rule)
    for method,(rep,leaf,shift,rule) in specs.items():
        prob = outer(rep,leaf,shift)
        d, margin = labels_margin(prob,priors,rule)
        if method=="v6_tree":
            np.testing.assert_array_equal(NAMES[d], lookup.decision)
            np.testing.assert_allclose(prob, lookup[classical.PROB].values, atol=2e-11, rtol=0)
        for i,p,chosen,m in zip(test,prob,d,margin):
            predictions.append(dict(validation=validation,held=held,row=int(i+1),date=days[i],
                method=method,view=rep,leaf=leaf,shift=shift,rule=rule,actual=NAMES[y[i]],
                decision=NAMES[chosen],margin=m,**dict(zip(classical.PROB,p))))
        if method not in ("v6_tree","inner_aug","inner_plain"):
            continue
        _, calibrating = labels_margin(inner(rep,leaf,shift),inner_priors,rule)
        for coverage in COVERAGES:
            cut=0. if coverage==1. else np.quantile(calibrating,1.-coverage)
            thresholds.append(dict(validation=validation,held=held,method=method,target_coverage=coverage,
                          margin_threshold=cut,training_n=len(train),train_fraction_kept=np.mean(calibrating>=cut)))


def paired_class_scores(oof, reference="v6_tree"):
    rng = np.random.default_rng(20261127)
    rows = []
    for validation, data in oof.groupby("validation"):
        ref=data[data.method==reference].sort_values("row")
        day, dates = pd.factorize(ref.date, sort=True)
        units = len(dates)
        draw = rng.multinomial(units,np.full(units,1/units),size=5000)
        y=pd.Categorical(ref.actual,categories=CALLS).codes
        old_hit=(ref.decision==ref.actual).to_numpy()
        count=np.zeros((units,3))
        np.add.at(count,(day,y),1)
        denominator=draw@count
        good=(denominator>0).all(axis=1)
        for method,model in data.groupby("method"):
            model=model.sort_values("row")
            np.testing.assert_array_equal(ref.row,model.row)
            difference=(model.decision==model.actual).to_numpy().astype(int)-old_hit.astype(int)
            change=np.zeros_like(count)
            np.add.at(change,(day,y),difference)
            observed=(change.sum(axis=0)/count.sum(axis=0)).mean()
            delta=((draw[good]@change)/denominator[good]).mean(axis=1)
            low, high=np.quantile(delta,[.025,.975])
            leave=((change.sum(axis=0)-change)/(count.sum(axis=0)-count)).mean(axis=1)
            rows.append(dict(validation=validation,method=method,reference=reference,
                  delta_ba=observed,lower=low,upper=high,bootstrap_valid=len(delta),
                  bootstrap_draws=5000,boot_below_minus_3pp=(delta<-.03).mean(),
                  boot_within_3pp=(np.abs(delta)<=.03).mean(),boot_above_plus_3pp=(delta>.03).mean(),
                  leave_date_out_positive=int((leave>0).sum()),dates=units))
    return rows


def matched_calendar(views, y, days, oof):
    """Match earlier-year training size, call counts and number of dates on identical 2022 tests."""
    generator = np.random.default_rng(11272026)
    old = np.array([d.startswith("2021") for d in days])
    target_counts = np.bincount(y[old], minlength=3)
    number_dates = len(np.unique(days[old]))
    held_dates = np.unique(days[~old])
    assert (old.sum(), number_dates) == (27, 6)
    forward = oof.query("validation=='year_forward' and method=='fixed_both'").sort_values("row")
    future = np.flatnonzero(~old)
    actual = y[future]
    np.testing.assert_array_equal(forward.row, future + 1)
    baseline = class_metrics(actual, pd.Categorical(forward.decision,categories=CALLS).codes,
                             forward[classical.PROB].to_numpy())
    records, splits = [], []
    for rep in range(20):
        probabilities = np.zeros((len(y), 3))
        for held in held_dates:
            other = held_dates[held_dates!=held]
            for attempt in range(1000):
                subset = generator.choice(other, number_dates, replace=False)
                available = np.flatnonzero(np.isin(days,subset))
                groups = [available[y[available]==c] for c in range(3)]
                if any(len(group)<size for group,size in zip(groups,target_counts)):
                    continue
                train = np.sort(np.concatenate([generator.choice(group, size, replace=False)
                                              for group,size in zip(groups,target_counts)]))
                if len(set(days[train])) == number_dates:
                    break
            else:
                raise RuntimeError(f"Could not construct matched training set for {held}")
            test = np.flatnonzero(days==held)
            probabilities[test] = tree_prediction(views,y,train,test,"both",5,0.)
            assert not set(test)&set(train)
            splits.extend(dict(repetition=rep+1,held=held,training_row=i+1,date=days[i],
                               call=CALLS[y[i]]) for i in train)
        pred,_=labels_margin(probabilities[future],target_counts/target_counts.sum(),"balanced")
        score=class_metrics(actual,pred,probabilities[future])
        records.append(dict(repetition=rep+1,train_rows=old.sum(),train_dates=number_dates,
                            forward_ba=baseline["balanced_accuracy"],
                            within_minus_forward=score["balanced_accuracy"]-baseline["balanced_accuracy"],
                            **score))
    save("q1v7_matched_calendar",records)
    save("q1v7_matched_training",splits)


def run_q1(data):
    views = acoustic_views(data)
    y = pd.Categorical(data.CallType,categories=CALLS).codes.astype(int)
    dates = data.session_date.dt.strftime("%Y-%m-%d").to_numpy()
    idx=np.arange(len(data))
    predictions,tuning,thresholds=[],[],[]
    v5=BASE/"第五版实验/results"
    for validation,fname in (("date_holdout","q1c_oof.csv"),("year_forward","q1c_temporal_oof.csv")):
        old=pd.read_csv(v5/fname).query("method=='classical_nested'")
        splits = [(d,idx[dates!=d],idx[dates==d]) for d in np.unique(dates)]
        if validation=="year_forward":
            prior=np.array([d.startswith("2021") for d in dates])
            splits=[("2022",idx[prior],idx[~prior])]
        for d,train,test in splits:
            q1_fold(views,y,dates,train,test,validation,d,old,predictions,tuning,thresholds)
    oof=pd.DataFrame(predictions)
    cuts=pd.DataFrame(thresholds)
    scores,selective,confusion=[],[],[]
    for (validation,method),part in oof.groupby(["validation","method"]):
        truth=pd.Categorical(part.actual,categories=CALLS).codes.astype(int)
        decision=pd.Categorical(part.decision,categories=CALLS).codes.astype(int)
        scores.append(dict(validation=validation,method=method,
                           **class_metrics(truth,decision,part[classical.PROB].values)))
        for i,name in enumerate(CALLS):
            for j,guess in enumerate(CALLS):
                confusion.append(dict(validation=validation,method=method,actual=name,decision=guess,
                                      count=int(((truth==i)&(decision==j)).sum())))
        c=cuts.query("validation==@validation and method==@method")
        for target,table in c.groupby("target_coverage"):
            cut=part.held.map(table.set_index("held").margin_threshold)
            assert cut.notna().all()
            keep=(part.margin>=cut).to_numpy(); n=int(keep.sum())
            hits=int((truth[keep]==decision[keep]).sum())
            selective.append(dict(validation=validation,method=method,target_coverage=target,total=len(part),
                        accepted=n,deferred=len(part)-n,coverage=n/len(part),correct_accepted=hits,
                        accuracy_accepted=hits/n if n else np.nan,
                        error_accepted=(n-hits)/n if n else np.nan,
                        tv_accepted=int((truth[keep]==2).sum()),
                        tv_correct_accepted=int(((truth[keep]==2)&(decision[keep]==2)).sum())))
    matched_calendar(views,y,dates,oof)
    comparisons=[]
    for reference,methods in [("fixed_common",("fixed_harmonic","fixed_shape","fixed_both")),
                              ("fixed_both",("fixed_shift_half","fixed_shift_one")),
                              ("inner_plain",("inner_aug",))]:
        subset=oof[oof.method.isin([reference,*methods])]
        comparisons.extend(row for row in paired_class_scores(subset,reference) if row["method"]!=reference)
    save("q1v7_contrasts",comparisons)
    for name,table in (("oof",oof),("tuning",tuning),("thresholds",cuts),("scores",scores),
                       ("confusion",confusion),("selective",selective),("paired",paired_class_scores(oof))):
        save("q1v7_"+name,table)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question",choices=["1","2","both"],default="both")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    calls,_,correlations=read_inputs()
    if args.question in ("2","both"):
        run_q2(correlations)
    if args.question in ("1","both"):
        run_q1(calls)
