"""Group responses to playback. Align observed onsets, full-window flags and score targets."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "第三版实验"))
from analyze_playback import SCORE_TIMES, fit_timing, select_penalty, representative_design
from models_playback import LAMBDAS, FEATURE_NAMES, design, response_fit, response_probs, time_fit, time_cdf
from penguin_data import (BEHAVIOURS, CONTEXTS, HORIZON, REPRESENTATIVE_N, SEED, STATE_NAMES,
                          STATES, STIMULI, context_weights, event_flags, input_path, latency_values, state_index)
from timing import fit_cox, risk_by_time

PRIMARY = "recorded_onset"
NET = "net_response"
EXPANDED = "approach_latency_inclusive"
MODELS = ("two_part", "two_part_zero_mass", "cox")
B_BOOT = 5000

def inputs():
    data = pd.read_csv(input_path(2))
    paired = data.groupby("expID").Type.value_counts().unstack()
    assert (paired[["call","control"]] == 1).all().all()
    track = data.loc[data.Type.eq("call")].set_index("expID").PlaybackTrack
    data["pair_track"] = data.expID.map(track)
    assert data.pair_track.notna().all()
    x = design(data.CallType, data.Context, data.Distance, data.Npenguins)
    time = latency_values(data)
    net = event_flags(data)
    before, after = data.VocalNcallsBefore.to_numpy(), data.VocalNcalls.to_numpy()
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert np.array_equal(net[:,1], after>before)
    assert np.array_equal(data.VocalRelNCalls, np.maximum(after-before,0))
    primary = net.copy()
    primary[:,1] = after > 0
    expanded = primary.copy()
    expanded[:,2] |= np.isfinite(time[:,2])
    assert np.array_equal(primary[:,0], np.isfinite(time[:,0]))
    assert np.array_equal(primary[:,1], np.isfinite(time[:,1]))
    assert np.array_equal(net[:,2], data.ApproachDuration.gt(0))
    assert np.array_equal(net[:,2], data.ApproachRelDist.gt(0))
    for f in [primary, net, expanded]:
        assert np.isfinite(time[f]).all() and ((time[f]>=0)&(time[f]<=HORIZON)).all()
    return data, x, time, {PRIMARY:primary, NET:net, EXPANDED:expanded}

def observations(d, endpoints, time):
    net = endpoints[NET]
    audit = d[["expID","Type","CallType","Distance","Npenguins","HeadsUpYesNo","HeadsUpLatency","HeadsUpDuration",
               "VocalNcallsBefore","VocalNcalls","VocalRelNCalls","VocalLatency","VocalYesNo",
               "ApproachYesNo","ApproachLatency","ApproachDuration","ApproachRelDist"]].copy()
    audit.insert(0, "source_csv_line", np.arange(len(d))+2)
    audit["any_post_call"] = endpoints[PRIMARY][:,1].astype(int)
    audit["approach_time_only"] = ~net[:,2] & np.isfinite(time[:,2])
    audit["heads_duration_beyond_window"] = d.HeadsUpLatency+d.HeadsUpDuration > HORIZON
    audit["approach_duration_beyond_window"] = d.ApproachLatency+d.ApproachDuration > HORIZON
    # The threshold-crossing event is the (before+1)-th call. No later per-call time is supplied.
    exact = net[:,1] & d.VocalNcallsBefore.eq(0).to_numpy()
    interval = net[:,1] & ~exact
    audit["net_cross_observation"] = np.select([exact,interval],["exact","interval"],default="right_censored")
    audit["net_cross_lower_s"] = np.where(net[:,1],time[:,1],HORIZON)
    audit["net_cross_upper_s"] = np.where(exact,time[:,1],np.where(interval,HORIZON,np.nan))
    counts, bounds, intensities = [], [], []
    for typ, mask in [("all",np.ones(len(d),bool)),("natural",d.Type.eq("call").to_numpy()),
                      *[(s,d.CallType.eq(s).to_numpy()) for s in STIMULI]]:
        counts.append((typ,mask.sum(),*net[mask].sum(axis=0),int(endpoints[PRIMARY][mask,1].sum()),
                       (mask&exact).sum(),(mask&interval).sum(),(mask&~net[:,1]).sum(),
                       (mask&audit.approach_time_only).sum()))
        for t in [0,*SCORE_TIMES]:
            definite = (exact & (time[:,1]<=t)) | (interval & (HORIZON<=t))
            possible = net[:,1] & (time[:,1]<=t)
            bounds.append((typ,t,mask.sum(),(mask&definite).sum(),(mask&possible).sum(),
                           definite[mask].mean(),possible[mask].mean(),
                           (endpoints[PRIMARY][mask,1] & (time[mask,1]<=t)).mean()))
        for field, e in [("HeadsUpDuration",net[:,0]),("ApproachDuration",net[:,2]),
                         ("VocalNcalls",endpoints[PRIMARY][:,1]),("VocalRelNCalls",net[:,1])]:
            use = mask & e
            v = d.loc[use,field]
            intensities.append((typ,field,"s" if "Duration" in field else "count",int(use.sum()),v.sum(),
                                v.quantile(.25),v.median(),v.quantile(.75)))
    return {"event_audit":audit,
            "endpoint_counts":pd.DataFrame(counts,columns=["stimulus","trials","heads_up","net_increase","confirmed_approach",
                  "any_post_call","cross_exact","cross_interval","cross_right_censored","uncorroborated_approach_time"]),
            "net_cross_bounds":pd.DataFrame(bounds,columns=["stimulus","time_s","trials","definite","possible",
                 "empirical_cdf_lower","empirical_cdf_upper","any_call_empirical_cdf"]),
            "intensities":pd.DataFrame(intensities,columns=["stimulus","field","unit","events","sum","q25","median","q75"])}

def time_models(x, flags, time):
    ordinary = fit_timing(x, flags, time)
    h = flags[:,0]
    zeros = time[h,0]==0
    assert h.any() and (~zeros).any()
    # Jeffreys smoothing for P(recorded immediately | a head lift occurs in the window).
    mass = (zeros.sum()+.5)/(len(zeros)+1)
    positive = time_fit(x[h][~zeros],time[h,0][~zeros])
    return ordinary, positive, mass

def fitted_curves(x, marginal, flags, time, train, test):
    ordinary, positive, mass = time_models(x[train],flags[train],time[train])
    rows = []
    for j,b in enumerate(BEHAVIOURS):
        u = x[test,None,:]
        cp = time_cdf(ordinary[j],u,SCORE_TIMES)
        alt = mass+(1-mass)*time_cdf(positive,u,SCORE_TIMES) if j==0 else cp
        cox = fit_cox(x[train,1:],flags[train,j],time[train,j])
        c = risk_by_time(cox,x[test,1:],SCORE_TIMES)
        risk = [marginal[:,j,None]*cp,marginal[:,j,None]*alt,c]
        assert all(np.isfinite(v).all() and (v>=0).all() and (v<=1+1e-12).all() for v in risk)
        np.testing.assert_allclose(risk[1][:,-1],marginal[:,j],atol=1e-10)
        for k,t in enumerate(SCORE_TIMES):
            rows.append(pd.DataFrame({"i":test,"behaviour":b,"time_s":t,
                        "observed":(flags[test,j]&(time[test,j]<=t)).astype(int),
                        **{name:a[:,k] for name,a in zip(MODELS,risk)}}))
    return rows

def validation(d, x, time, endpoints, smoke=False):
    groups = d.expID.to_numpy()
    tracks = d.pair_track.to_numpy()
    rows, timing_rows, tuning = [],[],[]
    designs = [(PRIMARY,"paired",groups,GroupKFold(5)),
               (NET,"paired",groups,GroupKFold(5)),
               (NET,"natural_track",tracks,LeaveOneGroupOut())]
    for endpoint,unit,unit_ids,splitter in designs:
        flags = endpoints[endpoint]
        y = state_index(flags)
        for fold,(train,test) in enumerate(splitter.split(x,y,unit_ids),1):
            if set(groups[train]) & set(groups[test]) or set(unit_ids[train]) & set(unit_ids[test]):
                raise AssertionError("A held-out pair or group was split across fit/test")
            penalty, path = select_penalty(x[train],y[train],unit_ids[train],4,True)
            tuning.extend((endpoint,unit,fold,len(np.unique(unit_ids[train])),len(np.unique(unit_ids[test])),
                           value,score,value==penalty) for value,score in zip(LAMBDAS,path))
            beta = response_fit(x[train],y[train],penalty)
            p = response_probs(beta,x[test])
            np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
            rows.append(pd.DataFrame({"endpoint":endpoint,"validation_unit":unit,"fold":fold,
                "i":test,"expID":groups[test],"pair_track":tracks[test],"stimulus":d.CallType.to_numpy()[test],
                "state":y[test],"lambda":penalty,**{f"p_{v}":p[:,i] for i,v in enumerate(STATE_NAMES)}}))
            if endpoint == PRIMARY:
                timing_rows.extend(t.assign(fold=fold) for t in fitted_curves(x,p@STATES,flags,time,train,test))
            if smoke:
                break
    state = pd.concat(rows,ignore_index=True).sort_values(["endpoint","validation_unit","i"])
    state["source_csv_line"] = state.pop("i")+2
    timing = pd.concat(timing_rows,ignore_index=True).sort_values(["i","behaviour","time_s"])
    timing["expID"] = groups[timing.i]
    timing["source_csv_line"] = timing.pop("i")+2
    params = pd.DataFrame(tuning,columns=["endpoint","validation_unit","outer_fold","inner_training_groups",
                 "outer_test_groups","lambda","inner_oof_log_loss","selected"])
    if not smoke:
        assert state.groupby(["endpoint","validation_unit"]).source_csv_line.nunique().eq(len(d)).all()
        assert len(timing) == len(d)*len(BEHAVIOURS)*len(SCORE_TIMES)
    return state,timing,params

def scores(state,timing):
    joint = []
    for (goal,key),tab in state.groupby(["endpoint","validation_unit"]):
        for subset,use in [("all",tab),("natural",tab[tab.stimulus!="control"]),("control",tab[tab.stimulus=="control"])]:
            y = use.state.to_numpy()
            p = use[[f"p_{s}" for s in STATE_NAMES]].to_numpy()
            joint.append((goal,key,subset,len(use),use.fold.nunique(),-np.log(p[np.arange(len(p)),y]).mean(),
                          np.mean(np.sum((np.eye(8)[y]-p)**2,axis=1)),np.mean(((1-p[:,0])-(y>0))**2)))
    values = timing.copy()
    for m in MODELS:
        values[m] = (values[m]-values.observed)**2
    by = values.groupby(["expID","behaviour","time_s"])[list(MODELS)].mean()
    r = np.random.default_rng(SEED+303)
    pairs = np.unique(timing.expID)
    w = r.multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=B_BOOT)/len(pairs)
    output = []
    for target,time in [(b,t) for b in BEHAVIOURS for t in SCORE_TIMES]+[(b,0) for b in BEHAVIOURS]+[("all_grid_mean",0)]:
        subset = by
        if target != "all_grid_mean":
            subset = subset[subset.index.get_level_values("behaviour")==target]
        if time:
            subset = subset[subset.index.get_level_values("time_s")==time]
        array = subset.groupby(level="expID").mean().reindex(pairs).to_numpy()
        mean = array.mean(axis=0)
        for j,k in [(2,0),(1,0),(2,1)]:
            delta = array[:,j]-array[:,k]
            lo,hi = np.quantile(w@delta,[.025,.975])
            output.append((target,time or np.nan,MODELS[j],MODELS[k],mean[j],mean[k],delta.mean(),lo,hi,len(pairs)))
    return {"state_scores":pd.DataFrame(joint,columns=["endpoint","validation_unit","subset","rows","folds","log_loss","multiclass_brier","any_brier"]),
            "timing_scores":pd.DataFrame(output,columns=["behaviour","time_s","model","reference","model_brier","reference_brier","delta","lower","upper","pairs"])}

def grid_results(d, x, time, endpoints):
    stim,dist,grid = representative_design()
    weights = context_weights(d)
    rows,shifts,betas,params = [],[],[],[]
    for name,flags in endpoints.items():
        y = state_index(flags)
        strength,_ = select_penalty(x,y,d.expID.to_numpy(),5,True)
        beta = response_fit(x,y,strength)
        betas.extend((name,i,b,strength) for i,b in enumerate(beta))
        ordinary,positive,mass = time_models(x,flags,time)
        for j,b in enumerate(BEHAVIOURS):
            params.extend((name,b,"continuous",f,v) for f,v in zip((*FEATURE_NAMES,"log_sigma"),ordinary[j]))
        params.extend((name,"HeadsUp","positive_component",f,v) for f,v in zip((*FEATURE_NAMES,"log_sigma"),positive))
        params.append((name,"HeadsUp","zero_mass","conditional_probability",mass))
        q = response_probs(beta,grid)
        m = (q@STATES)*weights[None,:,None]
        for i,(s,z) in enumerate(zip(stim,dist)):
            match = d.CallType.eq(s).to_numpy() & d.Distance.eq(z).to_numpy()
            for j,b in enumerate(BEHAVIOURS):
                n,k = int(match.sum()),int(flags[match,j].sum())
                p = m[i,:,j].sum()
                def cdf(t):
                    return m[i,:,j] @ time_cdf(ordinary[j],grid[i],t)
                def quantile(target):
                    if target>p+1e-10:
                        return np.nan
                    lo,hi=0.,HORIZON
                    for _ in range(38):
                        mid = (lo+hi)/2
                        if cdf(mid)<target: lo = mid
                        else: hi=mid
                    return .5*(lo+hi)
                median = quantile(.5*p)
                status = "no_trials" if n==0 else ("no_observed_event" if k==0 else "event_observed")
                rows.append((name,s,z,REPRESENTATIVE_N,b,n,k,status,p,
                            median if k else np.nan,median,quantile(.5)))
        if name in [NET,PRIMARY]:
            for ctx in CONTEXTS:
                probs = response_probs(beta,design(np.array(STIMULI),ctx,100,REPRESENTATIVE_N))
                for j,s in enumerate(STIMULI):
                    for i,state in enumerate(STATE_NAMES):
                        observed = d.Context.eq(ctx)&d.CallType.eq(s)&d.Distance.eq(100)
                        shifts.append((name,ctx,s,100,REPRESENTATIVE_N,state,probs[j,i],probs[0,i],probs[j,i]-probs[0,i],
                                       int(observed.sum()), int((observed & (y==i)).sum())))
    return {"scenarios":pd.DataFrame(rows,columns=["endpoint","stimulus","distance_m","group_size","behaviour","exact_trials",
                       "exact_events","support_status","p60","supported_conditional_median_s","conditional_projection_s","unconditional_median_s"]),
            "state_shifts":pd.DataFrame(shifts,columns=["endpoint","context","stimulus","distance_m","group_size","state",
                       "p_state","p_control","adjusted_difference","trials_at_exact_context_distance","observed_state_count"]),
            "state_parameters":pd.DataFrame(betas,columns=["endpoint","index","value","penalty"]),
            "time_parameters":pd.DataFrame(params,columns=["endpoint","behaviour","component","parameter","value"])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    d,x,time,endpoints = inputs()
    tables = observations(d,endpoints,time)
    if args.smoke:
        state,timing,params = validation(d,x,time,endpoints,True)
        assert state.shape[0]>0 and timing.shape[0]>0 and params.selected.any()
        print(tables["endpoint_counts"].to_string(index=False))
        return
    state,timing,params = validation(d,x,time,endpoints)
    tables.update(state_oof=state,timing_oof=timing,tuning=params)
    tables.update(scores(state,timing))
    tables.update(grid_results(d,x,time,endpoints))
    args.output.mkdir(exist_ok=True,parents=True)
    for title,table in tables.items():
        table.to_csv(args.output/f"q3v4_{title}.csv",index=False,float_format="%.12g")
    from behavior_intervals import intervals
    extra = intervals(d,x,time,endpoints,tables["scenarios"],tables["state_parameters"])
    for title,table in extra.items():
        table.to_csv(args.output/f"q3v4_{title}.csv",index=False,float_format="%.12g")
    print(tables["state_scores"].to_string(index=False))
    print(tables["timing_scores"].query('behaviour=="all_grid_mean"').to_string(index=False))

if __name__ == "__main__":
    main()
