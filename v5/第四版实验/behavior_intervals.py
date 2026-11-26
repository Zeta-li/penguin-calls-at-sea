"""Paired resampling of Q3 scenario estimates, holding selected penalty and context mix fixed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from analyze_playback import fit_timing, representative_design, standardized_timing
from models_playback import design, response_fit, response_probs
from penguin_data import BEHAVIOURS, CONTEXTS, REPRESENTATIVE_N, SEED, STATE_NAMES, STIMULI, context_weights, state_index

B_SCENARIO = 400

def intervals(data, x, time, endpoints, scenario, coefficients):
    generator = np.random.default_rng(SEED+417)
    groups = data.expID.to_numpy()
    ids = np.unique(groups)
    by_pair = np.array([np.flatnonzero(groups==g) for g in ids])
    sample_indices = generator.integers(len(ids),size=(B_SCENARIO,len(ids)))
    stimuli,distances,grid = representative_design()
    weights = context_weights(data)
    bins = [["000"],["100"],["010","110"],["001","011","101","111"]]
    names = ["none","heads_only","vocal_no_approach","any_approach"]
    transform = np.array([[s in group for group in bins] for s in STATE_NAMES],float)
    matrix = design(np.array(STIMULI)[None,:],np.array(CONTEXTS)[:,None],100,REPRESENTATIVE_N)
    row_summary,shift_summary = [],[]
    for endpoint in ("net_response","recorded_onset"):
        flags = endpoints[endpoint]
        y = state_index(flags)
        strength = coefficients.loc[coefficients.endpoint==endpoint,"penalty"].iloc[0]
        beta = coefficients[coefficients.endpoint.eq(endpoint)].sort_values("index").value.to_numpy()
        point = response_probs(beta,matrix)@transform
        point_diff = point-point[:,:1]
        n_conditions = len(grid)
        p_samples = np.empty((B_SCENARIO,n_conditions,len(BEHAVIOURS)))
        t_samples = np.empty_like(p_samples)
        shift_samples = np.empty((B_SCENARIO,len(CONTEXTS),len(STIMULI),len(bins)))
        for k,draw in enumerate(sample_indices):
            ix = by_pair[draw].ravel()
            fit = response_fit(x[ix], y[ix],strength)
            times = fit_timing(x[ix],flags[ix],time[ix])
            p_samples[k], t_samples[k] = standardized_timing(fit,times,grid,weights)
            prob = response_probs(fit,matrix)@transform
            shift_samples[k] = prob-prob[:,:1]
        assert np.isfinite(t_samples).all() and np.isfinite(p_samples).all()
        p_lo,p_hi = np.quantile(p_samples,[.025,.975],axis=0)
        t_lo,t_hi = np.quantile(t_samples,[.025,.975],axis=0)
        s_lo,s_hi = np.quantile(shift_samples,[.025,.975],axis=0)
        for k,(c,dist) in enumerate(zip(stimuli,distances)):
            for j,b in enumerate(BEHAVIOURS):
                row_summary.append((endpoint,c,dist,b,p_lo[k,j],p_hi[k,j],t_lo[k,j],t_hi[k,j],B_SCENARIO))
        for h,ctx in enumerate(CONTEXTS):
            for k,s in enumerate(STIMULI):
                for j,category in enumerate(names):
                    shift_summary.append((endpoint,ctx,s,100,REPRESENTATIVE_N,category,point[h,k,j],
                                          point[h,0,j],point_diff[h,k,j],s_lo[h,k,j],s_hi[h,k,j],B_SCENARIO))
    estimates = pd.DataFrame(row_summary,columns=["endpoint","stimulus","distance_m","behaviour",
                    "p60_lower","p60_upper","conditional_projection_lower_s","conditional_projection_upper_s","draws"])
    primary = scenario[scenario.endpoint.isin(["net_response","recorded_onset"])].merge(
        estimates,on=["endpoint","stimulus","distance_m","behaviour"],validate="one_to_one")
    return {"scenario_intervals":primary,
            "grouped_shifts":pd.DataFrame(shift_summary,columns=["endpoint","context","stimulus","distance_m","group_size","category",
                     "p_state","p_control","adjusted_difference","lower","upper","draws"])}
