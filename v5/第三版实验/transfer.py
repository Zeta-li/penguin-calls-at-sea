"""Carry alternative propagation shapes through the original behavioural scenarios."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from analyze_effectiveness import DESIGN, compute_radii, effective_point, probabilities
from models_acoustic import block_means, evaluate_nodes, monotone_nodes, smooth_fit, smooth_value
from models_effectiveness import BASE_WIDTH, SCENARIOS, ecological_values
from penguin_data import CALLS, DISTANCES, context_weights, read_inputs
from propagation import fidelity, fit_curves


def run(output, smoke=False):
    _, playback, recording = read_inputs()
    means = block_means(recording)[0].mean(axis=2)
    nodes = monotone_nodes(means).mean(axis=1)
    pava = np.array([evaluate_nodes(v,DESIGN) for v in nodes])
    spline = fidelity(fit_curves(means),DESIGN)
    params = [smooth_fit(DISTANCES, row) for row in means.mean(axis=1)]
    smooth = np.array([smooth_value(v,DESIGN) for v in params])
    prior = pd.read_csv(ROOT/"results/q3_response_parameters.csv")
    joint = probabilities(prior.value.to_numpy())
    weights = context_weights(playback)
    utilities = ecological_values()
    records = []
    for model, acoustic in (("pava",pava),("spline",spline),("parametric",smooth)):
        for sc in (SCENARIOS[:1] if smoke else SCENARIOS):
            p, score, _ = effective_point(acoustic,joint,weights,sc,utilities,BASE_WIDTH)
            for k, group in enumerate(compute_radii(score)):
                radius, status, disjoint = group[-1]
                records.append((model,sc[0],CALLS[k],radius,status,disjoint,p[k,-1,DESIGN==100].item()))
    columns = ["model","scenario","call","radius_m","status","later_segment","p100"]
    result = pd.DataFrame(records,columns=columns)
    ref = pd.read_csv(ROOT/"results/q4_scenario_results.csv").query("context == 'pooled'")
    check = result.query("model == 'pava'").merge(ref,on=["scenario","call"],validate="one_to_one")
    np.testing.assert_allclose(check.radius_m,check.functional_radius_m,rtol=0,atol=5e-6)
    np.testing.assert_allclose(check.p100,check.any_at_100m,rtol=0,atol=3e-9)
    assert check.status.eq(check.radius_status).all()
    base = result.query("scenario=='Baseline'").pivot(index="call",columns="model",values="p100")
    np.testing.assert_allclose(base.to_numpy(),np.tile(base.iloc[:,0:1].to_numpy(), (1,3)),atol=1e-12)
    if smoke:
        print(check[["call","radius_m","p100"]].to_string(index=False))
        return
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/"q4v3_shape_transfer.csv",index=False,float_format="%.12g")
    # Separate acoustic crossings of the parametric comparator, inside recorded support.
    rows = []
    for call, par in zip(CALLS,params):
        for tau in (.50,.60,.70):
            low,high=DISTANCES.min(),DISTANCES.max()
            if smooth_value(par,low)<tau:
                radius,status=low,"below_4m"
            elif smooth_value(par,high)>=tau:
                radius,status=high,"right_boundary"
            else:
                radius=brentq(lambda x: smooth_value(par,x)-tau,low,high)
                status="crossing"
            rows.append((call,tau,radius,status))
    pd.DataFrame(rows,columns=["call","threshold","radius_m","status"]).to_csv(
        output/"q2v3_parametric_ranges.csv",index=False,float_format="%.12g")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.output,args.smoke)
