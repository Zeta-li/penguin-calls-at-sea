"""Exploratory single forward-year temporal transport of the fixed acoustic methods."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from foundation import HERE, INPUTS, PROTOCOL, ROOT, observations, representations, sha
from penguin_data import CALLS, SEED, input_path

ARCHIVE = INPUTS / "q1t_held_probabilities.csv"


def cohort():
    x, y, days = observations()
    years = np.array([int(d[:4]) for d in days])
    if np.unique(years).size != 2:
        raise ValueError("Temporal protocol requires two sampling years")
    train = years == years.min()
    test = ~train
    if set(np.unique(y[train])) != {0,1,2} or set(np.unique(y[test])) != {0,1,2}:
        raise ValueError("Both sampling years must have all call types")
    assert days[train].max() < days[test].min()
    return x, y, days, train, test


def forward(cache, target):
    sys.path.insert(0,str(ROOT/"第二版实验"))
    import torch
    from frozen_models import TAB_FILE, ensure_asset, tabicl
    ensure_asset(cache,TAB_FILE)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    x,y,days,train,test = cohort()
    frame = pd.DataFrame({"row":np.flatnonzero(test)+1,"date":days[test],"actual":y[test]})
    for name,values in representations(x).items():
        probs = tabicl(cache,values[train],y[train],values[test])
        np.testing.assert_allclose(probs.sum(1),1.,atol=2e-6)
        for k,c in enumerate(CALLS):
            frame[name+"_"+c] = probs[:,k].astype(float)
    target.parent.mkdir(exist_ok=True,parents=True)
    frame.to_csv(target,index=False,float_format="%.12g")
    data = {"experiment":"frozen model; earlier-year train and later-year test; no model or cutoff selection",
            "raw_input_sha256":sha(input_path(1)),"q1_fixed_protocol_sha256":sha(PROTOCOL),
            "weight_sha256":sha(cache/TAB_FILE),"predictions_sha256":sha(target)}
    target.with_name("q1t_inputs.json").write_text(json.dumps(data,indent=2)+"\n")


def summarize(folder):
    _,y,days,train,test = cohort()
    data = pd.read_csv(ARCHIVE)
    sig = json.loads(ARCHIVE.with_name("q1t_inputs.json").read_text())
    assert sig["raw_input_sha256"]==sha(input_path(1))
    assert sig["predictions_sha256"]==sha(ARCHIVE)
    assert sig["q1_fixed_protocol_sha256"]==sha(PROTOCOL)
    np.testing.assert_array_equal(data.row,np.flatnonzero(test)+1)
    np.testing.assert_array_equal(data.date,days[test])
    np.testing.assert_array_equal(data.actual,y[test])
    counts=np.bincount(y[train],minlength=len(CALLS))
    priors=counts/counts.sum()
    raw=data[[f"raw_{c}" for c in CALLS]].to_numpy()
    aug=data[[f"augmented_{c}" for c in CALLS]].to_numpy()
    summary, preds=[], []
    for view,probs in [("raw",raw),("blend",.5*(raw+aug))]:
        for rule,decision in [("empirical",probs),("balanced",probs/priors)]:
            method=view+"_"+rule
            predicted=decision.argmax(1)
            recalls=[(predicted[y[test]==k]==k).mean() for k in range(len(CALLS))]
            logloss=-np.log(probs[np.arange(len(data)),y[test]]).mean()
            summary.append((method,len(data),np.mean(recalls),np.mean(predicted==y[test]),logloss,*recalls))
            for i in range(len(data)):
                preds.append((method,data.row[i],data.date[i],CALLS[y[test][i]],CALLS[predicted[i]],*probs[i],*priors))
    counts_frame=pd.DataFrame({"month":[d[:7] for d in days],"role":np.where(train,"train","test"),
                               "date":days,"label":np.array(CALLS)[y]})
    counts=(counts_frame.groupby(["role","month","label"]).size().unstack(fill_value=0).reset_index())
    counts["dates"]=counts_frame.groupby(["role","month"]).date.nunique().reindex(pd.MultiIndex.from_frame(counts[["role","month"]])).to_numpy()
    folder.mkdir(exist_ok=True,parents=True)
    counts.to_csv(folder/"q1t_cohorts.csv",index=False)
    pd.DataFrame(summary,columns=["method","rows","balanced_accuracy","accuracy","log_loss",*["recall_"+c for c in CALLS]]).to_csv(
        folder/"q1t_scores.csv",index=False,float_format="%.12g")
    pd.DataFrame(preds,columns=["method","row","date","actual","prediction",*["p_"+c for c in CALLS],
                                *["train_prior_"+c for c in CALLS]]).to_csv(folder/"q1t_predictions.csv",index=False,float_format="%.12g")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward",action="store_true")
    parser.add_argument("--cache",type=Path,default=HERE/".model_cache")
    parser.add_argument("--forward-output",type=Path,default=INPUTS)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args=parser.parse_args()
    if args.forward:
        target=args.forward_output/ARCHIVE.name
        forward(args.cache.resolve(),target)
        if target.resolve() != ARCHIVE.resolve():
            fresh,saved=pd.read_csv(target),pd.read_csv(ARCHIVE)
            assert fresh.iloc[:,:3].equals(saved.iloc[:,:3])
            np.testing.assert_allclose(fresh.iloc[:,3:],saved.iloc[:,3:],rtol=2e-6,atol=1e-7)
    summarize(args.output)
