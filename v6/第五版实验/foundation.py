"""Fixed dual-view acoustic classifier with a class-balanced decision rule."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from penguin_data import CALLS, FEATURES, SEED, input_path

INPUTS = HERE / "inputs"
ARCHIVE = INPUTS / "q1f_held_probabilities.csv"
SIGNATURE = INPUTS / "q1f_inputs.json"
V4 = ROOT / "第二版实验/results/q1_oof.csv"
PROTOCOL = INPUTS / "q1f_protocol.json"
N_DRAWS = 5000


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observations():
    frame = pd.read_csv(input_path(1), usecols=["Date", "CallType", *FEATURES])
    x = frame[list(FEATURES)].to_numpy(dtype=float)
    y = pd.Categorical(frame.CallType, categories=CALLS).codes.astype(int)
    dates = pd.to_datetime(frame.Date, format="%d/%m/%Y").dt.strftime("%Y-%m-%d").to_numpy()
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(y < 0):
        raise ValueError("Expect positive acoustic measurements and known call classes")
    return x, y, dates


def representations(x):
    _, pitch, peak, low, middle, high = x.T
    width = high - low
    if np.any(width <= 0):
        raise ValueError("IQR must be positive to compute these measurements")
    peak_ratio = peak/pitch
    columns = [np.log2(peak_ratio), np.log2(middle/pitch), width/middle,
               np.abs(peak_ratio - np.maximum(1., np.rint(peak_ratio))),
               (high + low - 2*middle)/width, (peak - middle)/width]
    return {"raw": x, "augmented": np.column_stack((x,*columns))}


def actual_forward(cache, target, smoke):
    sys.path.insert(0, str(ROOT/"第二版实验"))
    import torch
    from frozen_models import ASSETS, TAB_FILE, ensure_asset, tabicl
    cache = cache.resolve()
    weight = ensure_asset(cache, TAB_FILE)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    x, y, dates = observations()
    xs = representations(x)
    chunks = []
    for fold, date in enumerate(np.unique(dates), 1):
        training = dates != date
        testing = ~training
        assert len(np.unique(y[training])) == len(CALLS)
        frame = pd.DataFrame({"row": np.flatnonzero(testing)+1, "held_date":date,
                              "fold":fold, "actual":y[testing]})
        for view, values in xs.items():
            p = tabicl(cache, values[training], y[training], values[testing])
            np.testing.assert_allclose(p.sum(1),1,atol=2e-6)
            assert np.all((p >= 0) & (p <= 1))
            for k,c in enumerate(CALLS):
                frame[f"{view}_{c}"] = p[:,k].astype(float)
        chunks.append(frame)
        if smoke:
            break
    output = pd.concat(chunks).sort_values("row")
    target.parent.mkdir(exist_ok=True, parents=True)
    output.to_csv(target,index=False,float_format="%.12g")
    if not smoke:
        sig = {"raw_input_sha256":sha(input_path(1)), "protocol_sha256":sha(PROTOCOL),
               "probability_sha256":sha(target), "v4_reference_sha256":sha(V4),
               "weight_file":weight.name,"weight_sha256":ASSETS[TAB_FILE][1],
               "versions":{k:version(k) for k in ("numpy","pandas","torch","tabicl","scikit-learn")}}
        target.with_name("q1f_inputs.json").write_text(json.dumps(sig,indent=2)+"\n")
    print("Forward predictions", len(output), "rows,",len(chunks),"held-out dates.")


def weighted_score(y, p, predicted, weights):
    correct = np.column_stack([weights @ ((y==k)&(predicted==k)) for k in range(len(CALLS))])
    actual = np.column_stack([weights @ (y==k) for k in range(len(CALLS))])
    totals = weights.sum(1)
    with np.errstate(divide="ignore",invalid="ignore"):
        recalls = correct/actual
    losses = -np.log(np.maximum(1e-15,p[np.arange(len(y)),y]))
    return {"balanced_accuracy":recalls.mean(1), "accuracy":correct.sum(1)/totals,
            "log_loss":weights@losses/totals, **{f"recall_{c}":recalls[:,k] for k,c in enumerate(CALLS)}}


def evaluate(output):
    data = pd.read_csv(ARCHIVE).sort_values("row")
    _, y, dates = observations()
    sig = json.loads(SIGNATURE.read_text())
    for key,path in [("raw_input_sha256",input_path(1)),("protocol_sha256",PROTOCOL),
                     ("probability_sha256",ARCHIVE),("v4_reference_sha256",V4)]:
        if sha(path) != sig[key]:
            raise ValueError(f"{path.name} differs from its frozen reference")
    np.testing.assert_array_equal(data.row,np.arange(1,len(y)+1))
    np.testing.assert_array_equal(data.held_date,dates)
    np.testing.assert_array_equal(data.actual,y)
    groups, inverse = np.unique(dates,return_inverse=True)
    np.testing.assert_array_equal(data.fold,inverse+1)
    prior = np.zeros((len(y),len(CALLS)))
    for d in groups:
        counts = np.bincount(y[dates != d],minlength=len(CALLS))
        prior[dates == d] = counts/counts.sum()
    raw = data[[f"raw_{c}" for c in CALLS]].to_numpy()
    extended = data[[f"augmented_{c}" for c in CALLS]].to_numpy()
    v4 = pd.read_csv(V4).sort_values("row")
    np.testing.assert_array_equal(v4.actual,y)
    np.testing.assert_array_equal(v4.group,dates)
    np.testing.assert_allclose(raw,v4[[f"TabICLv2_{k}" for k in range(len(CALLS))]],rtol=2e-6,atol=1e-7)
    rng = np.random.default_rng(SEED+501)
    draw = rng.multinomial(len(groups), np.full(len(groups),1/len(groups)), size=N_DRAWS)
    weights = np.vstack((np.ones(len(y)),draw[:,inverse]))
    rows,comparison,confusion,by_day,all_scores=[],[],[],[],{}
    for view,p in [("raw",raw),("augmented",extended),("blend",.5*(raw+extended))]:
        for rule,z in [("empirical",p),("balanced",p/prior)]:
            key=f"{view}_{rule}"
            pred=z.argmax(1)
            all_scores[key]=weighted_score(y,p,pred,weights)
            for t in range(len(y)):
                rows.append([key,t+1,dates[t],inverse[t]+1,CALLS[y[t]],CALLS[pred[t]],*p[t],*prior[t]])
            for j in range(len(CALLS)):
                for k in range(len(CALLS)):
                    confusion.append([key,CALLS[j],CALLS[k],int(((y==j)&(pred==k)).sum())])
            for date in groups:
                use=dates==date
                by_day.append([key,date,use.sum(),(pred[use]==y[use]).sum(),-np.log(p[use,y[use]]).mean()])
    for method,scores in all_scores.items():
        for ref in ("raw_empirical","raw_balanced"):
            for metric,draws in scores.items():
                delta=draws-all_scores[ref][metric]
                diffs=delta[1:][np.isfinite(delta[1:])]
                lo,hi=np.quantile(diffs,[.025,.975])
                comparison.append([method,ref,metric,draws[0],delta[0],lo,hi,len(diffs)])
    output.mkdir(exist_ok=True,parents=True)
    def write(name,values,cols):
        pd.DataFrame(values,columns=cols).to_csv(output/("q1f_"+name+".csv"),index=False,float_format="%.12g")
    write("scores",comparison,["method","comparator","metric","estimate","difference","lower95","upper95","valid_draws"])
    write("predictions",rows,["method","row","held_date","fold","actual","prediction",
                              *["p_"+c for c in CALLS],*["train_prior_"+c for c in CALLS]])
    write("confusion",confusion,["method","actual","predicted","count"])
    write("by_date",by_day,["method","held_date","rows","correct","mean_log_loss"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward",action="store_true",help="Recompute the archived OOF probabilities from frozen weights")
    parser.add_argument("--cache",type=Path,default=HERE/".model_cache")
    parser.add_argument("--forward-output",type=Path,help="Write independent forward artifacts to this directory")
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true",help="Run one date using the two frozen views")
    args=parser.parse_args()
    if args.forward or args.smoke:
        folder=args.forward_output if args.forward_output else (args.output/"smoke" if args.smoke else INPUTS)
        actual_forward(args.cache,folder/ARCHIVE.name,args.smoke)
        if args.smoke:
            return
        if folder.resolve()!=INPUTS.resolve():
            fresh=pd.read_csv(folder/ARCHIVE.name)
            saved=pd.read_csv(ARCHIVE)
            assert fresh[["row","fold","actual","held_date"]].equals(saved[["row","fold","actual","held_date"]])
            np.testing.assert_allclose(fresh.iloc[:,4:],saved.iloc[:,4:],rtol=2e-6,atol=1e-7)
    evaluate(args.output)


if __name__ == "__main__":
    main()
