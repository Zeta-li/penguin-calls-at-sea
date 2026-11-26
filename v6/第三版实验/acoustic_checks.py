"""Exact within-date rank inference and out-of-date confusion counts."""
import argparse
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from analyze_acoustics import holm
from penguin_data import CALLS, FEATURES, input_path


def block_distribution(values, first):
    """Distribution of 2U−nm given the number of each label in one date."""
    n = int(first.sum())
    m = len(first)-n
    doubled = np.rint(2*rankdata(values, method="average")).astype(int)
    sums = [Counter() for _ in range(n+1)]
    sums[0][0] = 1
    for i, rank in enumerate(doubled):
        for size in range(min(n, i+1), 0, -1):
            for total, count in sums[size-1].items():
                sums[size][total+int(rank)] += count
    center = n*(n+1)+n*m
    null = Counter({key-center:count for key,count in sums[n].items()})
    assert sum(null.values()) == math.comb(n+m,n)
    observed = int(doubled[first].sum()) - center
    return observed, null, n*m


def exact_pair(data, feature, a, b):
    pair = data[data.CallType.isin([a,b])]
    distribution = {0:1}
    observed = pairs = dates = n_a = n_b = 0
    for _, day in pair.groupby("Date", sort=True):
        first = day.CallType.eq(a).to_numpy()
        if first.all() or not first.any():
            continue
        statistic, null, comparable = block_distribution(day[feature].to_numpy(),first)
        combined = Counter()
        for old, count in distribution.items():
            for term, multiplicity in null.items():
                combined[old+term] += count*multiplicity
        distribution = combined
        observed += statistic
        dates += 1
        pairs += comparable
        n_a += sum(first)
        n_b += sum(~first)
    total = sum(distribution.values())
    extreme = sum(c for value,c in distribution.items() if abs(value)>=abs(observed))
    return (feature,a,b,dates,n_a,n_b,pairs,observed,observed/pairs,extreme/total,str(total))


def analyze(output, smoke=False):
    output.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(input_path(1))
    fo = pd.read_csv(ROOT/"第二版实验/results/q1_oof.csv")
    assert np.array_equal(fo.row, np.arange(1,len(data)+1))
    assert np.array_equal(fo.actual, pd.Categorical(data.CallType, categories=CALLS).codes)
    assert np.array_equal(fo.group, pd.to_datetime(data.Date,format="%d/%m/%Y").dt.strftime("%Y-%m-%d"))
    assert fo.groupby("group").fold.nunique().eq(1).all() and fo.fold.nunique()==16
    # Known tie case: 3 assignments, one of them gives the largest |T|.
    s, dist, pairs = block_distribution([1,1,2], np.array([False,False,True]))
    assert (s,dict(dist),pairs)==(2,{-1:2,2:1},2)
    if smoke:
        print(exact_pair(data,"Fpeak","flat","mod"))
        return
    rows = [exact_pair(data, feature, a, b) for feature in FEATURES for a,b in combinations(CALLS,2)]
    out = pd.DataFrame(rows,columns=["feature","first","second","mixed_dates","first_n","second_n",
        "comparable_pairs","statistic","within_date_delta","p_exact","permutations"])
    out["p_holm"] = holm(out.p_exact)
    pooled = pd.read_csv(ROOT/"results/q1_pairwise.csv")
    out.merge(pooled[["feature","first","second","cliffs_delta","p_holm"]],
              on=["feature","first","second"],suffixes=("","_pooled"),validate="one_to_one").to_csv(
                  output/"q1v3_date_pairs.csv",index=False,float_format="%.10g")
    records = []
    for model in ("reference","TabICLv2","Mitra-v2"):
        proba=fo[[f"{model}_{i}" for i in range(3)]].to_numpy()
        predicted=proba.argmax(1)
        for a,call in enumerate(CALLS):
            counts=np.bincount(predicted[fo.actual==a],minlength=3)
            records.append((model,call,*counts))
    pd.DataFrame(records,columns=["model","actual",*CALLS]).to_csv(output/"q1v3_confusion.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    analyze(args.output,args.smoke)
