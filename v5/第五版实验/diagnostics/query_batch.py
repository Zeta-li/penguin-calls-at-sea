"""Check whether querying recordings separately alters the held-date predictions.

Run with the foundation inference environment. By default the model checkpoint is
looked up in 第五版实验/.model_cache; --cache can point to a prepared weights folder.
Results concern the two largest test dates, with original training sets fixed.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from foundation import ROOT, observations, representations
sys.path.insert(0, str(ROOT/"第二版实验"))
from frozen_models import SEED, TAB_FILE, ensure_asset, tabicl

ATOL = 1e-7
RTOL = 2e-6


def run(cache, output):
    ensure_asset(cache, TAB_FILE)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    x, y, dates = observations()
    views = representations(x)
    largest = pd.Series(dates).value_counts().head(2).index.tolist()
    rows = []
    for date in largest:
        train = dates != date
        test_rows = np.flatnonzero(~train)
        priors = np.bincount(y[train]) / train.sum()
        pairs = {}
        for name, features in views.items():
            together = tabicl(cache, features[train], y[train], features[test_rows])
            separate = np.concatenate([tabicl(cache, features[train], y[train], features[i:i+1])
                                       for i in test_rows])
            pairs[name] = together, separate
        pairs["blend"] = tuple((a.astype(float)+b.astype(float))/2
                               for a,b in zip(pairs["raw"],pairs["augmented"]))
        for name,(together,separate) in pairs.items():
            diff = np.abs(together-separate)
            raw_changes = np.count_nonzero(together.argmax(1)!=separate.argmax(1))
            balanced_changes = np.count_nonzero((together/priors).argmax(1)!=(separate/priors).argmax(1))
            rows.append([date,len(test_rows),name,diff.max(),diff.mean(),raw_changes,balanced_changes,
                         RTOL,ATOL,np.allclose(together,separate,rtol=RTOL,atol=ATOL)])
    columns = ["held_date","rows","view","max_abs_probability_difference","mean_abs_probability_difference",
               "argmax_changes","prior_adjusted_argmax_changes","rtol","atol","all_within_tolerance"]
    result = pd.DataFrame(rows,columns=columns)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False,float_format="%.12g")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=HERE.parent/".model_cache")
    parser.add_argument("--output",type=Path,default=HERE/"query_batch.csv")
    args=parser.parse_args()
    run(args.cache.resolve(),args.output)
