"""Date-held-out, shrinkage-based acoustic classification and an annotation audit."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from penguin_data import CALLS, SEED, input_path

OUT = HERE / "results"
FIG = HERE / "figures"
GRID = [(a, b) for a in (0., .5, 1.) for b in (.25, .6, 1.)]


def spectrum_features(data):
    """Represent original measurements by pitch and log frequency ratios."""
    return np.log(np.column_stack([
        data.Duration, data.F01, data.Fpeak/data.F01,
        data.Q50/data.F01, data.Q50/data.Q25, data.Q75/data.Q50,
    ]))


def discriminant(x, y, test, alpha, beta):
    center, scale = x.mean(axis=0), x.std(axis=0)
    z, v = (x-center)/scale, (test-center)/scale
    counts = np.bincount(y, minlength=3)
    means = np.stack([z[y == k].mean(axis=0) for k in range(3)])
    covs = np.stack([np.cov(z[y == k], rowvar=False) for k in range(3)])
    pooled = np.average(covs, axis=0, weights=counts-1)
    score = []
    for k in range(3):
        cov = (1-alpha)*pooled + alpha*covs[k]
        cov = (1-beta)*cov + beta*np.eye(z.shape[1])*np.trace(cov)/z.shape[1]
        delta = v-means[k]
        md2 = np.sum(np.linalg.solve(cov, delta.T).T*delta, axis=1)
        score.append(np.log(counts[k]/len(y))-.5*(np.linalg.slogdet(cov)[1]+md2))
    logits = np.stack(score, axis=1)
    return np.exp(logits-logsumexp(logits, axis=1, keepdims=True))


def loss(y, p):
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))


def metrics(y, p):
    classified = p.argmax(axis=1)
    recall = [(classified[y == c] == c).mean() for c in range(3)]
    return np.r_[loss(y, p).mean(), np.mean(classified == y), np.mean(recall), recall]


def run():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    df = pd.read_csv(input_path(1))
    x = spectrum_features(df)
    y = pd.Categorical(df.CallType, categories=CALLS).codes.astype(int)
    dates = pd.to_datetime(df.Date, format="%d/%m/%Y").dt.strftime("%Y-%m-%d").to_numpy()
    folds, prediction = [], np.zeros((len(y),3))
    for held in np.unique(dates):
        train, test = np.where(dates != held)[0], np.where(dates == held)[0]
        tuning = np.zeros(len(GRID))
        for sub, val in GroupKFold(4).split(x[train], y[train], dates[train]):
            tr, va = train[sub], train[val]
            for j, (a, b) in enumerate(GRID):
                tuning[j] += loss(y[va], discriminant(x[tr],y[tr],x[va],a,b)).sum()
        chosen = int(tuning.argmin())
        for j,(a,b) in enumerate(GRID):
            folds.append((held,a,b,tuning[j]/len(train),j==chosen))
        prediction[test] = discriminant(x[train], y[train], x[test], *GRID[chosen])
    prior = pd.read_csv(ROOT/"第二版实验/results/q1_oof.csv").sort_values("row")
    assert np.array_equal(prior.row,np.arange(1,len(df)+1))
    assert np.array_equal(prior.actual,y) and np.array_equal(prior.group,dates)
    methods = {k:prior[[f"{prefix}_{i}" for i in range(3)]].to_numpy() for k, prefix in
               [("Ridge","reference"),("TabICLv2","TabICLv2"),("Mitra-v2","Mitra-v2")]}
    methods["RDA"] = prediction
    has_f02 = df.F02.notna().to_numpy()
    assert np.array_equal(has_f02,y==2)
    artifact = prediction.copy()
    artifact[has_f02] = [0.,0.,1.]
    artifact[~has_f02,2] = 0.
    methods["RDA+F02-mask (audit)"] = artifact/artifact.sum(1,keepdims=True)
    pd.DataFrame(folds,columns=["test_date","class_cov_mix","spherical_mix","inner_log_loss","selected"]).to_csv(OUT/"q1_tuning.csv",index=False)
    names = ["log_loss","accuracy","balanced_accuracy",*["recall_"+c for c in CALLS]]
    scores, rows, deltas = [], [], []
    rng = np.random.default_rng(SEED)
    groups, group_id = np.unique(dates,return_inverse=True)
    # Hold predictions fixed and resample recording dates jointly across models.
    weights = rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=5000)
    for name,p in methods.items():
        assert np.allclose(p.sum(1),1.,atol=1e-6) and np.all(p>=0)
        scores.append([name,*metrics(y,p)])
        rows.extend((i+1,dates[i],CALLS[y[i]],name,*p[i]) for i in range(len(y)))
        for base in ("Ridge","TabICLv2"):
            diff = loss(y,p)-loss(y,methods[base])
            draws = (weights @ np.bincount(group_id,weights=diff))/(weights @ np.bincount(group_id))
            deltas.append([name,base,diff.mean(),*np.quantile(draws,[.025,.975])])
    pd.DataFrame(scores,columns=["method",*names]).to_csv(OUT/"q1_scores.csv",index=False)
    pd.DataFrame(rows,columns=["row","held_date","actual","method","p_flat","p_mod","p_tv"]).to_csv(OUT/"q1_predictions.csv",index=False)
    pd.DataFrame(deltas,columns=["method","comparator","delta_log_loss","lower","upper"]).to_csv(OUT/"q1_paired_scores.csv",index=False)
    pd.crosstab(df.CallType,has_f02).rename(columns={False:"F02_missing",True:"F02_measured"}).to_csv(OUT/"q1_marker_audit.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(8.2,3.1),gridspec_kw={"width_ratios":[1.05,1.2]})
    tab = pd.DataFrame(scores,columns=["method",*names])
    labs=["Ridge","TabICLv2","Mitra-v2","RDA","RDA + label cue*"]
    colors=["#596a78","#326b8b","#6793a6","#206554","#9d572c"]
    ax[0].barh(range(5),tab.balanced_accuracy*100,color=colors,height=.6)
    ax[0].set(yticks=range(5),yticklabels=labs,xlim=(0,100),xlabel="Balanced accuracy (%)")
    ax[0].invert_yaxis()
    for z,n in enumerate(tab.balanced_accuracy): ax[0].text(n*100+2,z,f"{n*100:.1f}",va="center",fontsize=8)
    z=np.arange(3)
    for k,n in enumerate((1,3,4)):
        ax[1].bar(z+(k-1)*.24,100*tab.loc[n,names[3:]].to_numpy(float),.23,color=colors[n],label=labs[n])
    ax[1].set(xticks=z,xticklabels=["Flat","Modulated","Two-voice"],ylim=(0,111),ylabel="Class recall (%)")
    ax[1].legend(loc="upper left",fontsize=7,frameon=False)
    for a in ax: a.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    from figures_apmcm import save
    save(fig, FIG, "result_q1_marker_audit")


if __name__=="__main__":
    run()
