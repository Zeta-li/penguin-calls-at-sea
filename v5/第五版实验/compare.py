"""Joint paired evaluation on the original leave-recording-date-out folds."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from penguin_data import CALLS, SEED, input_path

OUT = HERE / "results"
P_COLS = [f"p_{k}" for k in CALLS]
METRICS = ["balanced_accuracy","accuracy","log_loss","macro_f1",*[f"recall_{k}" for k in CALLS]]


def metrics(truth, prediction, prob, weights):
    """Compute paired metrics for each repeat, using the same sampled rows."""
    actual = np.stack([weights @ (truth == k) for k in range(3)],axis=1)
    chosen = np.stack([weights @ (prediction == k) for k in range(3)],axis=1)
    hit = np.stack([weights @ ((truth == k) & (prediction == k)) for k in range(3)],axis=1)
    total = actual.sum(axis=1)
    with np.errstate(invalid="ignore",divide="ignore"):
        recall = hit/actual
        f1 = np.divide(2*hit, actual+chosen, out=np.zeros_like(hit,dtype=float), where=(actual+chosen)>0)
    individual_loss = -np.log(np.clip(prob[np.arange(len(truth)),truth],1e-15,1))
    return np.column_stack([recall.mean(1),hit.sum(1)/total,weights@individual_loss/total,f1.mean(1),recall])


def read_methods():
    source = pd.read_csv(input_path(1),usecols=["Date","CallType"])
    truth = pd.Categorical(source.CallType,categories=CALLS).codes.astype(int)
    days = pd.to_datetime(source.Date,format="%d/%m/%Y").dt.strftime("%Y-%m-%d").to_numpy()
    ids = np.arange(1,len(truth)+1)
    learned = pd.read_csv(OUT/"q1f_predictions.csv")
    models = {}
    def take(key, table, date, predicted):
        t = table.sort_values("row")
        assert np.array_equal(t.row,ids) and np.array_equal(t[date],days)
        assert np.array_equal(t.actual,source.CallType)
        prob=t[P_COLS].to_numpy()
        labels=pd.Categorical(t[predicted],categories=CALLS).codes.astype(int)
        assert (labels>=0).all() and (prob>=0).all() and np.allclose(prob.sum(1),1,atol=2e-6)
        models[key]=(prob,labels)
    for name,t in learned.groupby("method"):
        take(name,t,"held_date","prediction")
    classical=pd.read_csv(OUT/"q1c_oof.csv")
    for key in ("classical_nested","common_six_only","ordinary_decision_only","kernel_only","trees_only"):
        take(key,classical[classical.method==key],"date","decision")
    v4=pd.read_csv(ROOT/"第二版实验/results/q1_oof.csv").sort_values("row")
    assert np.array_equal(v4.row,ids) and np.array_equal(v4.group,days) and np.array_equal(v4.actual,truth)
    train_priors = np.stack([np.bincount(truth[days!=date],minlength=3)/sum(days!=date) for date in days])
    mitra=v4[[f"Mitra-v2_{k}" for k in range(3)]].to_numpy()
    for name,p in [("mitra",mitra),("cross_arch",(mitra+models["blend_empirical"][0])/2)]:
        for suffix,point in [("empirical",p),("balanced",p/train_priors)]:
            models[f"{name}_{suffix}"]=(p,point.argmax(1))
    return source,truth,days,models


def run():
    source,truth,days,models=read_methods()
    groups,inv=np.unique(days,return_inverse=True)
    rng=np.random.default_rng(SEED+505)
    draws=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000)
    weights=np.vstack([np.ones(len(truth),int), draws[:,inv],1-np.eye(len(groups),dtype=int)[:,inv]])
    all_scores={key:metrics(truth,choice,p,weights) for key,(p,choice) in models.items()}
    rows,uncertainty,influence,conf,trans,obs,perdate=[],[],[],[],[],[],[]
    for name,(p,choice) in models.items():
        scores=all_scores[name]
        rows.append([name,len(truth),*scores[0]])
        for i in range(len(truth)):
            obs.append([i+1,days[i],CALLS[truth[i]],name,CALLS[choice[i]],*p[i]])
        for a,ka in enumerate(CALLS):
            for b,kb in enumerate(CALLS):
                conf.append([name,ka,kb,int(((truth==a)&(choice==b)).sum())])
        for i,g in enumerate(groups):
            used=(inv==i)
            perdate.append([name,g,used.sum(),int((truth[used]==choice[used]).sum()),
                            -np.log(p[np.where(used)[0],truth[used]]).mean()])
        for base in ("raw_empirical","raw_balanced"):
            diff=scores-all_scores[base]
            basechoice=models[base][1]
            for k,label in enumerate(METRICS):
                sample=diff[1:len(draws)+1,k]
                usable=sample[np.isfinite(sample)]
                interval=np.quantile(usable,[.025,.975])
                uncertainty.append([name,base,label,scores[0,k],all_scores[base][0,k],diff[0,k],*interval,len(usable)])
                for g,v in zip(groups,diff[len(draws)+1:,k]):
                    influence.append([name,base,label,g,v])
            for c,klass in enumerate(CALLS):
                mask=truth==c
                win=(choice==truth)&(basechoice!=truth);lose=(basechoice==truth)&(choice!=truth)
                trans.append([name,base,klass,mask.sum(),int((mask&win).sum()),int((mask&lose).sum())])
    for n,data,cols in [
        ("scores",rows,["method","n",*METRICS]),
        ("comparisons",uncertainty,["method","reference","metric","estimate","reference_value","delta","lower","upper","valid_draws"]),
        ("held_day_influence",influence,["method","reference","metric","omitted_test_date","delta"]),
        ("confusions",conf,["method","actual","decision","n"]),
        ("corrections",trans,["method","reference","actual","n","gained","lost"]),
        ("predictions",obs,["row","date","actual","method","decision",*P_COLS]),
        ("per_day",perdate,["method","date","n","correct","log_loss"])]:
        pd.DataFrame(data,columns=cols).to_csv(OUT/f"q1v5_{n}.csv",index=False,float_format="%.12g")
    # Historical date folds and one chronological test are reported as separate designs.
    make_figure(pd.DataFrame(uncertainty,columns=["name","base","metric","estimate","ref","delta","lo","hi","n"]))


def make_figure(table):
    import matplotlib.pyplot as plt
    from figures_apmcm import save
    names=["raw_balanced","augmented_balanced","blend_balanced","classical_nested","cross_arch_balanced"]
    labels=["Raw Tab + prior","Augmented Tab + prior","Two-view Tab + prior","Nested classical","Two-architecture + prior"]
    fig,axes=plt.subplots(1,2,figsize=(8.1,2.95),layout="constrained",sharey=True)
    for column,metric,factor,title in [(0,"balanced_accuracy",100,"Balanced accuracy difference (pp)"),(1,"log_loss",1,"Log loss difference")]:
        a=axes[column]
        t=table[(table.base=="raw_empirical")&(table.metric==metric)].set_index("name").loc[names]
        a.axvline(0,color="#647781",lw=.8,ls="--")
        for j,(_,r) in enumerate(t.iterrows()):
            color="#096653" if j==2 else "#466b80"
            a.plot(np.array([r.lo,r.hi])*factor,[j,j],c=color,lw=1.5)
            a.plot(r.delta*factor,j,"o",c=color,ms=5)
        a.set_xlabel(title,fontsize=9);a.set_yticks(range(len(labels)),labels,fontsize=9)
        a.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    folder=HERE/"figures";folder.mkdir(exist_ok=True)
    save(fig,folder,"result_q1_date_comparison")


if __name__=="__main__":
    run()
