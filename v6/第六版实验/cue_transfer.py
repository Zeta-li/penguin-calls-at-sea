"""Test whether reservoir curve shape adds field prediction beyond call type and distance."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from penguin_data import CALLS, METRICS, SEED, input_path
from models_playback import design


def curve_values(frame, results):
    weights = pd.read_csv(results / 'q2v6_weights.csv').set_index('criterion')
    params = pd.read_csv(results / 'q2v6_parameters.csv').set_index(['call', 'representation'])
    values = {k: np.zeros(len(frame)) for k in ('mean', 'class_shrunk')}
    for call in CALLS:
        ix = frame.CallType.eq(call).to_numpy()
        d = frame.Distance.to_numpy()[ix]
        part = params.loc[call].loc[list(METRICS)]
        a,b,c = (part[name].to_numpy()[:,None] for name in ('a','b_m','c'))
        fidelity = a/(1+(d[None,:]/b)**c)
        for k in values:
            w = weights.loc[k, list(METRICS)].to_numpy(dtype=float)
            np.testing.assert_allclose(w.sum(), 1)
            values[k][ix] = w @ fidelity
    return values


def evidence_features(x, types, logd, f, train, test):
    if f is None:
        return x[train], x[test]
    extra = np.zeros(len(types))
    for k in CALLS:
        take = train[types[train] == k]
        at = np.flatnonzero(types == k)
        basis = np.stack((np.ones_like(logd), logd), axis=1)
        coef = np.linalg.lstsq(basis[take], f[take], rcond=None)[0]
        extra[at] = f[at] - basis[at] @ coef
    return np.column_stack((x[train], extra[train])), np.column_stack((x[test], extra[test]))


def run(results):
    frame = pd.read_csv(input_path(2))
    frame['source_csv_line'] = np.arange(len(frame))+2
    keep = frame.Distance.between(4,256).groupby(frame.expID).transform('all')
    frame = frame[keep].reset_index(drop=True)
    assert frame.groupby('expID').size().eq(2).all()
    old = pd.read_csv(ROOT/'第四版实验/results/q3v4_state_oof.csv')
    old = old.query("endpoint == 'net_response' and validation_unit == 'paired'").set_index('source_csv_line')
    folds = old.loc[frame.source_csv_line,'fold'].to_numpy()
    days = frame.expID.to_numpy(); types = frame.CallType.to_numpy(); logd=np.log2(frame.Distance.to_numpy()/50)
    assert pd.DataFrame(dict(fold=folds,pair=days)).groupby('pair').fold.nunique().max()==1
    basic = design(types,frame.Context.to_numpy(),frame.Distance.to_numpy(),frame.Npenguins.to_numpy())[:,1:]
    basic = np.column_stack((basic,*[(types==k)*logd for k in CALLS]))
    candidates = {'behavior_only':None,**curve_values(frame,results)}
    probs=[]; picks=[]
    for target in ['VocalYesNo','ApproachYesNo']:
        y=frame[target].astype(int).to_numpy()
        for method,f in candidates.items():
            for fold in np.unique(folds):
                tr=np.flatnonzero(folds!=fold); te=np.flatnonzero(folds==fold)
                def predict(a,b,c):
                    va,vb=evidence_features(basic,types,logd,f,a,b)
                    scaler=StandardScaler().fit(va)
                    model=LogisticRegression(C=c,solver='lbfgs',max_iter=1000,tol=1e-9).fit(scaler.transform(va),y[a])
                    return model.predict_proba(scaler.transform(vb))[:,1]
                inner=list(GroupKFold(4).split(tr,groups=days[tr]))
                options=[]
                for c in (.1,1.,10.):
                    loss=np.concatenate([(predict(tr[a],tr[b],c)-y[tr[b]])**2 for a,b in inner]).mean()
                    options.append((loss,c));picks.append((target,method,fold,c,loss))
                p=predict(tr,te,min(options)[1]);
                for i,v in zip(te,p):probs.append((target,method,fold,frame.source_csv_line[i],days[i],int(y[i]),v))
    out=pd.DataFrame(probs,columns=['endpoint','method','fold','source_csv_line','pair','event','p'])
    out.to_csv(results/'q3cue_oof.csv',index=False,float_format='%.12g')
    pd.DataFrame(picks,columns=['endpoint','method','fold','C','inner_brier']).to_csv(results/'q3cue_tuning.csv',index=False,float_format='%.12g')
    rows=[]; rng=np.random.default_rng(SEED+604)
    pairs=np.unique(days); draws=rng.multinomial(len(pairs),np.full(len(pairs),1/len(pairs)),size=5000)
    for target,g in out.groupby('endpoint'):
        wide=g.pivot(index=['pair','source_csv_line','event'],columns='method',values='p')
        y=wide.index.get_level_values('event').to_numpy()
        err=wide.sub(y,axis=0).pow(2).groupby(level='pair').mean().loc[pairs]
        for method in candidates:
            d=(err[method]-err['behavior_only']).to_numpy();lo,hi=np.quantile(draws@d/len(d),[.025,.975])
            rows.append((target,method,len(y),len(pairs),int(y.sum()),err[method].mean(),d.mean(),lo,hi))
    pd.DataFrame(rows,columns=['endpoint','method','n','pairs','events','brier','delta','lower','upper']).to_csv(results/'q3cue_scores.csv',index=False,float_format='%.12g')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results'); run(parser.parse_args().results)
