"""Grouped comparison of playback slopes and the conditional-onset regularizer."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models_playback import design, INTERACTIONS, FEATURE_NAMES, LAMBDAS, time_cdf
from penguin_data import CALLS, STATES, STATE_NAMES, BEHAVIOURS, CONTEXTS, STIMULI, HORIZON, context_weights, state_index, SEED

TIMES = np.array([3.,10.,30.,60.])
TIME_PENALTIES = (1.,4.,16.)
SPEC = ('shared', 'call_slopes')
EXTRA = [f'{s}:log_{v}' for v in ('distance','size') for s in CALLS]


def covariates(stim, context, distance, count, model):
    x = design(stim,context,distance,count)
    if model == 'shared':
        return x
    if model != 'call_slopes':
        raise ValueError(model)
    return np.concatenate([x, x[...,1:4]*x[...,6,None], x[...,1:4]*x[...,7,None]],axis=-1)


def state_logp(beta, x):
    n = 3*x.shape[-1]
    score = (x @ beta[:n].reshape(-1,3)) @ STATES.T + INTERACTIONS @ beta[n:]
    return score-logsumexp(score,axis=-1,keepdims=True)


def state_fit(x, y, lam):
    num = x.shape[1]*3 + INTERACTIONS.shape[1]
    w = np.full(num,float(lam)); w[:3] *= .1
    def objective(beta):
        lp = state_logp(beta,x); p = np.exp(lp)
        loss = -lp[np.arange(len(x)),y].sum()+.5*np.dot(beta*w,beta)
        g = np.r_[(x.T@(p@STATES-STATES[y])).ravel(), (p@INTERACTIONS-INTERACTIONS[y]).sum(axis=0)]
        return loss, g+w*beta
    opt = minimize(objective,np.zeros(num),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500})
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x


def fit_clock(x, t, lam):
    # Same truncated distribution, offset and box constraints as the original fixed-lambda fit.
    shift = .5
    logs = np.log(t+shift); lo,hi=np.log(shift),np.log(HORIZON+shift)
    def objective(theta):
        mu=x@theta[:-1]; sigma=np.exp(theta[-1]); z=(logs-mu)/sigma
        a,b=(lo-mu)/sigma,(hi-mu)/sigma
        mass = np.maximum(ndtr(b)-ndtr(a),1e-14)
        pa,pb=np.exp(-a*a/2)/np.sqrt(2*np.pi),np.exp(-b*b/2)/np.sqrt(2*np.pi)
        val=(.5*z*z+theta[-1]+np.log(mass)+logs).sum()+.5*lam*(theta[1:-1]@theta[1:-1])
        slope=x.T@((-z+(pa-pb)/mass)/sigma); slope[1:]+=lam*theta[1:-1]
        spread=(1.-z*z+(a*pa-b*pb)/mass).sum()
        return val, np.r_[slope,spread]
    initial=np.r_[np.median(logs),np.zeros(x.shape[1]-1),np.log(np.clip(logs.std(),.2,2.))]
    opt=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=[(-6.,6.)]*x.shape[1]+[(np.log(.1),np.log(3.))],options={'ftol':1e-11,'gtol':1e-7,'maxiter':500})
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x


def probability_clock(x, days, idx, flags, onset, tune_rows, fold, name, endpoint):
    subset=flags[idx]; y=state_index(subset); dm=x[idx]; groups=days[idx]
    splits=list(GroupKFold(4).split(dm,groups=groups))
    scores=[]
    for lam in LAMBDAS:
        loss=[]
        for a,b in splits:
            lp=state_logp(state_fit(dm[a],y[a],lam),dm[b]);loss.extend(-lp[np.arange(len(b)),y[b]])
        scores.append(np.mean(loss))
    lam=LAMBDAS[np.argmin(scores)]
    tune_rows.extend((fold,name,endpoint,'state','all',k,v,k==lam) for k,v in zip(LAMBDAS,scores))
    beta=state_fit(dm,y,lam)
    clocks=[]
    if endpoint=='recorded_onset':
        for j,label in enumerate(BEHAVIOURS):
            paths=[]
            for penalty in TIME_PENALTIES:
                loss=[]
                for a,b in splits:
                    a=a[subset[a,j]];b=b[subset[b,j]]
                    if len(b):
                        p=fit_clock(dm[a],onset[idx[a],j],penalty)
                        pred=time_cdf(p,dm[b,None,:],TIMES[:3]); obs=onset[idx[b],j,None]<=TIMES[:3]
                        loss.extend(((pred-obs)**2).mean(axis=1))
                paths.append(np.mean(loss))
            penalty=TIME_PENALTIES[np.argmin(paths)]
            tune_rows.extend((fold,name,endpoint,'conditional_clock',label,k,v,k==penalty) for k,v in zip(TIME_PENALTIES,paths))
            positive=subset[:,j]; clocks.append(fit_clock(dm[positive],onset[idx[positive],j],penalty))
    return beta, clocks


def bootstrap_errors(frame,key,models,reference,loss):
    groups=np.sort(frame.expID.unique()); draw=np.random.default_rng(SEED+603).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=5000)/len(groups)
    errors=[]
    for t,g in [(('all','all'),frame),*list(frame.groupby(key,sort=False))]:
        if not isinstance(t,tuple): t=(t,'all')
        per=pd.DataFrame({m:loss(g,m) for m in models});per['expID']=g.expID.to_numpy();by=per.groupby('expID').mean().reindex(groups)
        assert by.notna().all().all()
        for m in models:
            delta=by[m]-by[reference]; lo,hi=np.quantile(draw@delta,[.025,.975]);errors.append((*t,m,reference,by[m].mean(),delta.mean(),lo,hi))
    return pd.DataFrame(errors,columns=['behaviour','time_s','model','reference','mean_score','delta','lower','upper'])


def full_grid(data, beta, clocks, model):
    weights=context_weights(data); rows=[]
    for stim in STIMULI:
        for dist in (50,100,150):
            x=covariates(stim,np.array(CONTEXTS),dist,8,model)
            p=np.exp(state_logp(beta,x))@STATES
            for j,beh in enumerate(BEHAVIOURS):
                px=p[:,j]*weights; total=px.sum(); lower,upper=0.,HORIZON
                for _ in range(45):
                    mid=(lower+upper)/2
                    if px@time_cdf(clocks[j],x,mid)<total*.5:lower=mid
                    else: upper=mid
                subset=data[(data.CallType==stim)&(data.Distance==dist)]; ev=(subset.VocalNcalls>0) if beh=='Vocal' else subset[f'{beh}YesNo'].eq(1)
                rows.append((model,stim,dist,8,beh,len(subset),ev.sum(),total,(lower+upper)/2 if ev.any() else np.nan))
    return rows


def run_timing(data, flags, onset, smoke=False):
    grp=data.expID.to_numpy(); primary=flags.copy();primary[:,1]=data.VocalNcalls.gt(0)
    foldseq=list(GroupKFold(5).split(data,groups=grp));tuning=[];state_rows=[];timing_rows=[];grid_rows=[];coefs=[]
    for model in SPEC:
        x=covariates(data.CallType,data.Context,data.Distance,data.Npenguins,model)
        for name,ev in [('recorded_onset',primary),('net_response',flags)]:
            for fold,(train,test) in enumerate(foldseq,1):
                assert not set(grp[train])&set(grp[test]); beta,clocks=probability_clock(x,grp,train,ev,onset,tuning,fold,model,name)
                lp=state_logp(beta,x[test]); p=np.exp(lp); y=state_index(ev[test]); marginal=p@STATES
                state_rows.append(pd.DataFrame({'source_csv_line':test+2,'expID':grp[test],'fold':fold,'model':model,'endpoint':name,'state':y,'log_loss':-lp[np.arange(len(test)),y],**{f'p_{s}':p[:,k] for k,s in enumerate(STATE_NAMES)}}))
                if name=='recorded_onset':
                    for j,b in enumerate(BEHAVIOURS):
                        risk=marginal[:,j,None]*time_cdf(clocks[j],x[test,None,:],TIMES)
                        for k,t in enumerate(TIMES):
                            timing_rows.append(pd.DataFrame({'source_csv_line':test+2,'expID':grp[test],'fold':fold,'model':model,'behaviour':b,'time_s':t,'observed':(ev[test,j]&(onset[test,j]<=t)).astype(int),'risk':risk[:,k]}))
                if smoke:break
            if smoke:continue
            beta,clocks=probability_clock(x,grp,np.arange(len(x)),ev,onset,tuning,'full',model,name)
            coefs.extend((model,name,k,v) for k,v in enumerate(beta))
            if name=='recorded_onset':grid_rows.extend(full_grid(data,beta,clocks,model))
    state=pd.concat(state_rows,ignore_index=True);timing=pd.concat(timing_rows,ignore_index=True)
    if smoke:return {'timing_oof':timing,'state_oof':state}
    key=['source_csv_line','expID','fold','behaviour','time_s','observed']; wide=timing.pivot(index=key,columns='model',values='risk').reset_index()
    v4=pd.read_csv(ROOT/'第四版实验/results/q3v4_timing_oof.csv'); wide=wide.merge(v4[key+['two_part','cox']],on=key,validate='one_to_one');assert len(wide)==len(data)*12
    score=bootstrap_errors(wide,['behaviour','time_s'],['two_part','cox',*SPEC],'two_part',lambda g,m:(g[m]-g.observed)**2)
    step=bootstrap_errors(wide,['behaviour','time_s'],list(SPEC),'shared',lambda g,m:(g[m]-g.observed)**2)
    state_scores=[]
    for ep,t in state.groupby('endpoint'):
        tab=t.pivot(index=['expID','source_csv_line'],columns='model',values='log_loss').reset_index(); a=tab.groupby('expID')[list(SPEC)].mean();delta=a.call_slopes-a.shared
        draw=np.random.default_rng(SEED+603).multinomial(len(a),np.full(len(a),1/len(a)),size=5000)/len(a)
        state_scores.append((ep,a.shared.mean(),a.call_slopes.mean(),delta.mean(),*np.quantile(draw@delta,[.025,.975])))
    archived=pd.read_csv(ROOT/'第四版实验/results/q3v4_state_oof.csv').query("validation_unit=='paired'")
    align=state.query("model=='shared'").merge(archived,on=['endpoint','expID','fold','source_csv_line','state'],suffixes=('_v6','_v4'),validate='one_to_one');assert len(align)==len(data)*2
    for name in STATE_NAMES:np.testing.assert_allclose(align[f'p_{name}_v4'],align[f'p_{name}_v6'],atol=2e-5,rtol=1e-5)
    return {'timing_oof':wide,'state_oof':state,'timing_scores':pd.concat([score,step],ignore_index=True),
            'state_scores':pd.DataFrame(state_scores,columns=['endpoint','shared','call_slopes','difference','lower','upper']),
            'timing_tuning':pd.DataFrame(tuning,columns=['fold','model','endpoint','component','behaviour','penalty','inner_score','selected']),
            'timing_scenarios':pd.DataFrame(grid_rows,columns=['model','stimulus','distance_m','group_size','behaviour','trials_at_distance','events_at_distance','p60','supported_median_s']),
            'state_parameters':pd.DataFrame(coefs,columns=['model','endpoint','index','value'])}
