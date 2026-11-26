"""Playback intensity, event–intensity association, and folded time-model diagnostics."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from penguin_data import STIMULI, CONTEXTS, HORIZON, SEED, input_path, event_flags, latency_values, context_weights
from q3_timing import covariates, run_timing

FIELDS={'HeadsUpDuration':0,'VocalNcalls':1,'ApproachDuration':2,'VocalRelNCalls':1}
PENALTIES=(1.,4.,16.)
MODELS=('train_mean','stimulus_mean','shared','call_slopes')


def fit_mean(x, y, family, penalty):
    """Ridge quasi-likelihood for a conditional mean. Intercepts remain unpenalized."""
    if family=='fractional' and not np.all((y>=0)&(y<=1)):
        raise ValueError('Fraction outside [0, 1]')
    def objective(beta):
        eta=x@beta
        if family=='fractional':
            mu=expit(eta);loss=(np.logaddexp(0.,eta)-eta*y).sum()
        else:
            mu=np.exp(eta);loss=(mu-eta*y).sum()
        grad=x.T@(mu-y); grad[1:]+=penalty*beta[1:]
        return loss+.5*penalty*(beta[1:]@beta[1:]),grad
    start=np.zeros(x.shape[1]); m=y.mean()
    start[0]=np.log((m+.01)/(1.-m+.01)) if family=='fractional' else np.log(max(.01,m))
    opt=minimize(objective,start,method='L-BFGS-B',jac=True,bounds=[(-12,12)]*len(start),options={'maxiter':300,'ftol':1e-11,'gtol':1e-7})
    if not opt.success:raise RuntimeError(opt.message)
    return opt.x


def level_fit(x,y,field,penalty):
    if 'Duration' in field:return fit_mean(x,y/HORIZON,'fractional',penalty)
    return fit_mean(x,y-1.,'poisson',penalty)


def level_predict(x,b,field):
    if 'Duration' in field:return HORIZON*expit(x@b)
    return 1.+np.exp(x@b)


def fit_hurdle(x,y,field,penalty):
    hit=y>0
    return fit_mean(x,hit.astype(float),'fractional',penalty),level_fit(x[hit],y[hit],field,penalty)


def hurdle_parts(x,coefs,field):
    b1,b2=coefs
    return expit(x@b1),level_predict(x,b2,field)


def tune(x, y, field, groups, idx):
    path=[]
    for lam in PENALTIES:
        errors=[]
        for train,test in GroupKFold(4).split(x[idx],groups=groups[idx]):
            a,b=idx[train],idx[test];p,m=hurdle_parts(x[b],fit_hurdle(x[a],y[a],field,lam),field)
            errors.extend((p*m-y[b])**2)
        path.append(np.mean(errors))
    best=PENALTIES[int(np.argmin(path))]
    return best,path


def aggregate_score(table,group_draws,ids,reference,alternatives,phase):
    cols={'all':'unconditional','responders':'positive','post_onset':'positive'}
    part=table if phase=='all' else table[table.observed>0]
    values=part.pivot(index=['source_csv_line','expID','observed'],columns='model',values=cols[phase]).reset_index()
    err=values[alternatives].sub(values.observed,axis=0)
    sumsq=err.pow(2).groupby(values.expID).sum().reindex(ids,fill_value=0)
    count=values.groupby('expID').size().reindex(ids,fill_value=0).to_numpy()
    draws=np.sqrt((group_draws@sumsq.to_numpy())/(group_draws@count)[:,None]);base=alternatives.index(reference)
    point=np.sqrt(err.pow(2).mean());mean_absolute=err.abs().mean(); rows=[]
    for i,m in enumerate(alternatives):
        lo,hi=np.quantile(draws[:,i]-draws[:,base],[.025,.975]);rows.append((phase,m,reference,len(part[part.model==m]),point[m],mean_absolute[m],point[m]-point[reference],lo,hi))
    return rows


def score_and_association(data,all_pred,latencies):
    ids=np.sort(data.expID.unique()); n=len(ids)
    draws=np.random.default_rng(SEED+624).multinomial(n,np.ones(n)/n,size=5000)
    scores=[];couplings=[]
    for k,(field,bi) in enumerate(FIELDS.items()):
        d=all_pred[all_pred.field==field]
        for reference in ('train_mean','stimulus_mean'):
            for phase in ('all','responders'):
                scores.extend((field,*r) for r in aggregate_score(d,draws,ids,reference,list(MODELS),phase))
        scores.extend((field,*r) for r in aggregate_score(d,draws,ids,'shared',['shared','call_slopes'],'all'))
        scores.extend((field,*r) for r in aggregate_score(d,draws,ids,'shared',['shared','shared_after_observed_onset'],'post_onset'))
        y=data[field].to_numpy(); hit=y>0; t=latencies[hit,bi]; yp=y[hit]; gids=data.loc[hit,'expID'].to_numpy()
        assert np.isfinite(t).all()
        point=spearmanr(t,yp).statistic
        # Re-sample all trial pairs; only responder rows contribute to each conditional correlation.
        index=np.searchsorted(ids,gids); sims=[]
        for w in draws[:2000]:
            ix=np.repeat(np.arange(len(yp)),w[index]);sims.append(spearmanr(t[ix],yp[ix]).statistic)
        lo,hi=np.nanquantile(sims,[.025,.975]);couplings.append((field,len(yp),len(set(gids)),point,lo,hi,int(np.isfinite(sims).sum())))
    return {'intensity_scores':pd.DataFrame(scores,columns=['field','population','model','reference','rows','rmse','mae','rmse_difference','lower','upper']),
            'onset_intensity_association':pd.DataFrame(couplings,columns=['field','observed_positive','event_pairs','spearman_r','lower','upper','valid_bootstraps'])}


def scenario_rows(data,field,model,coef):
    out=[]; w=context_weights(data)
    for stim in STIMULI:
        for dist in (50,100,150):
            x=covariates(stim,np.array(CONTEXTS),dist,8,model);p,m=hurdle_parts(x,coef,field); y=p*m
            whole=data[(data.CallType==stim)&(data.Distance==dist)]
            cells=[whole[whole.Context==c] for c in CONTEXTS]
            pred=[(c,p[i],m[i],y[i],cells[i]) for i,c in enumerate(CONTEXTS)]
            prob=w@p;mean=w@y;pred.append(('standardized',prob,mean/prob,mean,whole))
            for c,prob,positive,mean,cell in pred:
                count=len(cell);events=int((cell[field]>0).sum());exact=cell[cell.Npenguins==8]
                out.append((field,model,stim,dist,8,c,count,events,len(exact),int((exact[field]>0).sum()),'no_trials' if not count else ('no_positive' if not events else 'positive_observed'),prob,positive,mean))
    return out


def run_intensity(data,latencies,smoke=False):
    grp=data.expID.to_numpy(); stimuli=data.CallType.to_numpy();N=len(data)
    folds=list(GroupKFold(5).split(data,groups=grp));rows=[];tuning=[];scenarios=[]
    for field,bi in FIELDS.items():
        y=data[field].to_numpy(float); assert np.isfinite(y).all() and (y>=0).all()
        for model in ('shared','call_slopes'):
            x=covariates(data.CallType,data.Context,data.Distance,data.Npenguins,model)
            for fold,(train,test) in enumerate(folds,1):
                assert not set(grp[train])&set(grp[test]);k,path=tune(x,y,field,grp,train)
                tuning.extend((field,model,fold,lam,e,lam==k) for lam,e in zip(PENALTIES,path))
                h=fit_hurdle(x[train],y[train],field,k);p,m=hurdle_parts(x[test],h,field)
                pred={model:(p,m)}
                if model=='shared':
                    prob=np.mean(y[train]>0);cond=y[train][y[train]>0].mean();pred['train_mean']=(np.full(len(test),prob),np.full(len(test),cond))
                    sp=np.empty(len(test)); sm=np.empty(len(test))
                    for c in STIMULI:
                        a=y[train][stimuli[train]==c];b=stimuli[test]==c
                        sp[b]=np.mean(a>0);sm[b]=a[a>0].mean() if (a>0).any() else cond
                    pred['stimulus_mean']=(sp,sm)
                    # Retrospective estimate only. No test latency is ever fed to a pre-playback forecast.
                    event_train=train[y[train]>0];event_test=test[y[test]>0]
                    z=np.column_stack([x,np.log1p(latencies[:,bi])]);assert np.isfinite(z[event_train]).all()
                    beta=level_fit(z[event_train],y[event_train],field,k)
                    post=np.full(len(test),np.nan);post[y[test]>0]=level_predict(z[event_test],beta,field)
                    pred['shared_after_observed_onset']=(np.full(len(test),np.nan),post)
                for name,(prob,pos) in pred.items():
                    rows.append(pd.DataFrame({'source_csv_line':test+2,'expID':grp[test],'fold':fold,'field':field,'model':name,'observed':y[test],'p_positive':prob,'positive':pos,'unconditional':prob*pos}))
                if smoke:break
            if not smoke:
                k,path=tune(x,y,field,grp,np.arange(N));tuning.extend((field,model,'full',lam,e,lam==k) for lam,e in zip(PENALTIES,path));scenarios.extend(scenario_rows(data,field,model,fit_hurdle(x,y,field,k)))
    oof=pd.concat(rows,ignore_index=True)
    if smoke:return {'intensity_oof':oof}
    assert not oof.duplicated(['source_csv_line','field','model']).any() and oof.groupby(['field','model']).size().eq(N).all()
    tables=score_and_association(data,oof,latencies)
    tables.update(intensity_oof=oof,intensity_tuning=pd.DataFrame(tuning,columns=['field','model','fold','penalty','inner_mse','selected']),
                  intensity_scenarios=pd.DataFrame(scenarios,columns=['field','model','stimulus','distance_m','group_size','context','observed_trials_at_distance','observed_positive_at_distance','observed_trials_exact_size','observed_positive_exact_size','support','p_positive','mean_if_positive','mean_among_all']))
    return tables


def figure(tab,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figures_apmcm import save
    plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','Noto Sans CJK SC','Hiragino Sans GB','DejaVu Sans'], 'axes.unicode_minus':False,'figure.constrained_layout.use':False, 'figure.autolayout':False,'svg.hashsalt':'v6-intensity'})
    labs=['抬头时长','刺激后叫声数','靠近时长','净增叫声数']
    fig,axs=plt.subplots(1,2,figsize=(6.55,2.85),layout='none')
    fig.subplots_adjust(left=.16,right=.97,bottom=.21,top=.72,wspace=.18)
    for ax,phase,title in zip(axs,['all','responders'],['a　全部刺激','b　有正响应强度者']):
        sub=tab[(tab.population==phase)&(tab.reference=='train_mean')]; ax.axvline(1.,ls='--',c='.55',lw=.85)
        for model,offset,marker,col,label in [('stimulus_mean',-.15,'o','#5b7281','声型均值'),('shared',0,'s','#096950','共享斜率'),('call_slopes',.15,'^','#a24922','声型交互')]:
            ratio=[]
            for field in FIELDS:
                q=sub[sub.field==field].set_index('model');ratio.append(q.loc[model,'rmse']/q.loc['train_mean','rmse'])
            ax.plot(ratio,np.arange(4)+offset,ls='',marker=marker,ms=4,color=col,label=label)
        ax.set_xlabel('RMSE / 训练均值基准 RMSE',fontsize=8);ax.set_title(title,fontsize=9,loc='left');ax.invert_yaxis();ax.set_yticks(range(4),labs if ax is axs[0] else []);ax.tick_params(labelsize=8)
        ax.grid(axis='x',lw=.4,alpha=.35);ax.spines[['top','right']].set_visible(False)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.99),ncol=3,fontsize=8,frameon=False)
    destination.mkdir(exist_ok=True,parents=True);save(fig,destination,'result_q3_intensity')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HERE/'results');parser.add_argument('--figures',type=Path,default=HERE/'figures');parser.add_argument('--smoke',action='store_true');parser.add_argument('--section',choices=['intensity','timing','all'],default='all');args=parser.parse_args()
    data=pd.read_csv(input_path(2));flags=event_flags(data);times=latency_values(data)
    assert data.groupby('expID').size().eq(2).all() and data.groupby('expID').Type.nunique().eq(2).all()
    for field,bi in FIELDS.items():
        hit=data[field].gt(0).to_numpy();assert np.isfinite(times[hit,bi]).all()
        if field!='VocalNcalls':np.testing.assert_array_equal(hit,flags[:,bi])
    tables={}
    if args.section in ('intensity','all'):tables.update(run_intensity(data,times,args.smoke))
    if args.section in ('timing','all'):tables.update(run_timing(data,flags,times,args.smoke))
    if args.smoke:
        print({k:len(v) for k,v in tables.items()});return
    args.output.mkdir(exist_ok=True,parents=True)
    for name,table in tables.items():table.to_csv(args.output/f'q3v6_{name}.csv',index=False,float_format='%.12g')
    if 'intensity_scores' in tables:figure(tables['intensity_scores'],args.figures)


if __name__=='__main__':main()
