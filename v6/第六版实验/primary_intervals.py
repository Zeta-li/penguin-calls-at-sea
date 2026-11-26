"""Whole-pair refits for the primary Q3-to-Q4 behavioural model, and completion sensitivity."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

import ecology as e
from models_acoustic import monotone_nodes
from penguin_data import DISTANCES, STIMULI, SEED
from q3_timing import state_fit, state_logp

HERE = Path(__file__).resolve().parent
BOOT_SEED = SEED + 444  # same pair draws as V4; every coefficient vector below is refitted at lambda=2
REPS = 400


def projection(beta, xx, weights):
    p=np.exp(state_logp(beta, xx))
    pooled=np.einsum('kzds,z->kds',p,weights)
    return p,pooled[1:]-pooled[0]


def location(scores, threshold):
    return [e.functional_radius(e.DESIGN,scores[c,:,j],threshold)[:2]
            for c in range(len(e.CALLS)) for j in range(2)]


def pair_refits(playback, fit, x, y, xx, weights, penalty, reps):
    group=np.array(list(playback.groupby('expID',sort=True).indices.values()))
    assert group.shape==(139,2)
    samples=np.random.default_rng(BOOT_SEED).integers(0,len(group),size=(reps,len(group)))
    f=e.aggregate(e.components(fit,e.DESIGN,'parametric'),'mean')
    specs=[]
    for scenario,tau,loss in e.SCENARIOS:
        _, h=e.gates(f,tau,loss,e.WIDTH)
        for mech,par in [('shared',e.ZERO),('both',e.STANDARD)]:
            specs.append((scenario,mech,h,e.kernel(loss,par) @ e.STATES[:,1:3]))
    rows=[]
    for i,draw in enumerate(samples):
        ids=group[draw].ravel()
        beta=state_fit(x[ids],y[ids],penalty)
        _,diff=projection(beta,xx,weights)
        for scenario,mech,h,u in specs:
            score=h[...,None]*(diff@u)
            for (ci,ei),(r,status) in zip(np.ndindex(len(e.CALLS),2),location(score,e.CUTOFF)):
                rows.append((i,scenario,mech,e.CALLS[ci],e.ENDPOINTS[ei],r,status,score[ci,e.AT100,ei]))
    result=pd.DataFrame(rows,columns=['draw','scenario','mechanism','call','endpoint','clipped_m','status','index100'])
    intervals=[]
    for key,g in result.groupby(['scenario','mechanism','call','endpoint'],sort=False):
        intervals.append((*key,BOOT_SEED,penalty,len(g),*np.quantile(g.clipped_m,[.025,.50,.975]),
                          int((g.status==e.STATUS_NONE).sum()),int((g.status==e.STATUS_CAP).sum()),
                          *np.quantile(g.index100,[.025,.975])))
    summary=pd.DataFrame(intervals,columns=['scenario','mechanism','call','endpoint','seed','fixed_penalty','paired_draws',
                  'clipped_q025_m','clipped_q50_m','clipped_q975_m','n_empty','n_cap','index100_q025','index100_q975'])
    return result,summary


def retention_bounds(lo,hi,tau,loss):
    """Extrema of the clipped ratio on each interval: endpoints plus its possible minimum."""
    _,hl=e.gates(lo,tau,loss,e.WIDTH); _,hu=e.gates(hi,tau,loss,e.WIDTH)
    lower=np.minimum(hl,hu); upper=np.maximum(hl,hu)
    if loss>0:
        # In x=exp((f-.5)/w), the derivative sign is (1-loss)*x - loss - c*x**(1-loss).
        # For c>0 it has one positive zero, a minimum of the ratio before clipping.
        def slope(f):
            return expit((f-.5)/e.WIDTH)-loss-(1-loss)*expit(((1-loss)*f-tau)/e.WIDTH)
        a=lo.min(); b=hi.max()
        if slope(a)*slope(b)<0:
            critical=brentq(slope,a,b); _,hc=e.gates(critical,tau,loss,e.WIDTH)
            lower=np.where((lo<=critical)&(critical<=hi),np.minimum(lower,hc),lower)
    for t in (0.,.25,.5,.75,1.):
        _, h=e.gates(lo+t*(hi-lo),tau,loss,e.WIDTH)
        assert np.all(lower<=h+1e-12) and np.all(upper>=h-1e-12)
    return lower,upper


def structural_tables(prob, diff, weights, fit, raw):
    _,_,lower,upper=e.summary_cells(raw)
    completion=[{'pava':monotone_nodes(n),'distances':DISTANCES} for n in (lower,upper)]
    models={k:e.components(fit,e.DESIGN,k) for k in ('pava','spline','parametric')}
    models['spline_sigmoid_equal']=(models['spline']+models['parametric'])/2
    rows=[]; bound_rows=[]; individuals=[]
    for criterion in ['mean','all']:
        fl,fu=[e.aggregate(e.components(ff,e.DESIGN,'pava'),criterion) for ff in completion]
        for name,tau,loss in e.SCENARIOS:
            hlo,hhi=retention_bounds(fl,fu,tau,loss)
            for mech,par in [('shared',e.ZERO),('both',e.STANDARD)]:
                k=e.kernel(loss,par); base=diff@(k@e.STATES[:,1:3]); a=hlo[...,None]*base; b=hhi[...,None]*base
                low=np.minimum(a,b); high=np.maximum(a,b)
                mid=e.gates(e.aggregate(models['pava'],criterion),tau,loss,e.WIDTH)[1][...,None]*base
                assert np.all(low<=mid+1e-12) and np.all(high>=mid-1e-12)
                rlo=location(low,e.CUTOFF); rmid=location(mid,e.CUTOFF); rhi=location(high,e.CUTOFF)
                for (c,j),(rl,sl),(rm,sm),(rh,sh) in zip(np.ndindex(len(e.CALLS),2),rlo,rmid,rhi):
                    assert rl<=rm+1e-8 and rm<=rh+1e-8
                    bound_rows.append((criterion,name,mech,e.CALLS[c],e.ENDPOINTS[j],rl,sl,rm,sm,rh,sh))
                for model,comp in models.items():
                    f=e.aggregate(comp,criterion); _,h=e.gates(f,tau,loss,e.WIDTH); scores=h[...,None]*base
                    for (c,j),(r,status) in zip(np.ndindex(len(e.CALLS),2),location(scores,e.CUTOFF)):
                        rows.append((criterion,name,mech,model,e.CALLS[c],e.ENDPOINTS[j],r,status,scores[c,e.AT100,j]))
                    if model=='parametric' and criterion=='mean':
                        pn,pc=e.distributions(prob,weights,h,k);u=np.column_stack([e.STATES[:,1:3],e.STATES.any(axis=1)])
                        for d in (50,100,150):
                            j=np.where(e.DESIGN==d)[0].item(); val=pn[:,j]@u; ctrl=pc[j]@u
                            for c,call in enumerate(e.CALLS):
                                for z,endpoint in enumerate((*e.ENDPOINTS[:2],'any_flag')):
                                    p=val[c,z]; individuals.append((name,mech,call,d,endpoint,p,ctrl[z],p-ctrl[z],p,8*p,4.5*p))
    specification=pd.DataFrame(rows,columns=['criterion','scenario','mechanism','model','call','endpoint','clipped_m','status','index100'])
    bounds=pd.DataFrame(bound_rows,columns=['criterion','scenario','mechanism','call','endpoint','completion_lower_m','lower_status',
               'pava_measured_m','measured_status','completion_upper_m','upper_status'])
    people=pd.DataFrame(individuals,columns=['scenario','mechanism','call','distance_m','endpoint','p_natural','p_control','difference',
                                              'expected_min','expected_max','expected_if_half_others'])
    return specification,bounds,people


def main_figure(point, intervals, people, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style('en','report')
    os.environ.setdefault('SOURCE_DATE_EPOCH','1795651200')
    plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','Noto Sans CJK SC','Hiragino Sans GB','DejaVu Sans'],
                         'svg.hashsalt':'q4-v6-primary-pair','figure.autolayout':False,'figure.constrained_layout.use':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.05,3.0),layout='none')
    fig.subplots_adjust(left=.09,bottom=.19,right=.985,top=.85,wspace=.36)
    criteria="scenario=='Threshold 0.70' and mechanism=='shared'"
    ordered=point.query(criteria+" and criterion=='mean' and model=='parametric'").set_index(['call','endpoint'])
    ci=intervals.query(criteria).set_index(['call','endpoint'])
    colors=['#176c7c','#ad5e28']; labels=['联系','招募']
    for c,call in enumerate(e.CALLS):
        for j,k in enumerate(e.ENDPOINTS[:2]):
            d=ordered.loc[call,k]; z=ci.loc[call,k]; x=c+.17*(j-.5)
            ax.vlines(x,z.clipped_q025_m,z.clipped_q975_m,color=colors[j],lw=1.2)
            ax.scatter(x,d.clipped_m,marker='^' if d.status==e.STATUS_CAP else 'o',color=colors[j],s=28,clip_on=False)
    ax.set(xticks=[0,1,2],xticklabels=['平直型','调制型','双声型'],xlim=(-.38,2.38),ylim=(15,158),
           yticks=[20,50,75,100,125,150],ylabel='连续达标半径（m）')
    ax.set_title('a　阈值 0.70，无额外损失',loc='left',fontsize=10)
    ax.legend(handles=[Line2D([],[],marker='o',ls='none',color=c,label=l) for c,l in zip(colors,labels)],
              ncol=2,frameon=False,fontsize=7,loc='upper left')
    d=people.query("call=='tv' and distance_m==100 and scenario=='Fidelity loss 20%' and endpoint=='any_flag'").set_index('mechanism')
    xx=np.arange(3); width=.34
    for j, key in enumerate(('shared','both')):
        bx.bar(xx+width*(j-.5),100*d.loc[key,['p_natural','p_control','difference']].to_numpy(float),width,
               color=colors[j],hatch='' if j==0 else '//',edgecolor='white',linewidth=.5,label=('仅相对门控','警戒与抑制')[j])
    bx.set(xticks=xx,xticklabels=['自然声','海浪对照','两者差'],ylim=(0,70),ylabel='任一行为概率或其差（%）')
    bx.set_title('b　双声型，100 m，损失 20%',loc='left',fontsize=10)
    bx.legend(frameon=False,loc='upper right',fontsize=7)
    output.mkdir(parents=True,exist_ok=True);save(fig,output,'result_q4_primary_range');plt.close(fig)


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--figures',type=Path,default=HERE/'figures');p.add_argument('--no-figures',action='store_true')
    p.add_argument('--q3-results',type=Path,default=HERE/'results');p.add_argument('--replicates',type=int,default=REPS)
    args=p.parse_args()
    if args.replicates<1:p.error('--replicates must be positive')
    _,playback,corr=e.read_inputs();weights=e.context_weights(playback)
    x=e.design(playback.CallType,playback.Context,playback.Distance,playback.Npenguins);y=e.state_index(e.event_flags(playback))
    xx=e.design(np.asarray(STIMULI)[:,None,None],np.asarray(e.CONTEXTS)[None,:,None],e.DESIGN[None,None,:],8)
    tt=pd.read_csv(args.q3_results/'q3v6_timing_tuning.csv');ss=tt.query("model=='shared' and endpoint=='net_response' and component=='state' and selected")
    penalty=float(ss[ss.fold.astype(str)=='full'].penalty.item());beta=state_fit(x,y,penalty)
    q3=pd.read_csv(args.q3_results/'q3v6_state_parameters.csv').query("model=='shared' and endpoint=='net_response'").sort_values('index')
    np.testing.assert_allclose(beta,q3.value,rtol=2e-5,atol=2e-6)
    prob,diff=projection(beta,xx,weights);fit=e.fit_primary(e.summary_cells(corr)[1])
    specs,bounds,people=structural_tables(prob,diff,weights,fit,corr)
    _,grid=e.joint_grid(prob,weights,e.aggregate(e.components(fit,e.DESIGN,'parametric'),'mean'))
    draws,summary=pair_refits(playback,fit,x,y,xx,weights,penalty,args.replicates)
    tables={'specifications':specs,'pava_completion':bounds,'responders':people,'joint_summary':grid,
            'paired_draws':draws,'paired_intervals':summary}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(args.output/f'q4v6_primary_{name}.csv',index=False,float_format='%.12g')
    if not args.no_figures:main_figure(specs,summary,people,args.figures)
    print('Whole-pair refits:',args.replicates,'seed:',BOOT_SEED,'penalty:',penalty,flush=True)


if __name__=='__main__':main()
