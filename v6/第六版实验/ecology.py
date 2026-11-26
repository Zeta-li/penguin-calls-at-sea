"""Relative changes in controlled sea-surface playback contrasts under hypothetical extra masking."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / '第四版实验'
V5 = ROOT / '第五版实验'
sys.path[:0] = [str(ROOT), str(V4)]
from analyze_effectiveness import DESIGN, probabilities
from models_effectiveness import ECO_WEIGHTS, SCENARIOS, ecological_values, functional_radius
from models_playback import design, response_fit
from penguin_data import CALLS, CONTEXTS, STATES, context_weights, event_flags, read_inputs, state_index
from propagation_v4 import aggregate, components, fit_primary, summary_cells

ENDPOINTS = ('contact_net_vocal', 'recruitment_approach', 'context_composite')
UTILITY = np.stack((np.tile(STATES[:, 1], (3, 1)), np.tile(STATES[:, 2], (3, 1)), ecological_values()), axis=-1)
ZERO = (0., 0., 0.)
STANDARD = (.20, .15, .40)  # assumed H-on, V-off, A-off probabilities at 20% loss
WIDTH = .05
CUTOFF = .05
AT100 = np.flatnonzero(DESIGN == 100).item()
STATUS_NONE = 'no_segment_at_lower_bound'
STATUS_CAP = 'right_boundary'
PRIMARY_MODEL = 'parametric'


def kernel(loss, constants):
    h, v, a = np.asarray(constants) * (loss/.20)
    one = np.column_stack((STATES[:, 0] + (1 - STATES[:, 0])*h,
                           STATES[:, 1]*(1-v), STATES[:, 2]*(1-a)))
    k = np.where(STATES[None, :, :], one[:, None, :], 1-one[:, None, :]).prod(axis=-1)
    assert k.min() >= 0
    np.testing.assert_allclose(k.sum(axis=1), 1., atol=1e-14)
    return k


def gates(fidelity, tau, loss, width):
    base = expit((fidelity-.50)/width)
    h = np.minimum(1., expit(((1-loss)*fidelity-tau)/width) / base)
    if tau == .50 and loss == 0:
        np.testing.assert_array_equal(h, np.ones_like(h))
    return base, h


def benefit(diff, k, utility=UTILITY):
    # diff[call, context, distance, state] already includes observed context weights.
    return np.einsum('czds,zse->cde', diff, k @ utility, optimize=True)


def distributions(prob, weights, retention, k):
    natural = (prob[0] + retention[:, None, :, None]*(prob[1:]-prob[0])) @ k
    waves = prob[0] @ k
    for dist in [natural, waves]:
        assert dist.min() >= -1e-12
        np.testing.assert_allclose(dist.sum(axis=-1), 1., atol=2e-12)
    return np.einsum('czds,z->cds', natural, weights), np.einsum('zds,z->ds', waves, weights)


def tables_for_specs(prob, weights, fidelities):
    diff = (prob[1:] - prob[0]) * weights[None, :, None, None]
    rows, at_distance, responders = [], [], []
    baseline_benefit = benefit(diff, np.eye(8))
    for (model, criterion), fidelity in fidelities.items():
        for name, tau, loss in SCENARIOS:
            g0, h = gates(fidelity, tau, loss, WIDTH)
            for mech, constants in [('shared', ZERO), ('both', STANDARD)]:
                k = kernel(loss, constants)
                new = h[..., None]*benefit(diff, k)
                old = g0[..., None]*new
                if name == 'Baseline':
                    np.testing.assert_allclose(new, baseline_benefit, atol=1e-13)
                if model == PRIMARY_MODEL and criterion == 'mean':
                    natural, waves = distributions(prob, weights, h, k)
                    # For behavioural end points (not the composite), the new index must equal
                    # the natural-minus-waves marginal constructed directly from eight states.
                    direct = (natural-waves) @ STATES[:, 1:3]
                    np.testing.assert_allclose(new[:, :, :2], direct, atol=3e-13)
                    signals = np.column_stack((STATES[:, 1:3], STATES.any(axis=1)))
                    for ci, call in enumerate(CALLS):
                        for d in (50, 100, 150):
                            j = np.flatnonzero(DESIGN == d).item()
                            for e, label in enumerate((*ENDPOINTS[:2], 'any_flag')):
                                pn = float(natural[ci,j] @ signals[:,e]); pc = float(waves[j] @ signals[:,e])
                                responders.append((name, mech, call, d, label, pn, pc, pn-pc, pn, 8*pn, 4.5*pn))
                        for j in range(0, len(DESIGN), 4):
                            at_distance.append((name, mech, call, DESIGN[j],fidelity[ci,j],g0[ci,j],h[ci,j],
                                                *old[ci,j], *new[ci,j]))
                for c, call in enumerate(CALLS):
                    for e, endpoint in enumerate(ENDPOINTS):
                        ro, so, _ = functional_radius(DESIGN, old[c,:,e], CUTOFF)
                        rn, sn, later = functional_radius(DESIGN, new[c,:,e], CUTOFF)
                        rows.append((model,criterion,name,tau,loss,mech,call,endpoint,ro,so,rn,sn,later,
                                     old[c,AT100,e],new[c,AT100,e]))
    keys = ['model','criterion','scenario','tau','loss','mechanism','call','endpoint']
    results = pd.DataFrame(rows,columns=keys+['v5_clipped_m','v5_status','v6_clipped_m','v6_status','v6_later_segment',
                                             'v5_index100','v6_index100'])
    reference = pd.read_csv(V5 / 'results/q4v5_scenarios.csv')
    keys = ['model','criterion','scenario','mechanism','call','endpoint']
    joined = results.merge(reference, on=keys,validate='one_to_one')
    assert len(joined) == len(results) == len(reference) == 1008
    np.testing.assert_allclose(joined.v5_clipped_m, joined.clipped_radius_m, rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(joined.v5_index100, joined.index_100m, rtol=1e-7, atol=2e-8)
    np.testing.assert_array_equal(joined.v5_status,joined.status)
    curves = pd.DataFrame(at_distance, columns=['scenario','mechanism','call','distance_m','fidelity','g0','retention',
                                                 *['v5_'+e for e in ENDPOINTS], *['v6_'+e for e in ENDPOINTS]])
    people = pd.DataFrame(responders, columns=['scenario','mechanism','call','distance_m','endpoint','p_natural',
            'p_control','difference','expected_min','expected_max','expected_if_half_others'])
    return results, curves, people


def bounds_by_spec(detail):
    keys = ['criterion','scenario','call','endpoint']
    rows = []
    for group, data in detail.groupby(keys, sort=False):
        row = list(group)
        for version in ('v5','v6'):
            r = data[f'{version}_clipped_m']; s = data[f'{version}_status']
            lo = r.min(); left = (s==STATUS_NONE); cap = (s==STATUS_CAP)
            state = STATUS_NONE if left.any() else STATUS_CAP if cap.all() else 'crossing'
            row.extend([lo, state, r.max(), int(left.sum()), int(cap.sum())])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=keys+[v+'_'+field for v in ('v5','v6') for field in
                                             ['common_clipped_m','common_status','largest_clipped_m','n_empty','n_cap']])
    original = pd.read_csv(V5/'results/q4v5_joint_support.csv')
    matched = frame.merge(original, on=keys,validate='one_to_one')
    assert len(matched) == len(frame) == len(original)
    np.testing.assert_allclose(matched.v5_common_clipped_m, matched.common_clipped_radius_m, rtol=1e-7)
    np.testing.assert_array_equal(matched.v5_common_status, matched.common_status)
    return frame


def joint_grid(prob, weights, f):
    """Counts on a deterministic grid, without sampling weights for structural assumptions."""
    widths = (.025,.05,.10); cutoffs = (.025,.05,.10); scales = (0.,1.,1.5)
    pooled = np.einsum('kzds,z->kds',prob,weights)
    diff = pooled[1:]-pooled[0]
    anyflag = STATES.any(axis=1).astype(float)
    rows=[]
    for scenario, tau, loss in SCENARIOS:
        for width in widths:
            _, h = gates(f,tau,loss,width)
            before = pooled[0] + h[...,None]*diff
            for sc in product(scales,repeat=3):
                const = np.array(STANDARD)*np.array(sc)
                k = kernel(loss,const)
                score = h[...,None]*(diff @ (k @ STATES[:,1:3]))
                pn = before[:,AT100] @ (k @ anyflag)
                pc = pooled[0,AT100] @ (k @ anyflag)
                # Two contrast changes with the same h and a shared (identity) mechanism.
                nat_change = pn - before[:,AT100] @ anyflag
                excess_change = nat_change - (pc - pooled[0,AT100] @ anyflag)
                for cutoff in cutoffs:
                    for c,call in enumerate(CALLS):
                        pair = [functional_radius(DESIGN, score[c,:,j], cutoff) for j in range(2)]
                        rows.append((scenario,tau,loss,width,cutoff,*const,call,pair[0][0],pair[0][1],pair[1][0],pair[1][1],
                                     nat_change[c],excess_change[c]))
    grid = pd.DataFrame(rows, columns=['scenario','tau','loss','width','utility_cutoff','heads_on_at20',
                       'vocal_off_at20','approach_off_at20','call','contact_m','contact_status','recruitment_m',
                       'recruitment_status','natural_any_change100','net_any_change100'])
    summary=[]
    for (name,call),g in grid.groupby(['scenario','call'],sort=False):
        strict = g.contact_m>g.recruitment_m+1e-8
        reverse = g.contact_m<g.recruitment_m-1e-8
        para = (g.natural_any_change100>1e-12)&(g.net_any_change100<-1e-12)
        for endpoint in ['contact','recruitment']:
            radius = g[endpoint+'_m']; status=g[endpoint+'_status']
            summary.append((name,call,endpoint,len(g),radius.min(),radius.max(),int((status==STATUS_NONE).sum()),
                            int((status==STATUS_CAP).sum()),int(strict.sum()),int(reverse.sum()),int(para.sum())))
    cols=['scenario','call','endpoint','grid_n','min_clipped_m','max_clipped_m','n_empty','n_cap','n_contact_longer',
          'n_recruitment_longer','n_any_up_net_down']
    return grid,pd.DataFrame(summary,columns=cols)


def utility_sensitivity(prob, weights, f):
    diff=(prob[1:]-prob[0])*weights[None,:,None,None]
    patterns={'context_reference':ECO_WEIGHTS, 'equal':np.full((3,3),1/3),
              'contact_emphasis':np.tile([.15,.70,.15],(3,1)),
              'recruitment_emphasis':np.tile([.15,.15,.70],(3,1))}
    output=[]
    for nm,w in patterns.items():
        for bonus in [0.,.2,.4]:
            u=ecological_values(w,bonus)[...,None]
            for name,tau,loss in SCENARIOS:
                _, h=gates(f,tau,loss,WIDTH)
                for mech,params in [('shared',ZERO),('both',STANDARD)]:
                    score=h*benefit(diff,kernel(loss,params),u)[...,0]
                    for i,call in enumerate(CALLS):
                        rad,status,_=functional_radius(DESIGN,score[i],CUTOFF)
                        output.append((nm,bonus,name,mech,call,rad,status,score[i,AT100]))
    return pd.DataFrame(output,columns=['weight_profile','bonus','scenario','mechanism','call','clipped_m','status','index100'])


def behavioral_slope_sensitivity(weights, f, original_probability, results, playback):
    from q3_timing import covariates, state_logp, state_fit
    from penguin_data import STIMULI
    coefficients = pd.read_csv(results/'q3v6_state_parameters.csv')
    tuning = pd.read_csv(results/'q3v6_timing_tuning.csv')
    tuned = tuning[(tuning.fold.astype(str)=='full') & (tuning.endpoint=='net_response') &
                   (tuning.component=='state') & tuning.selected].set_index('model').penalty
    probabilities_by_design={'v5_shared_fixed_0.5':original_probability}
    for model in ('shared','call_slopes'):
        coeff = coefficients[(coefficients.model==model)&(coefficients.endpoint=='net_response')].sort_values('index').value.to_numpy()
        x = covariates(np.array(STIMULI)[:,None,None], np.array(CONTEXTS)[None,:,None], DESIGN[None,None,:], 8, model)
        probabilities_by_design[f'{model}_cv_{tuned[model]:g}']=np.exp(state_logp(coeff, x))
        if model=='call_slopes':
            dm=covariates(playback.CallType,playback.Context,playback.Distance,playback.Npenguins,model)
            b=state_fit(dm,state_index(event_flags(playback)),tuned['shared'])
            probabilities_by_design[f'call_slopes_same_penalty_{tuned["shared"]:g}']=np.exp(state_logp(b,x))
    output=[]
    for label,prob in probabilities_by_design.items():
        np.testing.assert_allclose(prob.sum(axis=-1),1.,atol=1e-12)
        diff = (prob[1:]-prob[0])*weights[None,:,None,None]
        for name,tau,loss in SCENARIOS:
            _,h=gates(f,tau,loss,WIDTH)
            for mechanism,params in [('shared',ZERO),('both',STANDARD)]:
                score=h[...,None]*benefit(diff,kernel(loss,params))
                for c,call in enumerate(CALLS):
                    for j,endpoint in enumerate(ENDPOINTS[:2]):
                        r,status,_=functional_radius(DESIGN,score[c,:,j],CUTOFF)
                        output.append((label,name,mechanism,call,endpoint,r,status,score[c,AT100,j]))
    return pd.DataFrame(output,columns=['behavioral_design','scenario','mechanism','call','endpoint','clipped_m','status','index100'])


def paired_uncertainty(weights, primary, playback):
    path=V4/'results/q4_paired_bootstrap_draws.npz'
    with np.load(path,allow_pickle=False) as archived:
        coefs=archived['fitted_coefficients']
        np.testing.assert_array_equal(archived['pair_ids'], np.sort(playback.expID.unique()))
        draws=archived['sampled_pair_indices']
        assert draws.shape == (len(coefs),139) and draws.min()>=0 and draws.max()<139
    options=[]
    for name,tau,loss in SCENARIOS:
        _, h=gates(primary,tau,loss,WIDTH)
        for mech,params in [('shared',ZERO),('both',STANDARD)]:
            options.append((name,mech,h,kernel(loss,params) @ STATES[:,1:3]))
    rows=[]
    for b,beta in enumerate(coefs):
        p=probabilities(beta)
        diff=np.einsum('czds,z->cds',p[1:]-p[0],weights)
        for name,mech,h,us in options:
            scores=h[...,None]*(diff@us)
            for c,call in enumerate(CALLS):
                for e,endpoint in enumerate(ENDPOINTS[:2]):
                    r,status,_=functional_radius(DESIGN,scores[c,:,e],CUTOFF)
                    rows.append((b,name,mech,call,endpoint,r,status,scores[c,AT100,e]))
    detail=pd.DataFrame(rows,columns=['draw','scenario','mechanism','call','endpoint','clipped_m','status','index100'])
    output=[]
    for key,part in detail.groupby(['scenario','mechanism','call','endpoint'],sort=False):
        output.append((*key,len(part),*np.quantile(part.clipped_m,[.025,.5,.975]),
                       (part.status==STATUS_NONE).sum(),(part.status==STATUS_CAP).sum(),
                       *np.quantile(part.index100,[.025,.975])))
    summary=pd.DataFrame(output,columns=['scenario','mechanism','call','endpoint','paired_draws','clipped_q025_m',
        'clipped_q50_m','clipped_q975_m','n_empty','n_cap','index100_q025','index100_q975'])
    return detail,summary


def make_figure(curves, people, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style('en','report')
    plt.rcParams.update({'font.sans-serif':['Arial Unicode MS','Noto Sans CJK SC','Hiragino Sans GB','DejaVu Sans'],
                         'svg.hashsalt':'q4-relative-v6','figure.autolayout':False,'figure.constrained_layout.use':False})
    os.environ.setdefault('SOURCE_DATE_EPOCH','1795651200')
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.95),layout='none')
    fig.subplots_adjust(left=.08,bottom=.19,right=.985,top=.85,wspace=.33)
    ax=axes[0]; data=curves.query("call=='flat' and scenario=='Baseline' and mechanism=='shared'")
    ax.plot(data.distance_m,data.v6_contact_net_vocal,color='#177280',lw=2,label='海上净回叫（第六版）')
    ax.plot(data.distance_m,data.v5_contact_net_vocal,color='#925328',lw=1.75,ls='--',label='额外乘基准门（第五版）')
    ax.axhline(CUTOFF,color='#555555',ls=':',lw=1,label='操作门槛 0.05')
    ax.set(xlim=(20,150),ylim=(0,None),xticks=[20,50,100,150],xlabel='距离（m）',ylabel='自然声减海浪的功能指标')
    ax.set_title('a　平直型基准情景',fontsize=10,loc='left'); ax.legend(frameon=False,fontsize=7,loc='best')
    ax=axes[1]; data=people.query("call=='tv' and distance_m==100 and scenario=='Fidelity loss 20%' and endpoint=='any_flag'").set_index('mechanism')
    xx=np.arange(3); w=.34; cols=['p_natural','p_control','difference']
    for i,(key,label,col) in enumerate([('shared','仅相对门控','#177280'),('both','增加警戒与选择性抑制','#b96c38')]):
        ax.bar(xx+(i-.5)*w,data.loc[key,cols].to_numpy(float)*100,w,color=col,label=label,hatch='' if i==0 else '//',edgecolor='white',linewidth=.4)
    ax.set(xticks=xx,xticklabels=['自然声','海浪对照','两者差'],ylabel='任一行为概率或其差（%）',ylim=(0,70))
    ax.set_title('b　双声型，100 m，损失 20%',fontsize=10,loc='left'); ax.legend(frameon=False,fontsize=7,loc='upper right')
    output.mkdir(parents=True,exist_ok=True)
    save(fig,output,'result_q4_relative_gate')
    plt.close(fig)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--figures',type=Path,default=HERE/'figures')
    parser.add_argument('--q3-results',type=Path,default=HERE/'results',help='Q3 state coefficient table directory.')
    parser.add_argument('--no-figures',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    _,playback,corr=read_inputs(); weights=context_weights(playback)
    x=design(playback.CallType,playback.Context,playback.Distance,playback.Npenguins)
    beta=response_fit(x,state_index(event_flags(playback)),.5)
    prob=probabilities(beta)
    fit=fit_primary(summary_cells(corr)[1])
    comps={model:components(fit,DESIGN,model) for model in ['pava','spline','parametric']}
    comps['spline_sigmoid_equal']=(comps['spline']+comps['parametric'])/2
    fidelities={(name,crit):aggregate(comp,crit) for name,comp in comps.items() for crit in ['mean','all']}
    radii,curves,people=tables_for_specs(prob,weights,fidelities)
    bounds=bounds_by_spec(radii)
    if args.smoke:
        print('Validated 1008 matched V5 scenarios, 126 intersections and all baseline equalities.')
        return
    main_f=fidelities[PRIMARY_MODEL,'mean']; grid,grid_summary=joint_grid(prob,weights,main_f)
    resamples,intervals=paired_uncertainty(weights,main_f,playback)
    tables=dict(radii=radii, model_envelope=bounds, curves=curves, responders=people,
                joint_grid=grid,joint_summary=grid_summary,utility_mix=utility_sensitivity(prob,weights,main_f),
                paired_draws=resamples,paired_intervals=intervals,
                behavior_slopes=behavioral_slope_sensitivity(weights,main_f,prob,args.q3_results,playback))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():
        frame.to_csv(args.output/f'q4v6_{name}.csv',index=False,float_format='%.12g')
    if not args.no_figures:make_figure(curves,people,args.figures)
    print({name:len(frame) for name,frame in tables.items()})


if __name__ == '__main__':
    main()
