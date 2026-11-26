"""Date-restricted distribution tests, cross-fitted cue salience, and component-wise acoustic ranges."""
import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / '第四版实验'), str(ROOT / '第五版实验')]
import classical
from penguin_data import CALLS, DISTANCES, FEATURES, METRICS, SEED, read_inputs
from propagation_v4 import components, completion_bounds, fit_primary, invert, summary_cells

PERMUTATIONS = 19999
KS_SEED = SEED + 601
THRESHOLDS = (.5, .6, .7)
V5 = ROOT / '第五版实验/results'


def conditional_ks(data, draws=PERMUTATIONS):
    """Max absolute difference between same-date, common-weight empirical CDFs. Ties use right limits."""
    rng = np.random.default_rng(KS_SEED)
    rows = []
    for first, second in combinations(CALLS, 2):
        pair = data[data.CallType.isin([first, second])].copy()
        counts = pair.groupby('Date').CallType.nunique()
        pair = pair[pair.Date.isin(counts[counts == 2].index)].reset_index(drop=True)
        coeff = np.empty((draws + 1, len(pair)))
        weight_sum = 0.
        for _, date in pair.groupby('Date'):
            indices = date.index.to_numpy()
            reference = date.CallType.eq(first).to_numpy()
            n, a = len(date), reference.sum()
            b = n - a
            weight = a * b / n
            weight_sum += weight
            # Each trial chooses a uniform subset of exactly a records from this date.
            permuted = np.argsort(rng.random((draws, n)), axis=1)[:, :a]
            is_first = np.zeros((draws, n), dtype=bool)
            np.put_along_axis(is_first, permuted, True, axis=1)
            coeff[:, indices] = weight * np.where(np.vstack([reference, is_first]), 1/a, -1/b)
        coeff /= weight_sum
        for feature in FEATURES:
            x = pair[feature].to_numpy()
            order = np.argsort(x, kind='stable')
            ends = np.r_[np.diff(x[order]) != 0, True]
            d = np.abs(np.cumsum(coeff[:, order], axis=1)[:, ends]).max(axis=1)
            # Tie tolerance is well below attainable empirical differences.
            exceed = int(np.sum(d[1:] >= d[0] - 1e-12))
            rows.append((first, second, feature, pair.Date.nunique(), len(pair), d[0], exceed,
                         draws, KS_SEED, (exceed + 1)/(draws + 1)))
    frame = pd.DataFrame(rows, columns=['class_a','class_b','feature','dates','n','ks','as_extreme',
                                        'permutations','seed','p_mc'])
    order = np.argsort(frame.p_mc.to_numpy(), kind='stable')
    adjusted = np.maximum.accumulate(frame.p_mc.to_numpy()[order] * np.arange(len(frame), 0, -1))
    frame.loc[order, 'p_holm'] = np.minimum(1., adjusted)
    return frame


def classification_summary():
    dates = pd.read_csv(V5/'q1v5_predictions.csv')
    year_f = pd.read_csv(V5/'q1t_predictions.csv').rename(columns={'prediction':'decision'})
    year_c = pd.read_csv(V5/'q1c_temporal_oof.csv')
    methods = {'raw_empirical','raw_balanced','blend_balanced','classical_nested','ordinary_decision_only'}
    result = []
    for sample, data in [('date_holdout',dates), ('year_forward',pd.concat([year_f, year_c]))]:
        for method, group in data[data.method.isin(methods)].groupby('method'):
            truth = pd.Categorical(group.actual, categories=CALLS).codes
            pred = pd.Categorical(group.decision, categories=CALLS).codes
            assert len(set(group.row)) == len(group)
            prob = group[classical.PROB].to_numpy()
            count = np.bincount(truth, minlength=3)
            hit = np.bincount(truth[truth == pred], minlength=3)
            flagged = np.bincount(pred, minlength=3)
            mean_nll = -np.log(prob[np.arange(len(truth)),truth]).mean()
            result.append((sample, method, len(truth), group.date.nunique(), np.mean(hit/count),
                           hit.sum()/len(truth), mean_nll, hit[2], count[2], flagged[2], hit[2]/count[2],
                           hit[2]/flagged[2] if flagged[2] else np.nan))
    return pd.DataFrame(result, columns=['validation','method','n','dates','balanced_accuracy','accuracy',
            'log_loss','tv_true_positive','tv_true','tv_predicted','tv_recall','tv_precision'])


def cue_importances(data):
    """Exact expectation over a test day's possible donors for each held recording and cue block."""
    day = pd.to_datetime(data.Date,format='%d/%m/%Y').dt.strftime('%Y-%m-%d').to_numpy()
    x = classical.extract(data)['acoustic']
    y = pd.Categorical(data.CallType, categories=CALLS).codes.astype(int)
    selection = pd.read_csv(V5/'q1c_selection.csv').query("method=='classical_nested'").set_index('held_date')
    archive = pd.read_csv(V5/'q1c_oof.csv').query("method=='classical_nested'").sort_values('row')
    np.testing.assert_array_equal(day, archive.date)
    records = []
    for date in np.unique(day):
        train, test = day != date, np.flatnonzero(day == date)
        spec = selection.loc[date]
        assert spec.candidate.startswith('trees_acoustic_') and spec.decision_rule == 'balanced'
        leaf = int(spec.candidate.rsplit('_',1)[1])
        model = ExtraTreesClassifier(n_estimators=128,min_samples_leaf=leaf,max_features=1.,
                                     random_state=classical.SEED,n_jobs=1).fit(x[train],y[train])
        priors = classical.class_prior(y[train])
        def predict(query):
            return .98 * model.predict_proba(query) + .02 * priors
        base = predict(x[test])
        np.testing.assert_allclose(base, archive.iloc[test][classical.PROB], atol=1e-11, rtol=0)
        base_loss = -np.log(base[np.arange(len(test)),y[test]])
        recipients, donors = np.repeat(test,len(test)), np.tile(test,len(test))
        for name, cols in [('duration',[0]),('spectrum',list(range(1,x.shape[1])))]:
            altered = x[recipients].copy()
            altered[:,cols] = x[donors][:,cols]
            p = predict(altered)
            loss = -np.log(p[np.arange(len(p)), y[recipients]]).reshape(len(test),len(test)).mean(axis=1)
            records.extend((int(j)+1,date,CALLS[y[j]],len(test),name,leaf,a,b,b-a)
                           for j,a,b in zip(test,base_loss,loss))
    return pd.DataFrame(records,columns=['row','date','actual','n_donors','cue','leaf','original_nll',
                                         'donor_expected_nll','delta_nll'])


def representation_weights(importance):
    """Keep the joint spectrogram at 1/3 and divide the other 2/3 by the two cue saliences."""
    salience = importance.groupby('cue').delta_nll.mean().clip(lower=0)
    fraction = salience['spectrum']/salience.sum() if salience.sum() else .5
    equal = np.ones(3)/3
    weighted = np.array([1/3,(2/3)*fraction,(2/3)*(1-fraction)])
    return {'mean':equal,'class_shrunk':(equal+weighted)/2,'class_weighted':weighted}


def reduce_parts(parts, criterion, weights):
    return parts.min(axis=1) if criterion == 'all' else np.einsum('kjd,j->kd',parts,weights[criterion])


def measurement_bracket(observed, threshold):
    passing = np.flatnonzero(observed >= threshold)
    reversals = np.count_nonzero(np.diff((observed >= threshold).astype(int)) == 1)
    if len(passing) == 0:
        return np.nan, DISTANCES[0], 'below_4m', reversals
    last = passing[-1]
    if last == len(DISTANCES)-1:
        return DISTANCES[last], np.nan, 'right_boundary', reversals
    return DISTANCES[last], DISTANCES[last+1], 'crossing', reversals


def propagation(raw, weights):
    valid, means, lower, upper = summary_cells(raw)
    full = fit_primary(means)
    model_names = ['parametric','pava','spline']
    parameters, ranges, held, nodes = [], [], [], []
    for k,call in enumerate(CALLS):
        for j,metric in enumerate(METRICS):
            a, logb, c = full['parametric'][k,j]
            parameters.append((call,metric,a,np.exp(logb),c,valid[k,j].min()))
    for criterion in [*weights, 'all']:
        target = reduce_parts(means, criterion, weights)
        def curve(d):
            return reduce_parts(components(full,d,'parametric'),criterion,weights)
        estimate = curve(DISTANCES)
        for k,call in enumerate(CALLS):
            nodes.extend((call,criterion,d,o,v) for d,o,v in zip(DISTANCES,target[k],estimate[k]))
            for tau in THRESHOLDS:
                radius,status,left,right = invert(lambda d: curve(d)[k],tau,estimate[k])
                support = measurement_bracket(target[k],tau)
                limiting = METRICS[np.argmin(components(full,[radius],'parametric')[k,:,0])]
                ranges.append((call,criterion,tau,radius,status,left,right,*support,limiting))
    for h in range(1,len(DISTANCES)-1):
        keep = np.arange(len(DISTANCES)) != h
        fit = fit_primary(means[:,:,keep],DISTANCES[keep])
        for model in model_names:
            pred = components(fit,[DISTANCES[h]],model)
            for criterion in [*weights,'all']:
                observed = reduce_parts(means[:,:,h:h+1],criterion,weights)[:,0]
                prediction = reduce_parts(pred,criterion,weights)[:,0]
                held.extend((call,DISTANCES[h],criterion,model,o,p,p-o) for call,o,p in zip(CALLS,observed,prediction))
    held = pd.DataFrame(held,columns=['call','held_distance_m','criterion','model','observed','predicted','error'])
    scores = []
    for (criterion,model),g in held.groupby(['criterion','model']):
        for call,block in [('pooled',g),*g.groupby('call')]:
            scores.append((criterion,model,call,len(block),(block.error**2).mean()**.5,block.error.abs().mean()))
    # A pairwise ranking changes with threshold only where the two monotone component-aggregates intersect.
    intersections = []
    grid = np.geomspace(DISTANCES[0],DISTANCES[-1],513)
    for criterion in [*weights,'all']:
        for model in model_names:
            def fidelity(d):
                return reduce_parts(components(full,np.atleast_1d(d),model),criterion,weights)
            curves = fidelity(grid)
            for a,b in combinations(range(3),2):
                diff = curves[a]-curves[b]
                for i in np.flatnonzero(diff[1:]*diff[:-1] < 0):
                    def difference(d):
                        values = fidelity(d)[:,0]
                        return values[a]-values[b]
                    d = brentq(difference,grid[i],grid[i+1]); f = fidelity(d)[a,0]
                    intersections.append((model,criterion,CALLS[a],CALLS[b],d,f))
    return {'parameters':pd.DataFrame(parameters,columns=['call','representation','a','b_m','c','min_valid_slot_count']),
            'ranges':pd.DataFrame(ranges,columns=['call','criterion','threshold','radius_m','status','model_last_pass_m',
                   'model_next_fail_m','observed_last_pass_m','observed_next_fail_m','observed_status','observed_reentries','lowest_component']),
            'nodes':pd.DataFrame(nodes,columns=['call','criterion','distance_m','observed','parametric']),
            'held':held,'scores':pd.DataFrame(scores,columns=['criterion','model','call','n','rmse','mae']),
            'intersections':pd.DataFrame(intersections,columns=['model','criterion','first','second','distance_m','threshold']),
            'completion_pava':completion_bounds(means,lower,upper)}


def figures(summary, results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from figures_apmcm import save
    from utils.plot_style import apply_publication_style
    apply_publication_style('en','report')
    plt.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False,
                        'svg.hashsalt':'penguins-v6-acoustics'})
    colors=['#307976','#aa5c35','#464c73']
    fig,axes=plt.subplots(1,2,figsize=(6.55,2.65),layout='constrained')
    methods=['raw_empirical','blend_balanced','classical_nested']
    for ax,metric,title in zip(axes,['balanced_accuracy','tv_recall'],['a   Balanced accuracy','b   Two-voice recall']):
        for i,key in enumerate(methods):
            block=summary[summary.method==key].set_index('validation')
            yy=100*block.loc[['date_holdout','year_forward'],metric].to_numpy()
            ax.plot([0,1],yy,lw=1.4,color=colors[i],marker=['o','s','D'][i],ms=4,label=['TabICLv2','Two-view','Nested trees'][i])
        ax.set(title=title,xlim=(-.15,1.15),ylim=(0,100),xticks=[0,1],xticklabels=['Hold one date','Train 2021 / test 2022'],ylabel='Percent')
        ax.grid(axis='y',lw=.4,alpha=.3)
    axes[0].legend(loc='upper center',fontsize=6,frameon=False,ncol=1)
    save(fig,out,'result_q1_transfer')
    fig,axes=plt.subplots(1,2,figsize=(6.55,2.78),sharey=True,layout='constrained')
    full=results['nodes']; param=results['parameters']; dense=np.geomspace(4,256,151)
    for ax,crit,title in zip(axes,['mean','all'],['a   Mean fidelity','b   Weakest component']):
        for i,call in enumerate(CALLS):
            p=param[param.call==call]
            parts=p.a.to_numpy()[:,None]/(1+(dense[None,:]/p.b_m.to_numpy()[:,None])**p.c.to_numpy()[:,None])
            pred=parts.mean(axis=0) if crit=='mean' else parts.min(axis=0)
            df=full.query('criterion==@crit and call==@call')
            ax.plot(dense,pred,color=colors[i],ls=['-','--','-.'][i],lw=1.4,label=['Flat','Modulated','Two-voice'][i]);ax.scatter(df.distance_m,df.observed,color=colors[i],marker=['o','s','D'][i],s=12)
        ax.set_xscale('log',base=2);ax.set(xticks=DISTANCES,xticklabels=[f'{d:g}' for d in DISTANCES],title=title,xlabel='Recording distance (m)',ylim=(0,1));ax.axhline(.5,color='.6',lw=.6)
    axes[0].set_ylabel('Structural similarity');axes[1].legend(fontsize=6,frameon=False,loc='upper right');save(fig,out,'result_q2_parametric')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--figures',type=Path,default=HERE/'figures')
    parser.add_argument('--no-figures',action='store_true'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); data,_,raw=read_inputs()
    if args.smoke:
        q = conditional_ks(data,39)
        assert len(q)==18 and q.p_holm.between(q.p_mc,1).all()
        print('18 stratified distribution comparisons and acoustic tables loaded');return
    ks=conditional_ks(data);classification=classification_summary();importance=cue_importances(data)
    all_weights=representation_weights(importance)
    weights=pd.DataFrame.from_dict(all_weights,orient='index',columns=METRICS).rename_axis('criterion').reset_index()
    summaries=[]
    for omit,block in [('none',importance),*((d,importance[importance.date!=d]) for d in sorted(importance.date.unique()))]:
        w=representation_weights(block)['class_shrunk'];means=block.groupby('cue').delta_nll.mean()
        summaries.append((omit,int(block.row.nunique()),means.duration,means.spectrum,*w))
    q2=propagation(raw,all_weights);args.output.mkdir(exist_ok=True,parents=True)
    tables={'q1v6_ks':ks,'q1v6_classification':classification,'q1v6_donor_loss':importance,
            'q1v6_importance':pd.DataFrame(summaries,columns=['omit_test_date','n','duration_delta_nll','spectrum_delta_nll',*METRICS]),
            'q2v6_weights':weights,**{f'q2v6_{k}':v for k,v in q2.items()}}
    for name,frame in tables.items():frame.to_csv(args.output/(name+'.csv'),index=False,float_format='%.12g')
    if not args.no_figures:
        args.figures.mkdir(exist_ok=True,parents=True);figures(classification,q2,args.figures)


if __name__=='__main__':main()
