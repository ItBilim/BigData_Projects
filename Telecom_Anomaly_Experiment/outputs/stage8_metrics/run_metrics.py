import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'stage5_methods'))
import methods  # Establish the previously installed, versioned dependency paths.
import hashlib,json,math,tarfile
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from metric_functions import top_k,selection_metrics

seeds=[42,43,44,45,46]; fractions=[.005,.01,.02,.05]; weeks=[32,33,34,35]
method_names=list(json.loads((ROOT/'stage5_methods'/'selected_parameters.json').read_text()))
config={'scenario':'provided_anomalies','arm':'causal','weeks':weeks,'seeds':seeds,'fractions':fractions,
 'tie_rule':'score descending then msisdn text ascending',
 'random_baseline':'Expected uniform random precision=weekly prevalence; also report realized same-seed random lift',
 'segment_fit_weeks':[26,27,28,29],'segment_fit_cohort':'primary',
 'segment_cut_rule':'1/3 and 2/3 quantiles; low<=q1, middle q1<x<=q2, high>q2',
 'magnitude':'absolute natural logarithm of revenue/revenue_original for injected rows; cutoff quantiles from initial training injections only',
 'unscored_scenarios':'Stage6 generated synthetic journals have no model scores and are excluded; this run measures all existing causal test scores.'}
(OUT/'metric_protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print('Loading keys, labels, training segments',flush=True)
index=pd.read_csv(ROOT/'stage4_features'/'feature_row_index.csv.gz',dtype={'msisdn':'string'})
data=np.load(ROOT/'stage4_features'/'base_features.npz')
base=pd.DataFrame(data['values'],columns=data['columns']); del data
fit=index.loc[index.week.between(26,29)&index.cohort.eq('primary')]
income=base.iloc[fit.feature_row].prior_mean_4
income_cuts=income.quantile([1/3,2/3]).to_numpy()
history_cuts=fit.prior_observation_count.quantile([1/3,2/3]).to_numpy()
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    labels=pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','revenue','revenue_original','is_injected','injection_type'])
train_labels=fit[['msisdn','week']].merge(labels,on=['msisdn','week'],validate='one_to_one')
injected=train_labels.loc[train_labels.is_injected.eq(1)]
magnitude_cuts=np.quantile(np.abs(np.log(injected.revenue/injected.revenue_original)),[1/3,2/3])
boundaries={'income':income_cuts.tolist(),'history':history_cuts.tolist(),'magnitude':magnitude_cuts.tolist(),
 'fit_weeks':[26,27,28,29],'fit_rows':len(fit),'injected_fit_rows':len(injected)}
(OUT/'segment_boundaries.json').write_text(json.dumps(boundaries,indent=2),encoding='utf-8')

def bins(values,cuts):
    values=np.asarray(values)
    return np.where(values<=cuts[0],'low',np.where(values<=cuts[1],'middle','high'))

cache={}; tables={}; distributions=[]
for week in weeks:
    keys=pd.read_csv(ROOT/'stage7_leakage'/f'test_keys_{week}.csv.gz',dtype={'msisdn':'string'})
    assert not keys.duplicated(['msisdn','week']).any()
    rows=keys.merge(index[['msisdn','week','feature_row','cohort','prior_observation_count']],on=['msisdn','week'],validate='one_to_one')
    rows=rows.merge(labels,on=['msisdn','week'],validate='one_to_one')
    assert len(rows)==len(keys) and rows.cohort.eq('primary').all()
    rows['income_segment']=bins(base.iloc[rows.feature_row].prior_mean_4.to_numpy(),income_cuts)
    rows['history_segment']=bins(rows.prior_observation_count.to_numpy(),history_cuts)
    rows['magnitude_segment']='not_injected'
    mask=rows.is_injected.eq(1)
    rows.loc[mask,'magnitude_segment']=bins(np.abs(np.log(rows.loc[mask,'revenue']/rows.loc[mask,'revenue_original'])),magnitude_cuts)
    tables[week]=rows
    for dimension in ['income_segment','history_segment','injection_type','magnitude_segment']:
        for group,g in rows.groupby(dimension):
            distributions.append({'week':week,'dimension':dimension,'segment':group,'rows':len(g),'positives':int(g.is_injected.sum())})
pd.DataFrame(distributions).to_csv(OUT/'segment_population.csv',index=False)
del base,index,labels,train_labels
metrics=[]; segments=[]; pooled_store={}; score_checks=[]
for method in method_names:
    deterministic=method not in ['random','isolation_forest','lightgbm_absolute','lightgbm_normalized']
    for seed in seeds:
        all_y=[]; all_scores=[]; selection_by_fraction={q:[] for q in fractions}
        random_selection_by_fraction={q:[] for q in fractions}
        offset=0
        for week in weeks:
            rows=tables[week]; identifiers=rows.msisdn.to_numpy(dtype=str); y=rows.is_injected.to_numpy(dtype=int)
            actual_seed=42 if deterministic else seed
            path=ROOT/'stage7_leakage'/f'scores_{method}_causal_w{week}_s{actual_seed}.npz'
            scores=np.load(path)['scores']
            assert len(scores)==len(y) and np.isfinite(scores).all()
            random_scores=np.load(ROOT/'stage7_leakage'/f'scores_random_causal_w{week}_s{seed}.npz')['scores']
            ap=average_precision_score(y,scores) if y.sum() else np.nan
            auc=roc_auc_score(y,scores) if len(np.unique(y))==2 else np.nan
            for fraction in fractions:
                k=max(1,math.ceil(len(y)*fraction))
                chosen=top_k(scores,identifiers,k)
                random_chosen=top_k(random_scores,identifiers,k)
                rp=float(y[random_chosen].mean())
                record={'scenario':'provided','arm':'causal','method':method,'seed':seed,'score_seed':actual_seed,
                    'deterministic_score_reused':deterministic,'week':week,'audit_fraction':fraction,
                    **selection_metrics(y,chosen,rp),'average_precision':ap,'roc_auc':auc}
                metrics.append(record)
                selection_by_fraction[fraction].extend((chosen+offset).tolist())
                random_selection_by_fraction[fraction].extend((random_chosen+offset).tolist())
                chosen_mask=np.zeros(len(y),dtype=bool);chosen_mask[chosen]=True
                random_mask=np.zeros(len(y),dtype=bool);random_mask[random_chosen]=True
                for dimension in ['income_segment','history_segment','injection_type','magnitude_segment']:
                    for group in sorted(rows[dimension].unique()):
                        group_mask=rows[dimension].eq(group).to_numpy()
                        members=np.flatnonzero(group_mask)
                        selected_global=np.flatnonzero(group_mask&chosen_mask)
                        positive_total=int(y[members].sum()); tp=int(y[selected_global].sum())
                        n_selected=len(selected_global); n=len(members)
                        random_tp=int(y[group_mask&random_mask].sum())
                        segments.append({'scenario':'provided','arm':'causal','method':method,'seed':seed,'score_seed':actual_seed,
                            'week':week,'audit_fraction':fraction,'dimension':dimension,'segment':group,
                            'population_rows':n,'positive_rows':positive_total,'global_selected_rows':k,
                            'selected_rows':n_selected,'true_positives':tp,'selected_without_injection':n_selected-tp,
                            'precision_within_selected_segment':tp/n_selected if n_selected else np.nan,
                            'recall_within_segment':tp/positive_total if positive_total else np.nan,
                            'precision_contribution_to_global':tp/k,'selected_share':n_selected/k,
                            'expected_random_tp':k*positive_total/len(y),
                            'lift_vs_global_random_allocation':tp/(k*positive_total/len(y)) if positive_total else np.nan,
                            'realized_same_seed_random_tp':random_tp,
                            'anomalies_per_100_segment_checks':100*tp/n_selected if n_selected else np.nan})
            all_y.append(y);all_scores.append(scores);offset+=len(y)
        y_all=np.concatenate(all_y);scores_all=np.concatenate(all_scores)
        pooled_ap=average_precision_score(y_all,scores_all)
        pooled_auc=roc_auc_score(y_all,scores_all)
        for fraction in fractions:
            chosen=np.asarray(selection_by_fraction[fraction]);rc=np.asarray(random_selection_by_fraction[fraction])
            pooled_store[(method,seed,fraction)]={'method':method,'seed':seed,'audit_fraction':fraction,
                'aggregation':'pooled_union_of_weekly_topk',**selection_metrics(y_all,chosen,float(y_all[rc].mean())),
                'pooled_score_average_precision':pooled_ap,'pooled_score_roc_auc':pooled_auc}
        print('Computed',method,seed,flush=True)
weekly=pd.DataFrame(metrics); segment=pd.DataFrame(segments);pooled=pd.DataFrame(pooled_store.values())
weekly.to_csv(OUT/'weekly_metrics.csv',index=False)
segment.to_csv(OUT/'segment_weekly_metrics.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
pooled.to_csv(OUT/'pooled_metrics_by_seed.csv',index=False)
metric_cols=['population_rows','positive_rows','selected_rows','true_positives','selected_without_injection','precision_at_k','recall_at_k','expected_random_precision','lift_vs_expected_random','same_seed_random_precision','lift_vs_same_seed_random','anomalies_per_100_checks','average_precision','roc_auc']
seed_week=weekly.groupby(['method','audit_fraction','week'])[metric_cols].mean().reset_index()
seed_week['aggregation']='seed_mean_within_week'
seed_week.to_csv(OUT/'weekly_seed_mean.csv',index=False)
macro=seed_week.groupby(['method','audit_fraction'])[metric_cols].mean().reset_index()
macro['aggregation']='equal_weight_week_mean_after_seed_mean'
macro.to_csv(OUT/'macro_weekly_metrics.csv',index=False)
pool_cols=[c for c in pooled.select_dtypes(include='number') if c not in ['seed','audit_fraction']]
poolmean=pooled.groupby(['method','audit_fraction'])[pool_cols].mean().reset_index()
poolmean['aggregation']='seed_mean_of_pooled_weekly_selections'
poolmean.to_csv(OUT/'pooled_seed_mean.csv',index=False)
segment_values=[c for c in segment.select_dtypes(include='number') if c not in ['seed','score_seed','week','audit_fraction']]
sg_week=segment.groupby(['method','audit_fraction','week','dimension','segment'])[segment_values].mean().reset_index()
sg_week.to_csv(OUT/'segment_weekly_seed_mean.csv',index=False)
sg_macro=sg_week.groupby(['method','audit_fraction','dimension','segment'])[segment_values].mean().reset_index()
sg_macro['aggregation']='equal_weight_observed_weeks_after_seed_mean'
sg_macro.to_csv(OUT/'segment_macro_metrics.csv',index=False)
print('DONE',len(weekly),'weekly metric rows',len(segment),'segment rows',flush=True)
print(macro.loc[macro.audit_fraction.eq(.01),['method','precision_at_k','recall_at_k','lift_vs_expected_random','average_precision']].to_string(index=False))
