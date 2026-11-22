from pathlib import Path
import json, sys, subprocess, tarfile, math
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'stage8_metrics'))
from metric_functions import top_k

def verify_scripts():
    checks=['stage4_features/verify_features.py','stage5_methods/verify_methods.py',
            'stage6_scenarios/verify_scenarios.py','stage8_metrics/verify_metrics.py',
            'stage9_stability/verify_bootstrap.py']
    results={}
    for name in checks:
        result=subprocess.run([sys.executable,'-B',str(ROOT/name)],capture_output=True,text=True)
        if result.returncode: raise RuntimeError(result.stderr)
        results[name]=result.stdout.strip()
    return results

results=verify_scripts()
print('Module checks passed; auditing full test population',flush=True)
stage7=ROOT/'stage7_leakage'
folds=json.loads((ROOT/'stage3_splits'/'temporal_splits.json').read_text())['folds']
lookup=pd.read_csv(ROOT/'stage4_features'/'feature_row_index.csv.gz',dtype={'msisdn':'string'})
archive=r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz'
with tarfile.open(archive,'r:gz') as tar:
    labels=pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},
                       usecols=['msisdn','week','revenue','revenue_original','is_injected','injection_type'])
assert labels.revenue.ne(labels.revenue_original).eq(labels.is_injected.eq(1)).all()
arr=np.load(ROOT/'stage4_features'/'base_features.npz')
names=arr['columns'].tolist()
base=pd.DataFrame(arr['values'],columns=names)
del arr
train=lookup.loc[lookup.week.between(26,29)&lookup.cohort.eq('primary')]
reference=base.iloc[train.feature_row]
high=float(reference.prior_mean_4.quantile(.9))
stable=float(reference.prior_std_4.quantile(.25))
volatile=float(reference.prior_std_4.quantile(.9))
labels['prior_injection']=labels.set_index(['msisdn','week']).index.isin(
    pd.MultiIndex.from_frame(labels.loc[labels.is_injected.eq(1),['msisdn','week']].assign(week=lambda d:d.week+1)))
weekly=pd.read_csv(ROOT/'stage8_metrics'/'weekly_metrics.csv')
weekly=weekly.loc[weekly.seed.eq(42)]
counts=[];examples=[]
for fold in folds:
    if fold['phase']!='test':continue
    week=fold['score_week']
    assert max(fold['fit_weeks'])<week
    keys=pd.read_csv(stage7/f'test_keys_{week}.csv.gz',dtype={'msisdn':'string'})
    cohort=lookup.loc[lookup.week.eq(week)]
    expected=cohort.loc[cohort.cohort.eq('primary'),['msisdn','week']].sort_values('msisdn').reset_index(drop=True)
    pd.testing.assert_frame_equal(keys,expected)
    rows=keys.merge(labels,on=['msisdn','week'],validate='one_to_one').merge(
        cohort[['msisdn','week','feature_row']],on=['msisdn','week'],validate='one_to_one')
    assert len(rows)==len(keys) and rows.msisdn.equals(keys.msisdn)
    x=base.iloc[rows.feature_row].reset_index(drop=True)
    y=rows.is_injected.to_numpy(dtype=bool)
    no_label=~y
    groups={
        'stable_high_revenue':(x.prior_mean_4.ge(high)&x.prior_std_4.le(stable)).to_numpy(),
        'rise_after_zero':(x.lag_1.eq(0)&x.current_revenue.gt(0)).to_numpy(),
        'high_volatility_without_current_injection':x.prior_std_4.ge(volatile).to_numpy()&no_label&~rows.prior_injection.to_numpy(),
        'prior_week_injection':rows.prior_injection.to_numpy(),
        'all_primary':np.ones(len(rows),dtype=bool)}
    short=cohort.loc[cohort.cohort.eq('short_history')].merge(labels,on=['msisdn','week'])
    counts.append(dict(week=week,method='unscored',case='short_history',population=len(short),injected=int(short.is_injected.sum())))
    for method,checks in weekly.loc[weekly.week.eq(week)].groupby('method'):
        scores=np.load(stage7/f'scores_{method}_causal_w{week}_s42.npz')['scores']
        assert len(scores)==len(keys)
        for check in checks.itertuples():
            k=math.ceil(len(keys)*check.audit_fraction)
            chosen=top_k(scores,keys.msisdn,k)
            assert len(np.unique(chosen))==check.selected_rows==k
            assert y[chosen].sum()==check.true_positives
        chosen=top_k(scores,keys.msisdn,math.ceil(len(keys)*.01))
        picked=np.zeros(len(rows),dtype=bool)
        picked[chosen]=True
        for name,group in groups.items():
            fn=group&y&~picked
            fp=group&no_label&picked
            counts.append(dict(week=week,method=method,case=name,population=int(group.sum()),
                injected=int((group&y).sum()),detected=int((group&y&picked).sum()),
                missed=int(fn.sum()),selected_without_injection=int(fp.sum())))
            if method not in {'global','lightgbm_absolute','isolation_forest'}:continue
            for status,mask in [('selected_without_injection',fp),('missed_injection',fn)]:
                candidates=np.flatnonzero(mask)
                order=np.argsort(-scores[candidates] if status.startswith('selected') else scores[candidates],kind='stable')
                for i in candidates[order[:2]]:
                    examples.append(dict(week=week,method=method,case=name,outcome=status,score=scores[i],
                       injection_type=rows.injection_type[i],revenue=x.current_revenue[i],lag_1=x.lag_1[i],
                       prior_mean=x.prior_mean_4[i],prior_std=x.prior_std_4[i],history=x.prior_observation_count[i]))
pd.DataFrame(counts).to_csv(OUT/'error_counts.csv',index=False)
pd.DataFrame(examples).to_csv(OUT/'anonymous_examples.csv',index=False)
times=pd.read_csv(stage7/'paired_results.csv').query("arm=='causal' and fraction==0.01")
cost=times.groupby('method').agg(measurements=('week','size'),fit_median_s=('fit_seconds','median'),
    fit_max_s=('fit_seconds','max'),score_week_median_s=('score_seconds','median'),score_week_max_s=('score_seconds','max')).reset_index()
cost['fit_shared_with']=np.where(cost.method.eq('lightgbm_normalized'),'lightgbm_absolute','')
cost.to_csv(OUT/'runtime.csv',index=False)
times[['method','week','seed','fit_seconds','score_seconds']].to_csv(OUT/'runtime_observations.csv',index=False)
env=json.loads((ROOT/'stage1_audit'/'environment.json').read_text())
(OUT/'hardware.json').write_text(json.dumps({k:env[k] for k in ['cpu','memory','platform','python']},indent=2))
results.update({'exact_journal_labels':True,'strict_temporal_fit':True,'shared_primary_keys':True,
                'reconciled_weekly_rankings':len(weekly),'example_count':len(examples)})
(OUT/'regression_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(cost.to_string(index=False),flush=True)
