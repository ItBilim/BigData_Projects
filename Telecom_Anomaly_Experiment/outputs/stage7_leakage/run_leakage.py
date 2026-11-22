import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'stage5_methods'))
from methods import AnomalyMethod
from feature_engineering import transform,BASE
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import hashlib,json,math,tarfile,time,warnings

config=json.loads((OUT/'config.json').read_text())
selected_path=ROOT/'stage5_methods'/'selected_parameters.json'
selected=json.loads(selected_path.read_text())
splits=json.loads((ROOT/'stage3_splits'/'temporal_splits.json').read_text())
states=json.loads((ROOT/'stage4_features'/'fold_scale_states.json').read_text())
lock={'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
 'selected_parameters_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
 'split_spec_sha256':splits['split_spec_sha256'],'start_utc':pd.Timestamp.now(tz='UTC').isoformat()}
(OUT/'run_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
print('Loading base and labels',flush=True)
data=np.load(ROOT/'stage4_features'/'base_features.npz')
causal=pd.DataFrame(data['values'],columns=data['columns']); del data
keys=pd.read_csv(ROOT/'stage4_features'/'panel_keys.csv.gz',dtype={'msisdn':'string'})
index=pd.read_csv(ROOT/'stage4_features'/'feature_row_index.csv.gz',dtype={'msisdn':'string'})
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    labels=pd.concat([c.loc[c.week.isin(config['test_weeks'])] for c in pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','is_injected'],chunksize=200000)],ignore_index=True)
print('Constructing hindsight neighbor control',flush=True)
revenue=causal.current_revenue
neighbors=[]; fillneighbors=[]; offsets=[]
for step in [-1,1,-2,2,-3,3,-4,4]:
    neighbors.append(revenue.groupby(keys.msisdn,sort=False).shift(-step).to_numpy())
    fillneighbors.append(keys.is_filled.groupby(keys.msisdn,sort=False).shift(-step).to_numpy())
    offsets.append(step)
raw=np.column_stack(neighbors)
order=np.argsort(~np.isfinite(raw),axis=1,kind='stable')[:,:4]
window=np.take_along_axis(raw,order,axis=1)
fill=np.take_along_axis(np.column_stack(fillneighbors),order,axis=1)
future=np.take_along_axis(np.broadcast_to(np.array(offsets),raw.shape),order,axis=1)>0
future &= np.isfinite(window)
leaky=causal.copy()
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning)
    leaky['prior_mean_2']=np.nanmean(window[:,:2],axis=1)
    leaky['prior_mean_4']=np.nanmean(window,axis=1)
    leaky['prior_median_4']=np.nanmedian(window,axis=1)
    leaky['prior_std_4']=np.nanstd(window,axis=1,ddof=1)
    leaky['prior_mad_4']=np.nanmedian(np.abs(window-leaky.prior_median_4.to_numpy()[:,None]),axis=1)
    leaky['prior_min_4']=np.nanmin(window,axis=1)
    leaky['prior_max_4']=np.nanmax(window,axis=1)
for i in range(4): leaky[f'lag_{i+1}']=window[:,i]
for stat in ['mean','median','min','max']:
    leaky[f'deviation_from_prior_{stat}']=revenue-leaky[f'prior_{stat}_4']
leaky['absolute_change_1']=np.abs(revenue-window[:,0])
count=np.isfinite(window).sum(axis=1)
leaky['prior_zero_share_4']=np.divide((window==0).sum(axis=1),count,out=np.full(len(keys),np.nan),where=count>0)
leaky['prior_filled_share_4']=np.divide(np.nansum(fill,axis=1),count,out=np.full(len(keys),np.nan),where=count>0)
coverage=[]; records=[]
for fold in splits['folds']:
    if fold['phase']!='test': continue
    week=fold['score_week']; state=states[fold['fold_id']]
    train=index.loc[index.week.isin(fold['fit_weeks']) & index.cohort.eq('primary')]
    score=index.loc[index.week.eq(week)&index.cohort.eq('primary')].sort_values('msisdn')
    tr=train.feature_row.to_numpy(); sc=score.feature_row.to_numpy()
    assert train.week.max()<week
    y=score[['msisdn','week']].merge(labels,on=['msisdn','week'],validate='one_to_one').is_injected.to_numpy()
    score[['msisdn','week']].to_csv(OUT/f'test_keys_{week}.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
    coverage.append({'week':week,'rows':len(sc),'score_rows_with_future':int(future[sc].any(axis=1).sum()),
       'fit_rows':len(tr),'fit_rows_with_future':int(future[tr].any(axis=1).sum()),'max_fit_key_week':int(train.week.max())})
    for seed in config['seeds']:
        choice=np.sort(np.random.default_rng(seed).choice(len(tr),min(len(tr),config['fit_cap']),replace=False))
        for arm,base in [('causal',causal),('hindsight',leaky)]:
            tx=transform(base.iloc[tr].reset_index(drop=True),state)
            sx=transform(base.iloc[sc].reset_index(drop=True),state)
            assert np.isfinite(sx.to_numpy()).all()
            for name in selected:
                if name=='lightgbm_normalized': continue
                stochastic=name in ['random','isolation_forest','lightgbm_absolute']
                if seed!=config['seeds'][0] and not stochastic: continue
                model=AnomalyMethod(name,selected[name]['params'],seed,config['threads'])
                learned=name in ['isolation_forest','lightgbm_absolute']
                start=time.perf_counter(); model.fit(tx.iloc[choice] if learned else tx,scale_floor=state['floor']); fit_time=time.perf_counter()-start
                variants=['lightgbm_absolute','lightgbm_normalized'] if name=='lightgbm_absolute' else [name]
                for variant in variants:
                    model.name=variant; start=time.perf_counter(); scores=model.score(sx); score_time=time.perf_counter()-start
                    assert len(scores)==len(sc) and np.isfinite(scores).all()
                    np.savez_compressed(OUT/f'scores_{variant}_{arm}_w{week}_s{seed}.npz',scores=scores)
                    ap=average_precision_score(y,scores)
                    for fraction in [.005,.01,.02,.05]:
                        k=math.ceil(len(y)*fraction); chosen=np.argsort(-scores,kind='stable')[:k]; tp=int(y[chosen].sum())
                        records.append({'method':variant,'arm':arm,'week':week,'seed':seed,'fraction':fraction,'k':k,'tp':tp,'positives':int(y.sum()),'precision':tp/k,'recall':tp/y.sum(),'average_precision':ap,'fit_seconds':fit_time,'score_seconds':score_time})
                print(week,seed,arm,name,flush=True)
                pd.DataFrame(records).to_csv(OUT/'paired_results.csv',index=False)
            del tx,sx
pd.DataFrame(coverage).to_csv(OUT/'future_information_coverage.csv',index=False)
results=pd.DataFrame(records)
weekly=results.groupby(['method','arm','week','fraction'])[['precision','recall','average_precision']].mean().reset_index()
means=weekly.groupby(['method','arm','fraction'])[['precision','recall','average_precision']].mean().reset_index()
rows=[]
for (method,fraction),group in means.groupby(['method','fraction']):
    a=group.set_index('arm')
    row={'method':method,'fraction':fraction}
    for metric in ['precision','recall','average_precision']:
        before=float(a.loc['causal',metric]); after=float(a.loc['hindsight',metric])
        row.update({f'causal_{metric}':before,f'hindsight_{metric}':after,
            f'delta_{metric}_percentage_points':100*(after-before),
            f'relative_{metric}_percent':100*(after-before)/before if before else np.nan})
    rows.append(row)
impact=pd.DataFrame(rows)
impact.to_csv(OUT/'information_impact.csv',index=False)
impact.loc[impact.fraction.eq(.01)].to_csv(OUT/'information_impact_at_1pct.csv',index=False)
print(impact.loc[impact.fraction.eq(.01),['method','causal_precision','hindsight_precision','delta_precision_percentage_points','relative_precision_percent']].to_string(index=False),flush=True)
