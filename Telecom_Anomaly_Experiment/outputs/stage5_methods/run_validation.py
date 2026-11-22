import sys
from pathlib import Path
import hashlib
import json
import math
import tarfile
import time
import importlib.metadata
from methods import AnomalyMethod
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from feature_engineering import transform

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
config_path=OUT/'config.json'
config=json.loads(config_path.read_text())
lock={'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm']}}
lock_path=OUT/'preregistered_run.json'
if lock_path.exists():
    assert json.loads(lock_path.read_text())['config_sha256']==lock['config_sha256']
else: lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
print('Loading features',flush=True)
data=np.load(ROOT/'stage4_features'/'base_features.npz')
base=pd.DataFrame(data['values'],columns=data['columns'])
del data
index=pd.read_csv(ROOT/'stage4_features'/'feature_row_index.csv.gz',dtype={'msisdn':'string'})
states=json.loads((ROOT/'stage4_features'/'fold_scale_states.json').read_text())
splits=json.loads((ROOT/'stage3_splits'/'temporal_splits.json').read_text())
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    labels=pd.concat([c.loc[c.week.isin(config['validation_weeks'])] for c in pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','is_injected'],chunksize=200000)],ignore_index=True)
fixed=['random','global','weekly_absolute','weekly_relative','personal_mad','simple_absolute','simple_normalized']
records=[]
for fold in splits['folds']:
    if fold['phase']!='validation': continue
    week=fold['score_week']; state=states[fold['fold_id']]
    training=index.loc[index.week.isin(fold['fit_weeks']) & index.cohort.eq('primary')]
    scoring=index.loc[index.week.eq(week) & index.cohort.eq('primary')].sort_values('msisdn')
    y=scoring[['msisdn','week']].merge(labels,on=['msisdn','week'],validate='one_to_one').is_injected.to_numpy()
    train_x=transform(base.iloc[training.feature_row].reset_index(drop=True),state)
    score_x=transform(base.iloc[scoring.feature_row].reset_index(drop=True),state)
    assert np.isfinite(score_x.to_numpy()).all()
    assert training.week.max()<week
    scoring[['msisdn','week']].to_csv(OUT/f'validation_keys_{week}.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
    for seed in config['seeds']:
        rng=np.random.default_rng(seed)
        chosen=np.sort(rng.choice(len(train_x),min(len(train_x),config['maximum_fit_rows_per_learned_model']),replace=False))
        jobs=[(name,0,{}) for name in fixed]
        jobs += [('isolation_forest',i,p) for i,p in enumerate(config['isolation_forest_candidates'])]
        jobs += [('lightgbm',i,config['lightgbm_fixed']|p) for i,p in enumerate(config['lightgbm_candidates'])]
        for name,candidate,params in jobs:
            if seed!=config['seeds'][0] and name in fixed and name!='random': continue
            learned=name in ['isolation_forest','lightgbm']
            model=AnomalyMethod(name,params,seed,config['threads'])
            start=time.perf_counter()
            model.fit(train_x.iloc[chosen] if learned else train_x,scale_floor=state['floor'])
            fit_seconds=time.perf_counter()-start
            variants=['lightgbm_absolute','lightgbm_normalized'] if name=='lightgbm' else [name]
            for variant in variants:
                model.name=variant
                start=time.perf_counter(); scores=model.score(score_x); elapsed=time.perf_counter()-start
                assert scores.shape==(len(scoring),) and np.isfinite(scores).all()
                np.savez_compressed(OUT/f'scores_{variant}_c{candidate}_w{week}_s{seed}.npz',scores=scores)
                for fraction in [.005,.01,.02,.05]:
                    k=max(1,math.ceil(len(y)*fraction))
                    # scoring keys are already sorted by textual identifier; stable sort breaks ties.
                    selected=np.argsort(-scores,kind='stable')[:k]
                    tp=int(y[selected].sum())
                    records.append({'method':variant,'candidate':candidate,'week':week,'seed':seed,'fraction':fraction,'k':k,'tp':tp,'precision':tp/k,'recall':tp/y.sum() if y.sum() else np.nan,
                        'average_precision':average_precision_score(y,scores),'fit_rows':len(chosen) if learned else len(train_x),'fit_seconds':fit_seconds,'score_seconds':elapsed})
                print(week,seed,variant,candidate,'P@1%',records[-3]['precision'],flush=True)
            pd.DataFrame(records).to_csv(OUT/'validation_results.csv',index=False)
    del train_x,score_x
results=pd.DataFrame(records)
main=results.loc[results.fraction.eq(.01)]
weekly=main.groupby(['method','candidate','week']).precision.mean().reset_index()
summary=weekly.groupby(['method','candidate']).precision.agg(['mean','min','max']).reset_index().rename(columns={'mean':'mean_weekly_precision_at_1pct','min':'worst_week_precision','max':'best_week_precision'})
summary.to_csv(OUT/'candidate_summary.csv',index=False)
best=summary.sort_values(['method','mean_weekly_precision_at_1pct','candidate'],ascending=[True,False,True]).groupby('method',sort=False).head(1)
best.to_csv(OUT/'selected_methods.csv',index=False)
selected={}
for row in best.itertuples():
    if row.method=='isolation_forest': params=config['isolation_forest_candidates'][row.candidate]
    elif row.method.startswith('lightgbm'): params=config['lightgbm_fixed']|config['lightgbm_candidates'][row.candidate]
    else: params={}
    selected[row.method]={'candidate':row.candidate,'params':params,'validation_precision_at_1pct':row.mean_weekly_precision_at_1pct}
(OUT/'selected_parameters.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
print(best.to_string(index=False),flush=True)
