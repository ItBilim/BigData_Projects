import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'stage5_methods'));sys.path.insert(0,str(ROOT/'stage6_scenarios'))
from methods import AnomalyMethod
from scenario_generator import apply_journal
from feature_engineering import build_features,fit_scale,transform
import hashlib,json,math,time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

config_path=OUT/'injection_seed_config.json'
config=json.loads(config_path.read_text());selected=json.loads((ROOT/'stage5_methods'/'selected_parameters.json').read_text())
(OUT/'injection_seed_lock.json').write_text(json.dumps({'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'started_utc':pd.Timestamp.now(tz='UTC').isoformat()},indent=2),encoding='utf-8')
panel=pd.read_csv(ROOT/'stage6_scenarios'/'original_panel.csv.gz',dtype={'msisdn':'string'})
manifest=pd.read_csv(ROOT/'stage2_protocol'/'cohort_manifest.csv.gz',dtype={'msisdn':'string'})
folds=json.loads((ROOT/'stage3_splits'/'temporal_splits.json').read_text())['folds']
records=[]
for injection_seed in config['injection_seeds']:
    journal=pd.read_csv(ROOT/'stage6_scenarios'/f'd1_f2_r0.01_s{injection_seed}.csv.gz',dtype={'msisdn':'string'})
    changed=apply_journal(panel,journal)
    keys,base=build_features(changed);del changed
    mapping=manifest.merge(keys.reset_index(names='feature_row'),on=['msisdn','week','is_filled'],validate='one_to_one')
    positives=set(zip(journal.msisdn,journal.week))
    for fold in folds:
        if fold['phase']!='test':continue
        week=fold['score_week']
        fit=mapping.loc[mapping.week.isin(fold['fit_weeks'])&mapping.cohort.eq('primary'),'feature_row'].to_numpy()
        score=mapping.loc[mapping.week.eq(week)&mapping.cohort.eq('primary')].sort_values('msisdn')
        mask=np.zeros(len(base),dtype=bool);mask[fit]=True
        state=fit_scale(base,mask)
        tx=transform(base.iloc[fit].reset_index(drop=True),state)
        sx=transform(base.iloc[score.feature_row].reset_index(drop=True),state)
        y=np.array([(sid,week) in positives for sid in score.msisdn],dtype=int)
        choice=np.sort(np.random.default_rng(42).choice(len(tx),min(len(tx),200000),replace=False))
        for name in selected:
            if name=='lightgbm_normalized':continue
            model=AnomalyMethod(name,selected[name]['params'],42,2)
            learned=name in ['isolation_forest','lightgbm_absolute']
            model.fit(tx.iloc[choice] if learned else tx,scale_floor=state['floor'])
            variants=['lightgbm_absolute','lightgbm_normalized'] if name=='lightgbm_absolute' else [name]
            for variant in variants:
                model.name=variant;scores=model.score(sx)
                assert np.isfinite(scores).all()
                path=OUT/f'synthetic_scores_{variant}_w{week}_inj{injection_seed}.npz'
                np.savez_compressed(path,scores=scores,labels=y)
                for fraction in [.005,.01,.02,.05]:
                    k=math.ceil(len(y)*fraction);rank=np.argsort(-scores,kind='stable')[:k];tp=int(y[rank].sum())
                    records.append({'scenario':config['scenario'],'injection_seed':injection_seed,'model_seed':42,'method':variant,'week':week,'fraction':fraction,'population':len(y),'positives':int(y.sum()),'k':k,'tp':tp,'precision':tp/k,'recall':tp/y.sum(),'average_precision':average_precision_score(y,scores),'roc_auc':roc_auc_score(y,scores)})
            pd.DataFrame(records).to_csv(OUT/'injection_seed_weekly.csv',index=False)
        print('Synthetic seed',injection_seed,'week',week,'done',flush=True)
        del tx,sx
    del base,mapping
r=pd.DataFrame(records)
macro=r.groupby(['method','injection_seed','fraction'])[['precision','recall','average_precision','roc_auc']].mean().reset_index()
macro.to_csv(OUT/'injection_seed_macro.csv',index=False)
stats=macro.groupby(['method','fraction'])[['precision','recall','average_precision','roc_auc']].agg(['mean','std','min','max'])
stats.columns=['_'.join(c) for c in stats.columns];stats.reset_index().to_csv(OUT/'injection_seed_variability.csv',index=False)
print('Injection seed sensitivity complete',flush=True)
