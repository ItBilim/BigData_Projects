import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from feature_engineering import *

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
split_path=ROOT/'stage3_splits'/'temporal_splits.json'
splits=json.loads(split_path.read_text())
manifest_path=ROOT/'stage2_protocol'/'cohort_manifest.csv.gz'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==splits['source_manifest_sha256']
print('Loading archive',flush=True)
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    panel=pd.read_csv(tar.extractfile('2_weekly_revenue_panel.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','revenue','is_filled'])
    anomaly=pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','revenue'])
panel=panel.merge(anomaly.rename(columns={'revenue':'scenario_revenue'}),on=['msisdn','week'],how='left',validate='one_to_one')
assert panel.scenario_revenue.notna().eq(panel.is_filled.eq(0)).all()
panel['revenue']=panel.scenario_revenue.fillna(panel.revenue)
print('Computing causal features',flush=True)
keys,features=build_features(panel)
del panel,anomaly
manifest=pd.read_csv(manifest_path,dtype={'msisdn':'string'})
indexed=keys.reset_index(names='feature_row')
selection=manifest.merge(indexed,on=['msisdn','week','is_filled'],validate='one_to_one')
selection.to_csv(OUT/'feature_row_index.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
states={}
quality=[]
for fold in splits['folds']:
    fit=selection.loc[selection.week.isin(fold['fit_weeks']) & selection.cohort.eq('primary'),'feature_row']
    mask=np.zeros(len(features),dtype=bool); mask[fit.to_numpy()]=True
    state=fit_scale(features,mask)
    state.update({'fit_weeks':fold['fit_weeks'],'score_week':fold['score_week']})
    states[fold['fold_id']]=state
    transformed=transform(features,state)
    score=selection.loc[selection.week.eq(fold['score_week']) & selection.cohort.eq('primary'),'feature_row']
    assert np.isfinite(transformed.iloc[score][list(DETECTOR_FEATURES)]).all().all()
    quality.append({'fold':fold['fold_id'],'scale_floor':state['floor'],'fit_rows':len(fit),'score_rows':len(score),'primary_nonfinite':0})
    del transformed
print('Saving reusable base features',flush=True)
np.savez_compressed(OUT/'base_features.npz',values=features.to_numpy(dtype='float64'),columns=np.array(BASE))
keys.to_csv(OUT/'panel_keys.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
(OUT/'fold_scale_states.json').write_text(json.dumps(states,indent=2),encoding='utf-8')
pd.DataFrame(quality).to_csv(OUT/'feature_quality.csv',index=False)
missing=features.isna().sum().rename('missing_count').to_frame()
missing.to_csv(OUT/'base_feature_missing.csv')
current=set(CURRENT_DEPENDENT)
definitions={
'current_revenue':('revenue[t]','t'),
'absolute_change_1':('abs(revenue[t]-revenue[t-1])','t'),
'relative_change_1':('(revenue[t]-lag_1)/max(abs(lag_1),floor)','t; training floor'),
'prior_std_4':('sample std of available t-4..t-1, ddof=1','t-1'),
'prior_mad_4':('median(abs(history-median(history))), unscaled','t-1'),
'prior_observation_count':('calendar observations before t, including provenance-flagged fills','t-1'),
'prior_unfilled_count':('unfilled observations before t','t-1'),
'weeks_since_last_positive':('t - latest positive revenue week strictly before t; NaN if none','t-1'),
'no_prior_positive':('1 if no positive revenue before t','t-1'),
'prior_zero_share_4':('zero count / available prior observations in t-4..t-1','t-1'),
'prior_filled_share_4':('filled count / available prior observations in t-4..t-1','t-1'),
'scaled_deviation_from_prior_median':('(revenue[t]-prior_median_4)/max(1.4826*prior_mad_4,floor)','t; training floor')}
dictionary=[]
for name in DETECTOR_FEATURES:
    formula,available=definitions.get(name,('lag_i = revenue[t-i]; prior_*_w uses available t-w..t-1; deviation_from_prior_* = revenue[t]-prior_*','t' if name in current else 't-1'))
    dictionary.append({'feature':name,'formula':formula,'latest_information':available,'forecast_allowed':name in FORECAST_FEATURES,
        'missing_policy':'Keep NaN for unavailable history; imputer must fit training only','dtype':'float64'})
pd.DataFrame(dictionary).to_csv(OUT/'feature_dictionary.csv',index=False,encoding='utf-8-sig')
contract={'version':VERSION,'extends_stage2':'1.0.0','reason':'Stage 4 requested mean/min/max deviations and fill provenance absent from initial whitelist',
 'detector_features':DETECTOR_FEATURES,'forecast_features':FORECAST_FEATURES,'forbidden':sorted(FORBIDDEN),
 'split_spec_sha256':splits['split_spec_sha256'],'checks':'Primary scoring rows finite in every fold'}
(OUT/'feature_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
print(json.dumps(quality),flush=True)
