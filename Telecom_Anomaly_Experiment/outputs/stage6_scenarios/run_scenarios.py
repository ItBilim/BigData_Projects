import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import pandas as pd
from scenario_generator import inject_phase,apply_journal

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'stage4_features'))
from feature_engineering import build_features
config={'version':'1.0.0','seeds':[42,43,44,45,46],'rates':[.001,.005,.01,.02],
 'factors':[1.5,2.,4.,.75,.5,.1],'durations':[1,2],
 'phases':{'validation':[30,31],'test':[32,33,34,35]},
 'training_contamination_rates':[0.,.01,.05],
 'minimum_delta_rule':'0.01 * median positive original revenue of primary fit rows weeks26..29',
 'growth_rule':'new=old+max(old*(factor-1),minimum_delta)',
 'dip_rule':'new=old*factor; require old-new>=minimum_delta; otherwise skip event',
 'rate_rule':'floor(N*rate/duration) nonoverlapping events; N=primary phase candidate rows',
 'phase_seed_rule':'SHA256(seed|phase|stage6-v1), excludes factor and rate for paired keys',
 'contamination_rule':'Training-only one-week factor 2. Independent training RNG; same held-out journals reused unchanged.'}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    panel=pd.read_csv(tar.extractfile('2_weekly_revenue_panel.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','revenue','is_filled'])
    original=pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','revenue_original','revenue','is_injected','injection_type'])
panel=panel.sort_values(['msisdn','week']).reset_index(drop=True)
merged=panel.merge(original,on=['msisdn','week'],how='left',suffixes=('_panel','_given'),validate='one_to_one')
panel['revenue']=merged.revenue_original.fillna(merged.revenue_panel)
assert merged.revenue_original.notna().eq(panel.is_filled.eq(0)).all()
manifest=ROOT/'stage2_protocol'/'cohort_manifest.csv.gz'
splits=json.loads((ROOT/'stage3_splits'/'temporal_splits.json').read_text())
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==splits['source_manifest_sha256']
candidates=pd.read_csv(manifest,dtype={'msisdn':'string'})
candidates=candidates.loc[candidates.cohort.eq('primary')]
train_keys=candidates.loc[candidates.week.between(26,29),['msisdn','week']]
train=panel.merge(train_keys,on=['msisdn','week'],validate='one_to_one')
threshold=.01*float(train.loc[train.revenue.gt(0),'revenue'].median())
summary={'minimum_absolute_delta':threshold,'training_original_positive_quantiles':train.loc[train.revenue.gt(0),'revenue'].quantile([0,.01,.5,.99,1]).to_dict(),
 'split_spec_sha256':splits['split_spec_sha256'],'source_manifest_sha256':splits['source_manifest_sha256'],
 'base_note':'Original revenue preserves natural variation; label zero means no artificial perturbation only.'}
(OUT/'distribution_and_lock.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
panel.to_csv(OUT/'original_panel.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
given=merged.loc[merged.is_injected.eq(1)].copy()
given_journal=pd.DataFrame({'scenario':'provided','seed':pd.NA,'msisdn':given.msisdn,'week':given.week,
 'panel_row':given.index,'original_revenue':given.revenue_original,'new_revenue':given.revenue_given,
 'type':given.injection_type,'absolute_delta':given.revenue_given-given.revenue_original})
given_journal['effective_factor']=given_journal.new_revenue/given_journal.original_revenue
given_journal.to_csv(OUT/'provided_journal.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
stats=[]; scenarios=[]
for duration in config['durations']:
  for factor in config['factors']:
    for rate in config['rates']:
      scenario=f'd{duration}_f{factor:g}_r{rate:g}'
      for seed in config['seeds']:
        pieces=[]
        for phase,weeks in config['phases'].items():
          journal,status=inject_phase(panel,candidates,phase,weeks,seed,rate,factor,duration,threshold,scenario)
          pieces.append(journal); stats.append({'scenario':scenario,'seed':seed,**status})
        journal=pd.concat(pieces,ignore_index=True)
        assert not journal.duplicated(['msisdn','week']).any()
        assert journal.absolute_delta.abs().ge(threshold-1e-12).all()
        path=OUT/f'{scenario}_s{seed}.csv.gz'
        journal.to_csv(path,index=False,compression={'method':'gzip','compresslevel':1})
        scenarios.append({'scenario':scenario,'seed':seed,'journal':path.name,'rows':len(journal),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
      print('Generated',scenario,flush=True)
for seed in config['seeds']:
  for rate in config['training_contamination_rates'][1:]:
    journal,status=inject_phase(panel,candidates,'training',[26,27,28,29],seed,rate,2.,1,threshold,f'train_r{rate:g}')
    alternating=np.zeros(len(journal),dtype=bool)
    # All selected originals pass half-value minimum delta in this population; verify explicitly.
    new=np.where(alternating,journal.original_revenue*.5,journal.new_revenue)
    assert np.all(np.abs(new-journal.original_revenue)>=threshold-1e-12)
    journal['new_revenue']=new; journal['requested_factor']=np.where(alternating,.5,2.)
    journal['type']=np.where(alternating,'dip','spike'); journal['absolute_delta']=new-journal.original_revenue
    journal['effective_factor']=new/journal.original_revenue
    path=OUT/f'training_r{rate:g}_s{seed}.csv.gz'
    journal.to_csv(path,index=False,compression={'method':'gzip','compresslevel':1})
    stats.append({'scenario':'training_contamination','seed':seed,**status})
    scenarios.append({'scenario':'training_contamination','seed':seed,'rate':rate,'journal':path.name,'rows':len(journal),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(stats).to_csv(OUT/'scenario_coverage.csv',index=False)
(OUT/'scenario_manifest.json').write_text(json.dumps(scenarios,indent=2),encoding='utf-8')
print('DONE',len(scenarios),'journals',flush=True)
