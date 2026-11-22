"""Freeze chronological folds using stage 2 eligibility, without labels."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
source = ROOT / 'stage2_protocol' / 'cohort_manifest.csv.gz'
lock = json.loads((ROOT/'stage2_protocol'/'protocol_lock.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == lock['manifest_sha256']
df = pd.read_csv(source, dtype={'msisdn':'string'})
assert sorted(df.week.unique()) == list(range(22,36))
assert not df.duplicated(['msisdn','week']).any()
df['phase'] = df.week.map(lambda w: 'initial_training' if w <= 29 else 'validation' if w <= 31 else 'test')
df['eligible_for_model_fit'] = df.cohort.eq('primary')
df.to_csv(OUT/'row_assignments.csv.gz', index=False, compression='gzip')
rows=[]
folds=[]
for week in range(30,36):
    phase = 'validation' if week <= 31 else 'test'
    fold_id=f'{phase}_{week}'
    train_weeks=list(range(26,week))
    assert max(train_weeks)<week
    fit=df.loc[df.week.isin(train_weeks) & df.cohort.eq('primary')]
    score=df.loc[df.week.eq(week)]
    assert len(fit)>0 and len(score)>0
    assert fit.prior_observation_count.ge(4).all()
    assert set(zip(fit.msisdn,fit.week)).isdisjoint(set(zip(score.msisdn,score.week)))
    folds.append({'fold_id':fold_id,'phase':phase,'fit_weeks':train_weeks,
        'history_weeks':list(range(22,week+1)), 'score_week':week,
        'fit_max_week':week-1,'hyperparameters_locked':phase=='test',
        'tuning_metric_weeks':[30,31] if phase=='test' else [],
        'fit_cohort':'primary','score_cohorts':['primary','short_history']})
    for role,part in [('fit',df.loc[df.week.isin(train_weeks)]),('score',score)]:
        primary=part.loc[part.cohort.eq('primary')]
        rows.append({'fold_id':fold_id,'phase':phase,'role':role,
            'weeks':','.join(map(str,sorted(part.week.unique()))),
            'candidate_rows':len(part),'candidate_subscribers':part.msisdn.nunique(),
            'primary_rows':len(primary),'primary_subscribers':primary.msisdn.nunique(),
            'short_history_rows':int(part.cohort.eq('short_history').sum()),
            'primary_share':len(primary)/len(part),'fit_max_week':week-1,
            'score_week':week,'actual_fit_rows':len(fit) if role=='fit' else 0})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'fold_summary.csv',index=False,encoding='utf-8-sig')
phase=[]
for name,part in df.groupby('phase'):
    main=part.loc[part.cohort.eq('primary')]
    phase.append({'phase':name,'weeks':','.join(map(str,sorted(part.week.unique()))),
        'rows':len(part),'subscribers':part.msisdn.nunique(),'primary_rows':len(main),
        'primary_subscribers':main.msisdn.nunique(),'primary_share':len(main)/len(part)})
pd.DataFrame(phase).to_csv(OUT/'phase_summary.csv',index=False,encoding='utf-8-sig')
payload={'version':'1.0.0','stage2_contract_sha256':lock['sha256'],
    'source_manifest_sha256':lock['manifest_sha256'],'folds':folds,
    'validation_aggregation':'Equal-weight mean of weekly Precision@1% for weeks 30 and 31',
    'selection_cutoff_week':31,'retraining':'Every scoring week, fit all learned transforms and models again using primary rows from weeks 26 through t-1',
    'previous_test_policy':'Earlier test weeks may enter subsequent fitting without evaluation labels. No hyperparameter or feature changes based on test metrics.',
    'seed_policy':'Same predefined seed list across all folds and methods',
    'checks':{'source_hash_verified':True,'unique_keys':True,'strict_past_fit':True,
        'fit_score_keys_disjoint':True,'minimum_history_verified':True},
    'artifacts':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['row_assignments.csv.gz','fold_summary.csv','phase_summary.csv']}}
canonical=json.dumps(payload,sort_keys=True).encode()
payload['split_spec_sha256']=hashlib.sha256(canonical).hexdigest()
(OUT/'temporal_splits.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(phase).to_string(index=False))
print(summary.to_string(index=False))
print('All split checks passed')
