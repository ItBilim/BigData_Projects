"""Create cohort manifest and verify the stage 2 contract without training."""
import hashlib
import json
from pathlib import Path
import tarfile
import pandas as pd
from evaluation_contract import *

OUT = Path(__file__).resolve().parent
ARCHIVE = Path(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz')
with tarfile.open(ARCHIVE, 'r:gz') as tar:
    panel = pd.read_csv(tar.extractfile('2_weekly_revenue_panel.csv'), dtype={'msisdn':'string'}, usecols=['msisdn','week','is_filled'])
    observed = pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'), dtype={'msisdn':'string'}, usecols=['msisdn','week'])
assert not panel.duplicated(list(KEY_COLUMNS)).any()
assert not observed.duplicated(list(KEY_COLUMNS)).any()
panel = panel.sort_values(list(KEY_COLUMNS))
panel['prior_observation_count'] = panel.groupby('msisdn').cumcount()
panel['first_week'] = panel.groupby('msisdn').week.transform('min')
assert panel.prior_observation_count.eq(panel.week-panel.first_week).all(), 'Panel has calendar gaps'
panel['prior_unfilled_count'] = panel.groupby('msisdn').is_filled.transform(lambda x: x.eq(0).cumsum()-x.eq(0).astype(int))
cohort = observed.merge(panel, on=list(KEY_COLUMNS), validate='one_to_one')
assert len(cohort)==len(observed)
assert cohort.is_filled.eq(0).all()
cohort['cohort'] = cohort.prior_observation_count.ge(MIN_PRIOR_WEEKS).map({True:'primary',False:'short_history'})
cohort = cohort.sort_values(list(KEY_COLUMNS))
cohort.to_csv(OUT/'cohort_manifest.csv.gz', index=False, compression='gzip')
summary = cohort.groupby(['week','cohort']).size().unstack(fill_value=0)
summary['total'] = summary.sum(axis=1)
summary['primary_share'] = summary['primary']/summary.total
summary.to_csv(OUT/'weekly_coverage.csv',encoding='utf-8-sig')
tests=[]
for bad in ['revenue_original','is_injected','injection_type','outlier_global','spike_local','msisdn','target_unknown','unregistered']:
    try: validate_feature_columns([bad])
    except ValueError: tests.append(bad)
    else: raise AssertionError(bad)
validate_fit_weeks([22,23],24)
try: validate_fit_weeks([22,24],24)
except ValueError: tests.append('future_fit_rejected')
else: raise AssertionError('Temporal check failed')
validate_scored_keys([('a',24)],[('a',24)])
try: validate_scored_keys([('a',24)],[('b',24)])
except ValueError: tests.append('different_cohort_rejected')
else: raise AssertionError('Key check failed')
payload=contract()
payload['sha256']=fingerprint()
payload['checks_passed']=tests
payload['cohort_counts']=cohort.cohort.value_counts().to_dict()
payload['manifest_sha256']=hashlib.sha256((OUT/'cohort_manifest.csv.gz').read_bytes()).hexdigest()
(OUT/'protocol_lock.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.to_string())
print(payload['cohort_counts'])
print('Contract checks passed',len(tests))
