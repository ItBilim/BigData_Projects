import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from materialize_features import scenario_features
OUT=Path(__file__).resolve().parent
manifest=json.loads((OUT/'scenario_manifest.json').read_text())
assert len(manifest)==250
changed=0
for entry in manifest:
    path=OUT/entry['journal']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    j=pd.read_csv(path,dtype={'msisdn':'string'})
    assert len(j)==entry['rows']
    assert not j.duplicated(['msisdn','week']).any()
    assert np.isfinite(j[['original_revenue','new_revenue','absolute_delta']]).all().all()
    assert j.new_revenue.ne(j.original_revenue).all()
    assert np.allclose(j.new_revenue-j.original_revenue,j.absolute_delta,rtol=0,atol=1e-10)
    changed+=len(j)
print('All journals verified; rebuilding full scenario history',flush=True)
keys,x,state,mapping=scenario_features('d2_f2_r0.01_s42.csv.gz','test_33','training_r0.01_s42.csv.gz')
selected=mapping.loc[mapping.week.eq(33)&mapping.cohort.eq('primary'),'feature_row']
assert np.isfinite(x.iloc[selected].to_numpy()).all()
j=pd.concat([pd.read_csv(OUT/'d2_f2_r0.01_s42.csv.gz',dtype={'msisdn':'string'}),pd.read_csv(OUT/'training_r0.01_s42.csv.gz',dtype={'msisdn':'string'})])
lookup={(sid,w):i for i,sid,w in keys[['msisdn','week']].itertuples()}
verified=0
for row in j.itertuples():
    nxt=lookup.get((row.msisdn,row.week+1))
    if nxt is not None:
        assert np.isclose(x.iloc[nxt].lag_1,row.new_revenue)
        verified+=1
summary={'journal_files':len(manifest),'journal_rows':changed,'full_history_rows_recomputed':len(keys),
 'primary_score_rows_checked':len(selected),'lag_propagations_checked':verified,'recomputed_fold_scale':state,
 'method_metrics_computed':False,'generator_checks':json.loads((OUT/'generator_checks.json').read_text())}
(OUT/'completion_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
