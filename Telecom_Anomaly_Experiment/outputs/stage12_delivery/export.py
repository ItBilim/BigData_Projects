"""Export keyed scores and anonymous history examples for the article."""
from pathlib import Path
import json, math, tarfile
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
cfg = json.loads((OUT/'experiment.json').read_text())
with tarfile.open(cfg['archive'], 'r:gz') as archive:
    labels = pd.read_csv(archive.extractfile('3_weekly_revenue_anomalies.csv'),
        dtype={'msisdn':'string'}, usecols=['msisdn','week','revenue','revenue_original','is_injected','injection_type'])
weeks = [32,33,34,35]
keys = {w:pd.read_csv(ROOT/'stage7_leakage'/f'test_keys_{w}.csv.gz',dtype={'msisdn':'string'}) for w in weeks}
ids = sorted(set().union(*(set(k.msisdn) for k in keys.values())))
# Run-local identifiers carry no phone number or reversible lookup in the export.
tokens = dict(zip(ids,np.random.default_rng(120426).permutation(len(ids))+1))
outdir = OUT/'scores'
outdir.mkdir(exist_ok=True)
manifest = []
for week, key in keys.items():
    rows = key.merge(labels,on=['msisdn','week'],validate='one_to_one')
    assert rows.msisdn.equals(key.msisdn)
    for source, pattern in [('stage7_leakage',f'scores_*_w{week}_s*.npz'),
                            ('stage9_stability',f'synthetic_scores_*_w{week}_inj*.npz')]:
        paths = sorted((ROOT/source).glob(pattern))
        scores, tags, synthetic_labels = [], [], []
        for path in paths:
            with np.load(path) as data:
                score = data['scores']
                assert len(score)==len(rows) and np.isfinite(score).all()
                scores.append(score)
                if 'labels' in data: synthetic_labels.append(data['labels'])
            tags.append(path.stem)
        assert len(paths)==(52 if source=='stage7_leakage' else 50)
        path = outdir/f'{source}_week{week}.npz'
        values = np.stack(scores)
        actual = np.stack(synthetic_labels).astype('uint8') if synthetic_labels else rows.is_injected.to_numpy(dtype='uint8')
        np.savez_compressed(path,score=values,series=np.array(tags),label=actual,
                            subject=key.msisdn.map(tokens).to_numpy(dtype='int32'),week=np.int16(week))
        manifest.append({'file':path.name,'week':week,'rows_per_series':len(rows),'series':len(tags),'scores':values.size})
pd.DataFrame(manifest).to_csv(OUT/'score_manifest.csv',index=False)

week=34
rows=keys[week].merge(labels,on=['msisdn','week'],validate='one_to_one')
scores=np.load(ROOT/'stage7_leakage'/f'scores_lightgbm_absolute_causal_w{week}_s42.npz')['scores']
order=np.argsort(-scores,kind='stable')
chosen=order[:math.ceil(.01*len(rows))]
rows['selected']=False
rows.loc[chosen,'selected']=True
picks=[rows.loc[rows.selected&rows.is_injected.eq(1)].iloc[0],
       rows.loc[rows.selected&rows.is_injected.eq(0)].iloc[0],
       rows.loc[~rows.selected&rows.is_injected.eq(1)].iloc[0]]
series=[]
for label,row in zip(['Selected injection','Selected unlabelled','Missed injection'],picks):
    history=labels.loc[labels.msisdn.eq(row.msisdn),['week','revenue','revenue_original','is_injected']].copy()
    history['case']=label
    history['evaluated_week']=week
    history['selected_at_evaluated_week']=bool(row.selected)
    series.append(history)
pd.concat(series).to_csv(OUT/'anonymous_histories.csv',index=False)
print('Exported',sum(x['scores'] for x in manifest),'scores',flush=True)
