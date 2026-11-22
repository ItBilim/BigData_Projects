"""Track two-week injections through onset, continuation and the next week."""
from pathlib import Path
import sys, math
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
for directory in ['stage4_features','stage6_scenarios','stage8_metrics']:
    sys.path.insert(0,str(ROOT/directory))
from feature_engineering import build_features
from scenario_generator import apply_journal
from metric_functions import top_k

source=ROOT/'stage6_scenarios'
panel=pd.read_csv(source/'original_panel.csv.gz',dtype={'msisdn':'string'})
journal=pd.read_csv(source/'d2_f2_r0.01_s42.csv.gz',dtype={'msisdn':'string'})
changed=apply_journal(panel,journal)
assert np.array_equal(np.flatnonzero(changed.revenue.ne(panel.revenue)),np.sort(journal.panel_row))
keys,x=build_features(changed)
index=pd.read_csv(ROOT/'stage4_features'/'feature_row_index.csv.gz',dtype={'msisdn':'string'})
episodes=journal.loc[journal.phase.eq('test')].sort_values('week').copy()
episodes['position']=episodes.groupby('event_id').cumcount()+1
assert episodes.groupby('event_id').size().eq(2).all()
after=episodes.loc[episodes.position.eq(2),['msisdn','week']].assign(week=lambda d:d.week+1,position=3)
marks=pd.concat([episodes[['msisdn','week','position']],after]).drop_duplicates(['msisdn','week','position'])
records=[]
for week in range(32,36):
    rows=index.loc[index.week.eq(week)&index.cohort.eq('primary')].sort_values('msisdn')
    fit=index.loc[index.week.between(26,week-1)&index.cohort.eq('primary')]
    data=x.iloc[rows.feature_row]
    center=x.iloc[fit.feature_row].current_revenue.median()
    scores={'global':(data.current_revenue-center).abs().to_numpy(),
            'simple_absolute':(data.current_revenue-data.prior_mean_4).abs().to_numpy(),
            'weekly_absolute':data.absolute_change_1.to_numpy()}
    for method,score in scores.items():
        flags=rows[['msisdn','week']].copy()
        flags['selected']=False
        flags.iloc[top_k(score,rows.msisdn,math.ceil(len(rows)*.01)),2]=True
        matched=marks.merge(flags,on=['msisdn','week'])
        for step,g in matched.groupby('position'):
            records.append(dict(week=week,method=method,episode_position=step,rows=len(g),selected=int(g.selected.sum()),rate=g.selected.mean()))
following=journal.assign(week=lambda d:d.week+1).merge(keys.reset_index(names='row'),on=['msisdn','week'])
assert np.allclose(x.iloc[following.row].lag_1,following.new_revenue,rtol=0,atol=1e-10)
pd.DataFrame(records).to_csv(OUT/'episode_audit.csv',index=False)
print('Verified changed rows and next-week lag for',len(following),'rows',flush=True)
