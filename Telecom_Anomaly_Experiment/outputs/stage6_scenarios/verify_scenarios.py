import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scenario_generator import *
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'stage4_features'))
from feature_engineering import build_features

panel=pd.DataFrame({'msisdn':['001']*14+['002']*14,'week':list(range(22,36))*2,
 'revenue':[5.]*14+[.001,0.]+[10.]*12,'is_filled':[0]*28})
candidates=panel[['msisdn','week']]
j,s=inject_phase(panel,candidates,'test',[32,33,34,35],42,.5,2,2,.1,'test')
j2,_=inject_phase(panel,candidates,'test',[32,33,34,35],42,.5,2,2,.1,'test')
assert_frame_equal(j,j2)
assert j.groupby('event_id').week.agg(lambda x:x.max()-x.min()).eq(1).all()
changed=apply_journal(panel,j)
assert int(changed.revenue.ne(panel.revenue).sum())==len(j)
k,x=build_features(changed)
for row in j.itertuples():
    following=k.index[(k.msisdn==row.msisdn)&(k.week==row.week+1)]
    if len(following): assert x.loc[following[0],'lag_1']==row.new_revenue
baseline=build_features(panel)[1]
assert_frame_equal(baseline.loc[k.week.lt(32)],x.loc[k.week.lt(32)])
zero=pd.DataFrame({'msisdn':['z','a'],'week':[30,30],'revenue':[0.,.001],'is_filled':[0,0]})
z,_=inject_phase(zero,zero,'validation',[30],42,1,1.5,1,.1,'zero')
assert z.absolute_delta.ge(.1-1e-12).all()
assert rng_for(42,'validation').random()!=rng_for(42,'test').random()
checks=['deterministic_seed','consecutive_nonoverlapping_episodes','journal_exact_changes',
 'past_anomaly_enters_next_lag','future_injection_does_not_change_past_features',
 'zero_and_tiny_growth_minimum_delta','independent_phase_rng']
(OUT/'generator_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Passed',len(checks),'checks')
