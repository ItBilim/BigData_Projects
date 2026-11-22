import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from feature_engineering import *

out=Path(__file__).resolve().parent
checks=[]
sample=pd.DataFrame({'msisdn':['001']*7+['002']*7,'week':list(range(22,29))*2,
 'revenue':[0.,2.,4.,6.,8.,10.,12.]+[5.]*7,'is_filled':[1,0,0,0,0,0,0]+[0]*7})
keys,x=build_features(sample)
r=x.iloc[4]
assert r.lag_1==6 and r.prior_mean_2==5 and r.prior_mean_4==3
assert r.prior_median_4==3 and r.prior_mad_4==2
assert np.isclose(r.prior_std_4,np.std([0,2,4,6],ddof=1))
assert r.prior_zero_share_4==.25 and r.prior_filled_share_4==.25
assert r.prior_min_4==0 and r.prior_max_4==6
assert r.prior_observation_count==4 and r.prior_unfilled_count==3
assert r.weeks_since_last_positive==1
checks.append('hand_calculated_four_week_window')
changed=sample.copy(); changed.loc[changed.week.ge(27),'revenue']=99999.
k2,x2=build_features(changed)
assert_frame_equal(x.loc[keys.week.lt(27)],x2.loc[k2.week.lt(27)])
checks.append('future_change_does_not_affect_past')
changed=sample.copy(); changed.loc[changed.week.eq(26),'revenue']=12345.
_,x3=build_features(changed)
assert_frame_equal(x.loc[keys.week.eq(26),list(FORECAST_FEATURES)],x3.loc[keys.week.eq(26),list(FORECAST_FEATURES)])
checks.append('forecast_features_independent_of_current_revenue')
mask=keys.week.lt(27).to_numpy()
state=fit_scale(x,mask)
assert state==fit_scale(x2,mask)
transformed=transform(x,state)
assert np.isfinite(transformed.loc[keys.msisdn.eq('002') & keys.week.ge(26)].to_numpy()).all()
checks.append('training_scale_independent_of_future_and_constant_series_finite')
assert np.isnan(x.iloc[0].lag_1) and np.isnan(x.iloc[0].prior_mean_4)
assert x.iloc[0].no_prior_positive==1 and np.isnan(x.iloc[0].weeks_since_last_positive)
checks.append('cold_start_keeps_missing_history')
for bad in ['revenue_original','is_injected','target_x','msisdn','week']:
    try: validate_features([bad])
    except ValueError: pass
    else: raise AssertionError(bad)
checks.append('forbidden_columns_rejected')
try: build_features(sample.drop(index=2))
except ValueError: checks.append('calendar_gap_rejected')
else: raise AssertionError('gap accepted')
_,repeat=build_features(sample)
assert_frame_equal(x,repeat)
checks.append('deterministic_repeat')
(out/'verification.json').write_text(json.dumps({'passed':checks,'count':len(checks)},indent=2),encoding='utf-8')
print('Passed',len(checks),'checks')
