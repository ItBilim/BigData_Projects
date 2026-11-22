import numpy as np
import pandas as pd
from methods import AnomalyMethod
from feature_engineering import build_features,transform,fit_scale,FORECAST_FEATURES

sample=pd.DataFrame({'msisdn':['001']*8+['002']*8,'week':list(range(22,30))*2,
 'revenue':[1.,2.,3.,4.,5.,6.,7.,20.]+[5.]*8,'is_filled':[0]*16})
keys,base=build_features(sample)
fit=keys.week.between(26,28).to_numpy()
score=keys.week.eq(29).to_numpy()
state=fit_scale(base,fit)
x=transform(base,state)
for name in ['random','global','weekly_absolute','weekly_relative','personal_mad','simple_absolute','simple_normalized','isolation_forest','lightgbm_absolute','lightgbm_normalized']:
    params={'n_estimators':5,'max_samples':4} if name=='isolation_forest' else {'n_estimators':5,'min_child_samples':1,'verbosity':-1} if name.startswith('lightgbm') else {}
    model=AnomalyMethod(name,params)
    model.fit(x.loc[fit],scale_floor=state['floor'])
    scores=model.score(x.loc[score])
    assert len(scores)==2 and np.isfinite(scores).all(),name
    if name.startswith('lightgbm'):
        assert model.model.feature_name_==list(FORECAST_FEATURES)
first=AnomalyMethod('random',seed=42).fit(x).score(x)
assert np.array_equal(first,AnomalyMethod('random',seed=42).fit(x).score(x))
assert not np.array_equal(first,AnomalyMethod('random',seed=43).fit(x).score(x))
print('All 10 method variants finite; forecast inputs and seed reproducibility checked')
