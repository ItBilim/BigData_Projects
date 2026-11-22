"""Shared fit/score interface. Scores increase with anomalousness."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work'/'ml_dependencies'))
sys.path.insert(0,str(ROOT/'outputs'/'stage4_features'))
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from feature_engineering import FORECAST_FEATURES, DETECTOR_FEATURES, validate_features

class AnomalyMethod:
    def __init__(self,name,params=None,seed=42,threads=4):
        self.name=name; self.params=params or {}; self.seed=seed; self.threads=threads

    def fit(self,x,y=None,scale_floor=1.):
        validate_features(x.columns)
        self.floor=scale_floor
        if self.name=='global':
            self.center=float(x.current_revenue.median())
            self.spread=max(float((x.current_revenue-self.center).abs().median())*1.4826,self.floor)
        elif self.name=='isolation_forest':
            self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            xx=self.imputer.fit_transform(x[list(DETECTOR_FEATURES)])
            self.model=IsolationForest(**self.params,random_state=self.seed,n_jobs=self.threads)
            self.model.fit(xx)
        elif self.name.startswith('lightgbm'):
            columns=list(FORECAST_FEATURES)
            validate_features(columns,forecasting=True)
            self.model=LGBMRegressor(**self.params,random_state=self.seed,n_jobs=self.threads)
            self.model.fit(x[columns],x.current_revenue)
        return self

    def score(self,x):
        validate_features(x.columns)
        if self.name=='random': return np.random.default_rng(self.seed).random(len(x))
        if self.name=='global': return np.abs(x.current_revenue.to_numpy()-self.center)/self.spread
        if self.name=='weekly_absolute': return x.absolute_change_1.to_numpy()
        if self.name=='weekly_relative': return np.abs(x.relative_change_1.to_numpy())
        if self.name=='personal_mad': return np.abs(x.scaled_deviation_from_prior_median.to_numpy())
        if self.name=='isolation_forest':
            return -self.model.score_samples(self.imputer.transform(x[list(DETECTOR_FEATURES)]))
        if self.name.startswith('simple_'):
            predicted=x.prior_mean_4.to_numpy()
        elif self.name.startswith('lightgbm'):
            predicted=np.maximum(0,self.model.predict(x[list(FORECAST_FEATURES)]))
        else: raise ValueError(self.name)
        residual=np.abs(x.current_revenue.to_numpy()-predicted)
        if self.name.endswith('normalized'):
            residual=residual/np.maximum(x.prior_std_4.to_numpy(),self.floor)
        return residual
