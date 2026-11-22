"""Causal calendar-week features and training-only scale calibration."""
import warnings
import numpy as np
import pandas as pd

VERSION = '2.0.0'
BASE = ('current_revenue','lag_1','lag_2','lag_3','lag_4','absolute_change_1',
        'prior_mean_2','prior_mean_4','prior_median_4','prior_std_4','prior_mad_4',
        'deviation_from_prior_mean','deviation_from_prior_median','prior_zero_share_4',
        'prior_observation_count','weeks_since_last_positive','prior_min_4','prior_max_4',
        'deviation_from_prior_min','deviation_from_prior_max','prior_unfilled_count',
        'prior_filled_share_4','no_prior_positive')
DERIVED = ('relative_change_1','scaled_deviation_from_prior_median')
DETECTOR_FEATURES = BASE + DERIVED
CURRENT_DEPENDENT = {'current_revenue','absolute_change_1','relative_change_1',
    'deviation_from_prior_mean','deviation_from_prior_median',
    'scaled_deviation_from_prior_median','deviation_from_prior_min','deviation_from_prior_max'}
FORECAST_FEATURES = tuple(c for c in DETECTOR_FEATURES if c not in CURRENT_DEPENDENT)
FORBIDDEN = {'msisdn','week','revenue_original','is_injected','injection_type','outlier_global','spike_local','is_filled'}

def validate_features(columns, forecasting=False):
    columns=list(columns)
    allowed=set(FORECAST_FEATURES if forecasting else DETECTOR_FEATURES)
    if len(columns)!=len(set(columns)) or set(columns)-allowed or set(columns)&FORBIDDEN or any(c.startswith('target_') for c in columns):
        raise ValueError('Unregistered or forbidden feature')

def build_features(panel):
    """Input is complete from each subscriber's first appearance; no gap filling."""
    needed=['msisdn','week','revenue','is_filled']
    df=panel[needed].sort_values(['msisdn','week']).reset_index(drop=True)
    if df.duplicated(['msisdn','week']).any(): raise ValueError('Duplicate calendar keys')
    g=df.groupby('msisdn',sort=False)
    gaps=g.week.diff()
    if gaps.dropna().ne(1).any(): raise ValueError('Missing calendar weeks must be resolved explicitly')
    if not np.isfinite(df.revenue).all(): raise ValueError('Nonfinite revenue')
    if not df.is_filled.isin([0,1]).all(): raise ValueError('Invalid provenance flag')
    lag=np.column_stack([g.revenue.shift(i).to_numpy() for i in range(1,5)])
    filled=np.column_stack([g.is_filled.shift(i).to_numpy() for i in range(1,5)])
    count=np.isfinite(lag).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean2=np.nanmean(lag[:,:2],axis=1)
        mean4=np.nanmean(lag,axis=1)
        median=np.nanmedian(lag,axis=1)
        mad=np.nanmedian(np.abs(lag-median[:,None]),axis=1)
        std=np.nanstd(lag,axis=1,ddof=1)
        low=np.nanmin(lag,axis=1); high=np.nanmax(lag,axis=1)
    zero=np.divide((lag==0).sum(axis=1),count,out=np.full(len(df),np.nan),where=count>0)
    fillshare=np.divide(np.nansum(filled,axis=1),count,out=np.full(len(df),np.nan),where=count>0)
    active=df.week.where(df.revenue.gt(0)).groupby(df.msisdn,sort=False).ffill()
    previous_active=active.groupby(df.msisdn,sort=False).shift()
    no_active=previous_active.isna()
    unfilled=df.is_filled.eq(0).astype('int32')
    prior_unfilled=unfilled.groupby(df.msisdn,sort=False).cumsum()-unfilled
    revenue=df.revenue.to_numpy()
    values=[revenue,*lag.T,np.abs(revenue-lag[:,0]),mean2,mean4,median,std,mad,
            revenue-mean4,revenue-median,zero,g.cumcount().to_numpy(),
            (df.week-previous_active).to_numpy(),low,high,revenue-low,revenue-high,
            prior_unfilled.to_numpy(),fillshare,no_active.astype('int8').to_numpy()]
    features=pd.DataFrame(dict(zip(BASE,values)))
    return df[['msisdn','week','is_filled']],features

def fit_scale(features, mask):
    """Use positive training MAD median; fall back to positive training revenue."""
    mad=features.loc[mask,'prior_mad_4'].to_numpy(dtype=float)*1.4826
    positive=mad[np.isfinite(mad)&(mad>0)]
    source='median_positive_training_scaled_mad'
    if not len(positive):
        revenue=features.loc[mask,'current_revenue'].to_numpy(dtype=float)
        positive=revenue[np.isfinite(revenue)&(revenue>0)]
        source='median_positive_training_revenue'
    floor=float(np.median(positive)) if len(positive) else 1.0
    return {'floor':floor,'source':source,'training_rows':int(np.count_nonzero(mask)),
            'rule':'median positive training 1.4826*MAD; revenue fallback; unit fallback'}

def transform(features,state):
    result=features.copy()
    floor=state['floor']
    result['relative_change_1']=(result.current_revenue-result.lag_1)/np.maximum(np.abs(result.lag_1),floor)
    result['scaled_deviation_from_prior_median']=result.deviation_from_prior_median/np.maximum(1.4826*result.prior_mad_4,floor)
    validate_features(result.columns)
    return result
