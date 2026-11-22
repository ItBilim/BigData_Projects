"""Frozen evaluation contract. No model fitting is performed here."""
import hashlib
import json

PROTOCOL_VERSION = '1.0.0'
KEY_COLUMNS = ('msisdn', 'week')
MIN_PRIOR_WEEKS = 4
AUDIT_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
EVALUATION_COLUMNS = (
    'revenue_original', 'is_injected', 'injection_type',
    'outlier_global', 'spike_local', 'target_next_week',
    'target_next_is_filled',
)
DETECTOR_FEATURES = (
    'current_revenue', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
    'absolute_change_1', 'relative_change_1',
    'prior_mean_2', 'prior_mean_4', 'prior_median_4',
    'prior_std_4', 'prior_mad_4', 'deviation_from_prior_median',
    'scaled_deviation_from_prior_median', 'prior_zero_share_4',
    'prior_observation_count', 'weeks_since_last_positive',
)
FORECAST_FEATURES = tuple(c for c in DETECTOR_FEATURES if c not in (
    'current_revenue', 'absolute_change_1', 'relative_change_1',
    'deviation_from_prior_median', 'scaled_deviation_from_prior_median',
))

def validate_feature_columns(columns, forecasting=False):
    allowed = set(FORECAST_FEATURES if forecasting else DETECTOR_FEATURES)
    columns = list(columns)
    if len(columns) != len(set(columns)):
        raise ValueError('Duplicate feature names')
    forbidden = set(EVALUATION_COLUMNS) | set(KEY_COLUMNS)
    invalid = set(columns) - allowed
    invalid |= set(columns) & forbidden
    invalid |= {c for c in columns if c.startswith('target_')}
    if invalid:
        raise ValueError(f'Forbidden or unregistered features: {sorted(invalid)}')

def validate_fit_weeks(fit_weeks, scoring_week):
    if not fit_weeks or max(fit_weeks) >= scoring_week:
        raise ValueError('Fitting requires nonempty strictly earlier weeks')

def validate_scored_keys(expected_keys, actual_keys):
    expected = list(expected_keys)
    actual = list(actual_keys)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Scored keys differ from the shared evaluation cohort')

def contract():
    return {
        'version': PROTOCOL_VERSION, 'minimum_prior_calendar_weeks': MIN_PRIOR_WEEKS,
        'keys': KEY_COLUMNS, 'detector_features': DETECTOR_FEATURES,
        'forecast_features': FORECAST_FEATURES, 'evaluation_columns': EVALUATION_COLUMNS,
        'audit_fractions': AUDIT_FRACTIONS,
        'history_source': '2_weekly_revenue_panel.csv',
        'evaluation_population': 'Rows present in 3_weekly_revenue_anomalies.csv',
        'filled_history': 'Retain panel zero and is_filled for history accounting',
        'short_history': 'Separate supplementary cohort, never pooled into primary ranking',
    }

def fingerprint():
    return hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()

validate_feature_columns(DETECTOR_FEATURES)
validate_feature_columns(FORECAST_FEATURES, forecasting=True)
assert not set(DETECTOR_FEATURES) & set(EVALUATION_COLUMNS)
