import json
from pathlib import Path
import numpy as np
from metric_functions import top_k,selection_metrics

chosen=top_k(np.array([.5,.5,.1,.1]),np.array(['02','01','04','03']),2)
assert chosen.tolist()==[1,0]
m=selection_metrics(np.array([1,0,1,0]),chosen,.5)
assert m['true_positives']==1 and m['selected_without_injection']==1
assert m['precision_at_k']==.5 and m['recall_at_k']==.5
assert m['lift_vs_expected_random']==1 and m['anomalies_per_100_checks']==50
empty=selection_metrics(np.array([0,0]),np.array([0]),0)
assert np.isnan(empty['recall_at_k']) and np.isnan(empty['lift_vs_same_seed_random'])
try: top_k(np.array([np.nan]),['a'],1)
except ValueError: pass
else: raise AssertionError('Nonfinite score accepted')
checks=['lexicographic_tie_break','hand_computed_precision_recall_counts_lift',
        'zero_positive_denominator','zero_random_denominator','invalid_score_rejected']
(Path(__file__).resolve().parent/'metric_unit_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Passed',len(checks),'metric checks')
