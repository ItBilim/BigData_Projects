"""Metrics preserve the weekly top-K selection before segment attribution."""
import numpy as np

def top_k(scores, identifiers, k):
    scores=np.asarray(scores)
    identifiers=np.asarray(identifiers,dtype=str)
    if not np.isfinite(scores).all() or not 0<k<=len(scores):
        raise ValueError('Invalid score or K')
    return np.lexsort((identifiers,-scores))[:k]

def selection_metrics(labels,selected,random_precision):
    labels=np.asarray(labels,dtype=int)
    n=len(labels); k=len(selected); positives=int(labels.sum())
    tp=int(labels[selected].sum())
    precision=tp/k if k else np.nan
    return {'population_rows':n,'positive_rows':positives,'selected_rows':k,
      'true_positives':tp,'selected_without_injection':k-tp,
      'precision_at_k':precision,'recall_at_k':tp/positives if positives else np.nan,
      'expected_random_precision':positives/n if n else np.nan,
      'lift_vs_expected_random':precision/(positives/n) if positives and k else np.nan,
      'same_seed_random_precision':random_precision,
      'lift_vs_same_seed_random':precision/random_precision if random_precision>0 else np.nan,
      'anomalies_per_100_checks':100*precision}
