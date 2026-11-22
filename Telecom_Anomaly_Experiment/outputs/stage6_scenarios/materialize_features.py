"""Rebuild complete causal history after applying shared scenario journals."""
import json
import sys
from pathlib import Path
import pandas as pd
from scenario_generator import apply_journal
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'stage4_features'))
from feature_engineering import build_features,fit_scale,transform

def scenario_features(journal_name,fold_id,training_journal=None):
    panel=pd.read_csv(OUT/'original_panel.csv.gz',dtype={'msisdn':'string'})
    journal=pd.read_csv(OUT/journal_name,dtype={'msisdn':'string'})
    if training_journal:
        train=pd.read_csv(OUT/training_journal,dtype={'msisdn':'string'})
        journal=pd.concat([journal,train],ignore_index=True)
    changed=apply_journal(panel,journal)
    keys,base=build_features(changed)
    manifest=pd.read_csv(OUT.parent/'stage2_protocol'/'cohort_manifest.csv.gz',dtype={'msisdn':'string'})
    mapping=manifest.merge(keys.reset_index(names='feature_row'),on=['msisdn','week','is_filled'],validate='one_to_one')
    folds=json.loads((OUT.parent/'stage3_splits'/'temporal_splits.json').read_text())['folds']
    fold=next(f for f in folds if f['fold_id']==fold_id)
    fit_rows=mapping.loc[mapping.week.isin(fold['fit_weeks']) & mapping.cohort.eq('primary'),'feature_row']
    mask=keys.index.isin(fit_rows)
    state=fit_scale(base,mask)
    return keys,transform(base,state),state,mapping
