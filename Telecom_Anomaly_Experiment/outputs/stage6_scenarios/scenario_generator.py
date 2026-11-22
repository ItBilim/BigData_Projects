"""Label-blind, phase-independent, deterministic revenue perturbations."""
import hashlib
import numpy as np
import pandas as pd
_POSITION_CACHE = {}
_PHASE_CACHE = {}

def rng_for(seed,phase):
    # Same phase/seed keys across severities and frequencies enable paired comparisons.
    digest=hashlib.sha256(f'{seed}|{phase}|stage6-v1'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))

def inject_phase(panel,candidates,phase,weeks,seed,rate,factor,duration,min_delta,scenario):
    phase_key=(id(candidates),tuple(weeks))
    if phase_key not in _PHASE_CACHE:
        eligible=candidates.loc[candidates.week.isin(weeks)].sort_values(['msisdn','week'])
        starts=list(zip(eligible.msisdn,eligible.week))
        _PHASE_CACHE[phase_key]=(candidates,starts,set(starts))
    _,starts,keys=_PHASE_CACHE[phase_key]
    target=int(np.floor(len(starts)*rate))
    events=target//duration
    cache_key=id(panel)
    if cache_key not in _POSITION_CACHE:
        _POSITION_CACHE[cache_key]=(panel,{(sid,int(w)):int(i) for i,sid,w in panel[['msisdn','week']].itertuples()})
    positions=_POSITION_CACHE[cache_key][1]
    revenues=panel.revenue.to_numpy()
    order=rng_for(seed,phase).permutation(len(starts))
    used=set(); records=[]; accepted=0
    for j in order:
        if accepted>=events: break
        sid,week=starts[j]
        block=[(sid,int(week)+offset) for offset in range(duration)]
        if any(k not in keys or k in used for k in block): continue
        proposals=[]
        for key in block:
            row=positions[key]; old=float(revenues[row])
            if factor>1:
                delta=max(old*(factor-1),min_delta)
                new=old+delta
            else:
                new=old*factor
                if old-new<min_delta: break
            proposals.append((key,row,old,new))
        if len(proposals)!=duration: continue
        event=f'{scenario}|s{seed}|{phase}|e{accepted}'
        for key,row,old,new in proposals:
            records.append({'scenario':scenario,'seed':seed,'phase':phase,'event_id':event,
                'msisdn':key[0],'week':key[1],'panel_row':row,'original_revenue':old,
                'new_revenue':new,'type':'spike' if factor>1 else 'dip','duration':duration,
                'requested_factor':factor,'effective_factor':new/old if old else np.nan,
                'absolute_delta':new-old,'requested_rate':rate})
        used.update(block); accepted+=1
    if accepted<events: raise ValueError(f'Insufficient valid events {scenario} {phase}')
    return pd.DataFrame(records),{'phase':phase,'candidate_rows':len(starts),'target_rows':target,
        'actual_rows':len(records),'actual_rate':len(records)/len(starts) if len(starts) else 0,
        'events':accepted,'duration':duration}

def apply_journal(panel,journal):
    # Avoid copying the optional large lookup cache into derived scenario data.
    result=pd.DataFrame({c:panel[c].copy() for c in panel.columns})
    if journal.empty: return result
    if journal.duplicated(['msisdn','week']).any(): raise ValueError('Overlapping perturbations')
    rows=journal.panel_row.to_numpy(dtype=int)
    if not np.array_equal(result.iloc[rows].msisdn.to_numpy(),journal.msisdn.to_numpy()): raise ValueError('Wrong panel ordering')
    if not np.array_equal(result.iloc[rows].week.to_numpy(),journal.week.to_numpy()): raise ValueError('Wrong weeks')
    if not np.allclose(result.iloc[rows].revenue,journal.original_revenue,rtol=0,atol=1e-10): raise ValueError('Wrong base revenue')
    result.loc[rows,'revenue']=journal.new_revenue.to_numpy()
    return result
