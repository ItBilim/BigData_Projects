"""Exact top-K on a subscriber-cluster resample represented by row weights."""
import numpy as np

def weighted_metrics(weights,labels,tie_ends,fractions):
    w=np.asarray(weights,dtype=float)
    y=np.asarray(labels,dtype=float)
    cw=np.cumsum(w); ct=np.cumsum(w*y)
    total=cw[-1]; positive=ct[-1]; negative=total-positive
    if total<=0 or positive<=0 or negative<=0:
        raise ValueError('Degenerate bootstrap population')
    k=np.maximum(1,np.ceil(total*np.asarray(fractions))).astype(int)
    boundary=np.searchsorted(cw,k,side='left')
    prior_w=np.where(boundary>0,cw[np.maximum(boundary-1,0)],0)
    prior_tp=np.where(boundary>0,ct[np.maximum(boundary-1,0)],0)
    tp=prior_tp+(k-prior_w)*y[boundary]
    precision=tp/k; recall=tp/positive
    group_cw=cw[tie_ends]; group_ct=ct[tie_ends]
    group_positive=np.diff(np.r_[0,group_ct])
    group_negative=np.diff(np.r_[0,group_cw-group_ct])
    valid=group_cw>0
    ap=np.sum(group_positive[valid]*group_ct[valid]/group_cw[valid])/positive
    negatives_before=np.r_[0,(group_cw-group_ct)[:-1]]
    # Score order is descending, so lower-ranked negatives count as concordant.
    auc=np.sum(group_positive*(negative-negatives_before-.5*group_negative))/(positive*negative)
    return np.column_stack([precision,recall,precision/(positive/total),100*precision,
                            np.full(len(k),ap),np.full(len(k),auc)])
