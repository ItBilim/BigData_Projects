import numpy as np
from bootstrap_metrics import weighted_metrics

fractions=[.25,.5,1.]
w=np.array([2,0,3,1]);y=np.array([1,0,0,1]);ends=np.array([0,2,3])
result=weighted_metrics(w,y,ends,fractions)
# Explicit expanded sample, score groups [3,2,2,1], tie keys already ordered.
expanded=np.repeat(y,w)
for i,q in enumerate(fractions):
    k=int(np.ceil(len(expanded)*q));tp=expanded[:k].sum()
    assert np.isclose(result[i,0],tp/k)
    assert np.isclose(result[i,1],tp/expanded.sum())
assert np.isclose(result[0,4],(2*1+1*.5)/3)
assert np.isclose(result[0,5],2/3)
# Boundary may include only one copy of a repeatedly sampled row.
assert result[0,0]==1 and result[1,0]==2/3
print('Weighted bootstrap matches explicit duplicates, tie-aware AP/AUC and partial boundary selection')
