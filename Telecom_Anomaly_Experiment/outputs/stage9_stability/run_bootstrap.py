from pathlib import Path
import hashlib,json,os,tarfile,time
import numpy as np
import pandas as pd
from bootstrap_metrics import weighted_metrics

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent
config=json.loads((OUT/'config.json').read_text())
seeds=config['model_seeds'];weeks=config['test_weeks'];fractions=[.005,.01,.02,.05]
methods=list(json.loads((ROOT/'stage5_methods'/'selected_parameters.json').read_text()))
metrics=config['metrics'];B=config['bootstrap_replicates']
lock={'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
      'registered_utc':pd.Timestamp.now(tz='UTC').isoformat(),
      'selected_parameters_sha256':hashlib.sha256((ROOT/'stage5_methods'/'selected_parameters.json').read_bytes()).hexdigest()}
(OUT/'analysis_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
print('Loading paired test rows and score ranks',flush=True)
tables={w:pd.read_csv(ROOT/'stage7_leakage'/f'test_keys_{w}.csv.gz',dtype={'msisdn':'string'}) for w in weeks}
ids=sorted(set().union(*(set(t.msisdn) for t in tables.values())))
idmap={sid:i for i,sid in enumerate(ids)}
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz','r:gz') as tar:
    labels=pd.concat([c.loc[c.week.isin(weeks)] for c in pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'),dtype={'msisdn':'string'},usecols=['msisdn','week','is_injected'],chunksize=200000)],ignore_index=True)
ranks={}; index={}; points=np.zeros((len(methods),len(seeds),len(weeks),4,6))
for wi,week in enumerate(weeks):
    t=tables[week].merge(labels,on=['msisdn','week'],validate='one_to_one')
    users=t.msisdn.map(idmap).to_numpy(dtype='int32'); y=t.is_injected.to_numpy(dtype='int8')
    index[week]=users
    for mi,method in enumerate(methods):
        deterministic=method not in ['random','isolation_forest','lightgbm_absolute','lightgbm_normalized']
        for si,seed in enumerate(seeds):
            actual=42 if deterministic else seed
            key=(week,method,actual)
            if key not in ranks:
                scores=np.load(ROOT/'stage7_leakage'/f'scores_{method}_causal_w{week}_s{actual}.npz')['scores']
                assert len(scores)==len(t) and np.isfinite(scores).all()
                # Keys are verified to already have the declared textual identifier order.
                assert t.msisdn.tolist()==sorted(t.msisdn.tolist())
                order=np.argsort(-scores,kind='stable')
                sorted_scores=scores[order]
                ends=np.r_[np.flatnonzero(sorted_scores[:-1]!=sorted_scores[1:]),len(scores)-1].astype('int32')
                ranks[key]=(users[order],y[order],ends)
            uid,yy,ends=ranks[key]
            points[mi,si,wi]=weighted_metrics(np.ones(len(yy)),yy,ends,fractions)
point=points.mean(axis=(1,2))
# Cluster resampling holds week structure fixed and repeats all rows of a selected subscriber.
conditional=np.zeros((B,len(methods),4,6)); hierarchical=np.zeros_like(conditional)
rng=np.random.default_rng(config['bootstrap_seed'])
seed_rng=np.random.default_rng(config['bootstrap_seed']+1)
start=time.perf_counter()
checkpoint=OUT/'bootstrap_checkpoint.npz'
completed=0
if checkpoint.exists() and os.environ.get('TELECOM_FRESH_BOOTSTRAP')!='1':
    with np.load(checkpoint) as saved:
        completed=int(saved['completed'])
        conditional[:completed]=saved['conditional']
        hierarchical[:completed]=saved['hierarchical']
    for _ in range(completed):
        rng.integers(0,len(ids),size=len(ids))
        seed_rng.integers(0,len(seeds),size=len(seeds))
    print('Resuming from',completed,flush=True)
for b in range(completed,B):
    weights=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids))
    per_seed=np.zeros((len(methods),len(seeds),4,6))
    cache={}
    for wi,week in enumerate(weeks):
        for mi,method in enumerate(methods):
            deterministic=method not in ['random','isolation_forest','lightgbm_absolute','lightgbm_normalized']
            for si,seed in enumerate(seeds):
                actual=42 if deterministic else seed;key=(week,method,actual)
                if key not in cache:
                    uid,yy,ends=ranks[key]
                    cache[key]=weighted_metrics(weights[uid],yy,ends,fractions)
                per_seed[mi,si]+=cache[key]/len(weeks)
    conditional[b]=per_seed.mean(axis=1)
    sampled_seeds=seed_rng.integers(0,len(seeds),size=len(seeds))
    hierarchical[b]=per_seed[:,sampled_seeds].mean(axis=1)
    if (b+1)%25==0:
        print('Bootstrap',b+1,'/',B,'elapsed',round(time.perf_counter()-start,1),'seconds',flush=True)
        np.savez_compressed(OUT/'bootstrap_checkpoint.npz',completed=b+1,conditional=conditional[:b+1],hierarchical=hierarchical[:b+1])
np.savez_compressed(OUT/'bootstrap_distributions.npz',conditional=conditional,hierarchical=hierarchical,
                    point=point,methods=np.array(methods),metrics=np.array(metrics),fractions=np.array(fractions))
intervals=[];differences=[];ref=methods.index('global')
for interval_type,draws in [('cluster_conditional',conditional),('cluster_plus_model_seed',hierarchical)]:
    for mi,method in enumerate(methods):
        for qi,q in enumerate(fractions):
            for ki,metric in enumerate(metrics):
                low,high=np.quantile(draws[:,mi,qi,ki],[.025,.975])
                intervals.append({'method':method,'fraction':q,'metric':metric,'estimate':point[mi,qi,ki],
                    'ci_low':low,'ci_high':high,'interval_type':interval_type,'bootstrap_replicates':B})
                if method=='global':continue
                observed=point[mi,qi,ki]-point[ref,qi,ki]
                paired=draws[:,mi,qi,ki]-draws[:,ref,qi,ki]
                low,high=np.quantile(paired,[.025,.975])
                p=(1+np.count_nonzero(np.abs(paired-observed)>=abs(observed)))/(B+1)
                differences.append({'method':method,'reference':'global','fraction':q,'metric':metric,
                    'difference':observed,'ci_low':low,'ci_high':high,'interval_type':interval_type,
                    'p_centered_bootstrap_two_sided':p,'primary':method=='lightgbm_absolute' and q==.01 and metric=='precision'})
pd.DataFrame(intervals).to_csv(OUT/'confidence_intervals.csv',index=False)
diff=pd.DataFrame(differences)
diff['holm_adjusted_p']=np.nan
for kind in ['cluster_conditional','cluster_plus_model_seed']:
    selected=diff.loc[diff.interval_type.eq(kind)&diff.fraction.eq(.01)&diff.metric.eq('precision')].sort_values('p_centered_bootstrap_two_sided')
    adjusted=np.minimum(1,np.maximum.accumulate(selected.p_centered_bootstrap_two_sided.to_numpy()*np.arange(len(selected),0,-1)))
    diff.loc[selected.index,'holm_adjusted_p']=adjusted
diff.to_csv(OUT/'paired_method_differences.csv',index=False)
seed_rows=[]
for mi,method in enumerate(methods):
    for si,seed in enumerate(seeds):
        for qi,q in enumerate(fractions):
            seed_rows.append({'method':method,'seed':seed,'fraction':q,**dict(zip(metrics,points[mi,si].mean(axis=0)[qi]))})
pd.DataFrame(seed_rows).to_csv(OUT/'model_seed_results.csv',index=False)
print('DONE bootstrap seconds',time.perf_counter()-start,flush=True)
