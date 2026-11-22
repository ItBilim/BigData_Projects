from pathlib import Path
import sys
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.append(str(ROOT.parent/'work'/'plot_dependencies'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
DIR=OUT/'figures'
DIR.mkdir(exist_ok=True)
NAMES={'lightgbm_absolute':'LightGBM absolute','global':'Global robust score',
       'simple_absolute':'Rolling-mean forecast','isolation_forest':'Isolation Forest',
       'personal_mad':'Personal MAD','random':'Random'}
main=list(NAMES)[:4]
palette=['#157d87','#9c4a27','#435eac','#705284','#356742','#666666']
m=pd.read_csv(ROOT/'stage8_metrics'/'macro_weekly_metrics.csv')
ci=pd.read_csv(ROOT/'stage9_stability'/'confidence_intervals.csv').query("interval_type=='cluster_plus_model_seed'")
def save(fig,name):
    fig.savefig(DIR/f'{name}.svg',bbox_inches='tight')
    fig.savefig(DIR/f'{name}.png',bbox_inches='tight',dpi=300)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(8,2),layout='constrained')
for start,end,label,color in [(22,25,'History',palette[3]),(26,29,'Initial fit',palette[0]),(30,31,'Validation',palette[1]),(32,35,'Test',palette[2])]:
    ax.barh(0,end-start+1,left=start-.5,color=color)
    ax.text((start+end)/2,0,label,ha='center',va='center',color='white')
ax.set(yticks=[],xticks=range(22,36),xlabel='Dataset week',title='Temporal split and expanding-window evaluation')
save(fig,'01_temporal_split')

t=m.query('audit_fraction==0.01').sort_values('precision_at_k',ascending=False).copy()
p=ci.query("fraction==0.01 and metric=='precision'").set_index('method')
t['precision_ci_low']=t.method.map(p.ci_low);t['precision_ci_high']=t.method.map(p.ci_high)
t.to_csv(OUT/'article_main_table.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
ax.axis('off')
cells=[[NAMES.get(r.method,r.method.replace('_',' ')),
        f'{r.precision_at_k*100:.2f} [{r.precision_ci_low*100:.2f}, {r.precision_ci_high*100:.2f}]',
        f'{100*r.recall_at_k:.2f}',f'{100*r.average_precision:.2f}'] for r in t.itertuples()]
tab=ax.table(cellText=cells,colLabels=['Method','Precision % [95% CI]','Recall %','AP %'],loc='center',cellLoc='left',colWidths=[.34,.30,.16,.16])
tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.3)
for (row,col),cell in tab.get_celld().items():
    cell.set_edgecolor('white');cell.set_facecolor('#eaf1f5' if row%2 else '#f7f8fa')
ax.set_title('Top 1% audit budget | test weeks 32–35')
save(fig,'02_main_results_table')

fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for name,color in zip(main,palette):
    d=ci.query("method==@name and metric=='recall'").sort_values('fraction')
    x=100*d.fraction
    ax.plot(x,100*d.estimate,'o-',label=NAMES[name],color=color)
    ax.fill_between(x,100*d.ci_low,100*d.ci_high,color=color,alpha=.13)
ax.set(xlabel='Weekly audit budget (%)',ylabel='Injected anomalies found (%)',title='Audit budget and recall | test weeks 32–35')
ax.legend(frameon=False)
save(fig,'03_budget_recall')

s=pd.read_csv(ROOT/'stage8_metrics'/'segment_macro_metrics.csv').query('audit_fraction==0.01')
fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True,layout='constrained')
for ax,dimension,groups,title in [(axes[0],'injection_type',['dip','spike_moderate','spike_strong'],'Injection type'),
                                  (axes[1],'magnitude_segment',['low','middle','high'],'Absolute log-fold change')]:
    for i,(name,color) in enumerate(zip(main,palette)):
        d=s.loc[s.method.eq(name)&s.dimension.eq(dimension)].set_index('segment')
        ax.bar(np.arange(3)+(i-1.5)*.19,100*d.recall_within_segment.reindex(groups),.19,color=color,label=NAMES[name])
    ax.set(xticks=range(3),xticklabels=[g.replace('spike_','') for g in groups],title=title)
axes[0].set_ylabel('Recall within group (%)')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Anomaly detection at top 1% | test weeks 32–35')
save(fig,'04_type_and_magnitude')

d=pd.read_csv(ROOT/'stage7_leakage'/'information_impact_at_1pct.csv').sort_values('delta_precision_percentage_points')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
delta=d.delta_precision_percentage_points
ax.barh([NAMES.get(n,n.replace('_',' ')) for n in d.method],delta,color=np.where(delta<0,'#9c4a27','#157d87'))
ax.axvline(0,color='#333333',lw=.8)
ax.set(xlabel='Hindsight minus available-history precision (percentage points)', title='Future-information control | top 1%, weeks 32–35')
save(fig,'05_future_information')

w=pd.read_csv(ROOT/'stage8_metrics'/'weekly_seed_mean.csv').query('audit_fraction==0.01')
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for name,color in zip(main,palette):
    d=w.loc[w.method.eq(name)]
    ax.plot(d.week,100*d.precision_at_k,'o-',color=color,label=NAMES[name])
ax.set(xticks=[32,33,34,35],xlabel='Test week',ylabel='Precision at top 1% (%)',title='Weekly test performance | mean over available model seeds')
ax.legend(frameon=False)
save(fig,'06_weekly_stability')

h=pd.read_csv(OUT/'anonymous_histories.csv')
fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
for ax,(case,d) in zip(axes,h.groupby('case',sort=False)):
    d=d.sort_values('week').set_index('week').reindex(range(22,36))
    ax.plot(d.index,d.revenue,'o-',label='Observed revenue',color=palette[0])
    ax.plot(d.index,d.revenue_original,':',label='Before injection',color=palette[3])
    injected=d.is_injected.eq(1)
    ax.scatter(d.index[injected],d.revenue[injected],facecolors='none',edgecolors=palette[1],s=75,label='Injection')
    ax.axvline(34,color='#666666',ls='--',lw=.9)
    ax.set(title=case,ylabel='Revenue (source units)')
axes[0].legend(frameon=False,ncol=3,fontsize=8)
axes[-1].set(xlabel='Dataset week',xticks=range(22,36))
fig.suptitle('Anonymous histories | LightGBM absolute, test week 34')
save(fig,'07_subscriber_histories')
