from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
ci=pd.read_csv(OUT/'confidence_intervals.csv')
diff=pd.read_csv(OUT/'paired_method_differences.csv')
assert len(ci)==480 and len(diff)==432
assert ci[['estimate','ci_low','ci_high']].notna().all().all()
assert ci.ci_low.le(ci.ci_high).all()
point=pd.read_csv(ROOT/'stage8_metrics'/'macro_weekly_metrics.csv')
mapping={'precision':'precision_at_k','recall':'recall_at_k','lift':'lift_vs_expected_random','per100':'anomalies_per_100_checks','average_precision':'average_precision','roc_auc':'roc_auc'}
for metric,column in mapping.items():
    checked=ci.loc[ci.metric.eq(metric)].merge(point[['method','audit_fraction',column]],left_on=['method','fraction'],right_on=['method','audit_fraction'])
    assert np.allclose(checked.estimate,checked[column]),metric
seed=pd.read_csv(OUT/'model_seed_results.csv')
seed_stats=seed.groupby(['method','fraction'])[['precision','recall','average_precision','roc_auc']].agg(['mean','std','min','max'])
seed_stats.columns=['_'.join(c) for c in seed_stats.columns]
seed_stats.reset_index().to_csv(OUT/'model_seed_variability.csv',index=False)
weekly=pd.read_csv(ROOT/'stage8_metrics'/'weekly_seed_mean.csv')
temporal=weekly.groupby(['method','audit_fraction'])[['precision_at_k','recall_at_k','average_precision','roc_auc']].agg(['mean','std','min','max'])
temporal.columns=['_'.join(c) for c in temporal.columns]
temporal.reset_index().to_csv(OUT/'four_week_stability.csv',index=False)
injections=pd.read_csv(OUT/'injection_seed_weekly.csv')
assert len(injections)==800
assert set(injections.injection_seed)=={42,43,44,45,46}
assert set(injections.week)=={32,33,34,35}
assert not injections.duplicated(['injection_seed','method','week','fraction']).any()
primary=diff.loc[diff.primary & diff.interval_type.eq('cluster_plus_model_seed')]
assert len(primary)==1
primary.to_csv(OUT/'primary_comparison.csv',index=False)
draws=np.load(OUT/'bootstrap_distributions.npz')
names=draws['methods'].tolist(); mi=names.index('lightgbm_absolute'); ref=names.index('global')
convergence=[]
for size in [100,250,500,1000]:
    for kind,array_name in [('cluster_conditional','conditional'),('cluster_plus_model_seed','hierarchical')]:
        values=draws[array_name][:size,mi,1,0]-draws[array_name][:size,ref,1,0]
        low,high=np.quantile(values,[.025,.975])
        convergence.append({'replicates':size,'interval_type':kind,'primary_difference_ci_low':low,'primary_difference_ci_high':high})
pd.DataFrame(convergence).to_csv(OUT/'bootstrap_interval_convergence.csv',index=False)
view=ci.loc[ci.metric.eq('precision')&ci.fraction.eq(.01)&ci.interval_type.eq('cluster_plus_model_seed'),['method','estimate','ci_low','ci_high']].sort_values('estimate',ascending=False)
display=view.copy();display[['estimate','ci_low','ci_high']]*=100
effect=primary.iloc[0]
injection_stats=pd.read_csv(OUT/'injection_seed_variability.csv')
iv=injection_stats.loc[injection_stats.fraction.eq(.01),['method','precision_mean','precision_std','precision_min','precision_max']].copy()
iv.iloc[:,1:]*=100
lines=['Сенімділік пен тұрақтылық нәтижелері','',
'Дайын аномалия дерегінде 1000 жұпталған абоненттік cluster bootstrap орындалды. Абоненттердің біріктірілген сынақ жиыны қайта таңдалып, бірдей салмақтар барлық әдіс, апта және seed үшін қолданылды. Әр қайталауда әр аптаның N және ceil(N*q) мәні қайта есептелді. Шекаралық жазбаның тек K-ға қажет көшірмелері алынды.',
'Негізгі салыстыру LightGBM абсолют қалдығы және жалпы статистикалық балл, Precision@1%. Бұл салыстыру bootstrap алдында, бірақ нүктелік нәтижелер көрілгеннен кейін бекітілді. Сондықтан қорытынды барлау сипатында. Нәтижені тәуелсіз жаңа кезеңде растау қажет.',
'Екі интервал берілген. Cluster conditional бес үйретілген seed бойынша орташаланған модельдердің сынақ популяциясына қатысты белгісіздігін береді. Cluster plus model seed сол кластерлермен бірге бес seed индексін жұпталған түрде қайта таңдайды. Модельдер bootstrap ішінде қайта оқытылған жоқ, оқыту жиынының белгісіздігі толық қамтылмайды.',
'Негізгі кестеде cluster plus model seed percentile 95% интервалдары көрсетілген. Precision мәндері пайызбен.',
display.to_string(index=False),'',
f"Негізгі айырма {100*effect['difference']:.4f} пайыздық тармақ, 95% интервал [{100*effect['ci_low']:.4f}; {100*effect['ci_high']:.4f}]. Орталықтандырылған bootstrap p={effect['p_centered_bootstrap_two_sided']:.6f}, тоғыз салыстыруға Holm түзетуінен кейін p={effect['holm_adjusted_p']:.6f}.",
'Қосымша отбасы тоғыз әдісті global әдісімен Precision@1% бойынша салыстырады. Негізгі салыстыру осы отбасына да енгізілді. Holm түзетуі әр интервал түрі үшін бөлек қолданылды. Басқа көлемдер, метрикалар мен интервалдар барлау нәтижелері, олардың p мәндеріне көптік түзету жасалмаған. Кестедегі 95% интервалдар жеке интервалдар, бір мезгілдік family-wise интервалдар емес.',
'Барлық p мәндері bootstrap айырмасының бақыланған айырмаға қатысты орталықтандырылған таралуынан жуық есептелді. Олар permutation тесті емес. 1000 қайталауда төменгі ажыратылым 1/1001. AP және AUC тең баллдар топтарын ескереді.',
'Апталар 32–35 болып бекітілген. Апталар қайта таңдалған жоқ. Уақыттық режимдердің саны төрт қана, ұзақ мерзімді немесе жаңа маусымдағы тұрақтылық туралы қорытынды жасауға жеткіліксіз. Аптааралық стандартты ауытқу сенімділік интервалы ретінде берілмейді.',
'', 'Инъекция seed өзгергіштігі',
'Жаңа d1_f2_r0.01 сценарийі бес инъекция seed бойынша қайта бағаланды. Модель seed=42 болып тұрақты қалды. Бұл бастапқы түсімдегі табиғи ауытқулар сақталған, бір апталық екі есе өсім және 1% жиілік жағдайы. Басқа тип пен жиілікке қорытынды таралмайды. Модель seed пен инъекция seed өзара әсері бұл тәжірибеде бағаланбайды.',
'Төменде бес инъекция seed арасындағы орташа, sample SD, минимум және максимум precision пайызбен берілген.',
iv.to_string(index=False),
'Дайын деректің бастапқы инъекция seed-і белгісіз. Сондықтан дайын деректегі кластерлік интервал инъекция генерациясының белгісіздігін қамтымайды. Жаңа сценарийдегі бес seed нәтижесін сол интервалға тікелей қосуға болмайды.',
'240 жаңа сценарийдің біреуінің бес seed-і осы кезеңде бағаланды. Қалған сценарийлер бойынша модель сапасы бұл есепке кірмейді.',
'Тексерулер: нүктелік бағалар сегізінші кезеңге сәйкес. Салмақталған top-K нақты көшірмелерді іріктеумен салыстырылды, шекаралық көшірме, AP/AUC tie өңдеуі және ортақ кластер салмағы тексерілді.']
(OUT/'stability_report_kk.txt').write_text('\n\n'.join(lines),encoding='utf-8-sig')
(OUT/'completion_checks.json').write_text(json.dumps({'bootstrap_replicates':1000,'interval_rows':len(ci),'difference_rows':len(diff),'synthetic_weekly_rows':len(injections),'point_estimates_match_stage8':True,
 'sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['confidence_intervals.csv','paired_method_differences.csv','injection_seed_variability.csv']}},indent=2),encoding='utf-8')
print(display.to_string(index=False))
print(primary.to_string(index=False))
