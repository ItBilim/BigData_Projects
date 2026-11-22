from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
w=pd.read_csv(OUT/'weekly_metrics.csv')
s=pd.read_csv(OUT/'segment_weekly_metrics.csv.gz')
p=pd.read_csv(OUT/'pooled_metrics_by_seed.csv')
m=pd.read_csv(OUT/'macro_weekly_metrics.csv')
assert len(w)==800 and len(p)==200 and len(m)==40
assert not w.duplicated(['method','seed','week','audit_fraction']).any()
assert (w.true_positives+w.selected_without_injection).eq(w.selected_rows).all()
assert np.allclose(w.precision_at_k,w.true_positives/w.selected_rows)
assert np.allclose(w.recall_at_k,w.true_positives/w.positive_rows)
assert np.allclose(w.anomalies_per_100_checks,w.precision_at_k*100)
totals=s.groupby(['method','seed','week','audit_fraction','dimension'])[['population_rows','positive_rows','selected_rows','true_positives','selected_without_injection']].sum().reset_index()
comp=totals.merge(w,on=['method','seed','week','audit_fraction'],suffixes=('_segments','_global'),validate='many_to_one')
for col in ['population_rows','positive_rows','selected_rows','true_positives','selected_without_injection']:
    assert comp[f'{col}_segments'].eq(comp[f'{col}_global']).all(),col
old=pd.read_csv(OUT.parent/'stage7_leakage'/'information_impact.csv')
joined=m.merge(old,left_on=['method','audit_fraction'],right_on=['method','fraction'],validate='one_to_one')
assert np.allclose(joined.precision_at_k,joined.causal_precision)
assert np.allclose(joined.recall_at_k,joined.causal_recall)
assert np.allclose(joined.average_precision,joined.causal_average_precision)
# Pool segment counts first, then recompute ratios. Global weekly selections are preserved.
group=['method','seed','audit_fraction','dimension','segment']
counts=['population_rows','positive_rows','global_selected_rows','selected_rows','true_positives','selected_without_injection','expected_random_tp','realized_same_seed_random_tp']
sp=s.groupby(group)[counts].sum().reset_index()
sp['precision_within_selected_segment']=sp.true_positives/sp.selected_rows.replace(0,np.nan)
sp['recall_within_segment']=sp.true_positives/sp.positive_rows.replace(0,np.nan)
sp['precision_contribution_to_global']=sp.true_positives/sp.global_selected_rows
sp['lift_vs_global_random_allocation']=sp.true_positives/sp.expected_random_tp.replace(0,np.nan)
sp['anomalies_per_100_segment_checks']=100*sp.precision_within_selected_segment
sp['aggregation']='pooled_segment_counts_across_weekly_topk'
sp.to_csv(OUT/'segment_pooled_by_seed.csv',index=False)
bound=json.loads((OUT/'segment_boundaries.json').read_text())
view=m.loc[m.audit_fraction.eq(.01),['method','precision_at_k','recall_at_k','lift_vs_expected_random','average_precision','roc_auc']].sort_values('precision_at_k',ascending=False)
display=view.copy();display[['precision_at_k','recall_at_k','average_precision']]*=100
report=['Негізгі сынақ метрикалары','',
'Дерек жағдайы — файлда берілген жасанды аномалиялар. Негізгі causal белгілер. Сынақ аптасы 32–35. Бағалау жиыны 510 968 абонент–апта. Кездейсоқ және үйренетін әдістерде бес seed қолданылады. Детерминдік әдістердің бірдей баллдары seed ортақ базасымен салыстыру үшін қайта пайдаланылды, олар бес тәуелсіз тәжірибе ретінде түсіндірілмейді.',
'Төмендегі precision, recall және AP пайызбен, lift еселікпен берілген. Апталар тең салмақпен орташаланған.',
display.to_string(index=False),'',
'Негізгі lift анықтамасы precision/сол аптадағы аномалия жиілігі. Бұл бірдей K көлеміндегі кездейсоқ іріктеудің күтілетін precision мәніне қатысты. Нақты сол seed кездейсоқ таңдаумен салыстыру да бөлек бағанда берілген. Оның precision мәні нөл болса, эмпирикалық lift анықталмаған болып сақталады.',
'macro_weekly_metrics.csv — апталар бойынша тең салмақты орташа. pooled_seed_mean.csv — апталық top-K жиындарының біріктірілген TP/K және TP/барлық позитив қатынастарының seed орташа мәні. Біріктірілген популяциядан top-K қайта таңдалған жоқ. Pooled AP/AUC барлық аптаның шикі баллдарын бірге ранжирлейді, олар аптааралық шкала салыстырмалылығына тәуелді. Негізгі AP әр аптада есептеледі.',
'Сегмент шектері тек 26–29 апталардағы 423 735 негізгі оқыту жолынан алынды. Табыс сегменті ағымдағы түсімнен емес, causal prior_mean_4 белгісінен анықталды. Тарих шектері '+str(bound['history'])+', табыс шектері '+str(bound['income'])+'.',
'Шама abs(log(revenue/revenue_original)) арқылы бағаланды. Оның шектері тек оқыту кезеңіндегі енгізілген аномалиялардан есептелді: '+str(bound['magnitude'])+'. Бұл мәндер тек бағалау сегменттеріне қолданылады, модельге берілмейді.',
'Сегменттер үшін жалпы апталық top-K сақталған. Precision within segment таңдауға түскен сол сегменттің жазбаларына қатысты, precision contribution to global жалпы K-ға қатысты. Аномалия түрі мен шамасы бойынша сегменттер меткаға тәуелді, сондықтан олардың ішкі precision мәнін модельдің дербес дәлдігі деп түсіндіруге болмайды. Түрлер бойынша recall мен жалпы precision-ға үлесті пайдалану керек.',
'Сынақ мерзімі ұзарған сайын абоненттердің тарихы табиғи түрде ұзарады. Оқыту шектері 5 және 6 алдыңғы апта болғандықтан, high тобы сынақта басым болуы мүмкін. Нақты сегмент саны segment_population.csv ішінде көрсетілген. Primary жиын төрт аптадан басталады, қысқа тарих негізгі нәтижеге кірмейді.',
'Тең балл кезінде мәтіндік msisdn өсу ретімен іріктеледі. Үлкен балл жоғары күмәнділікті білдіреді. K=ceil(N*q). Әр әдіс үшін бірдей кандидаттар қолданылады.',
'Жаңа 240 синтетикалық сценарийге арналған модель баллдары әлі есептелмеген. Бұл кестелер сол сценарийлердің сапасын сипаттамайды. Олар үшін әдістерді қайта оқыту/бағалау және осы модульмен метрика шығару бөлек орындалуы қажет. Сенімділік интервалдары тоғызыншы кезеңде есептеледі.',
'Тексерулер: 800 апталық метрика жолы, 200 pooled жол, 40 macro жол. Сегменттердің барлық санақтары жалпы санақтарға қайта қосылады. Негізгі нәтижелер жетінші кезеңдегі causal нәтижелермен толық сәйкес.']
(OUT/'metrics_report_kk.txt').write_text('\n\n'.join(report),encoding='utf-8-sig')
checks={'weekly_rows':len(w),'segment_rows':len(s),'pooled_rows':len(p),'macro_rows':len(m),
 'segment_count_reconciliation':True,'stage7_causal_reconciliation':True,
 'new_synthetic_scenario_metrics_available':False,
 'output_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['weekly_metrics.csv','macro_weekly_metrics.csv','pooled_seed_mean.csv','segment_weekly_metrics.csv.gz']}}
(OUT/'completion_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(display.to_string(index=False))
print('All reconciliation checks passed')
