import json
import pathlib
import shutil
import tarfile
import pandas as pd
import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
out = root / 'outputs' / 'stage1_audit'
with tarfile.open(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz', 'r:gz') as tar:
    panel = pd.read_csv(tar.extractfile('2_weekly_revenue_panel.csv'), dtype={'msisdn':'string'}, usecols=['msisdn','week','revenue','is_filled'])
    anomaly = pd.read_csv(tar.extractfile('3_weekly_revenue_anomalies.csv'), dtype={'msisdn':'string'}, usecols=['msisdn','week','revenue_original'])
joined = panel.merge(anomaly, on=['msisdn','week'], how='outer', indicator=True, validate='one_to_one')
cross = {'membership': joined['_merge'].value_counts().to_dict(),
         'panel_only_filled': joined.loc[joined['_merge'].eq('left_only'),'is_filled'].value_counts().to_dict(),
         'original_revenue_mismatch': int((~np.isclose(joined.loc[joined['_merge'].eq('both'),'revenue'], joined.loc[joined['_merge'].eq('both'),'revenue_original'], atol=1e-8, rtol=0)).sum())}
(out/'cross_dataset_check.json').write_text(json.dumps(cross,indent=2),encoding='utf-8')
a=json.loads((out/'dataset_audit.json').read_text(encoding='utf-8'))
env=json.loads((out/'environment.json').read_text(encoding='utf-8'))
lines=['Телеком деректерінің бірінші кезең аудиті','', 'Тексеру 2026 жылғы 22 қарашада жүргізілді. Бастапқы архив өзгертілмеді. Төрт CSV толық оқылды. Негізгі екі апталық файл барлық жол бойынша тексерілді. Parquet бөліктерінің архивтегі атауы мен өлшемі тіркелді, ішкі мазмұны тексерілмеді.','', 'Файлдар']
for name,d in a.items():
    lines.append(f"{name} — {d['rows']:,} жол, {d['column_count']} баған.")
    if 'unique_subscribers' in d:
        lines.append(f"Абоненттер {d['unique_subscribers']:,}. Қайталанатын кілт {d['duplicate_keys']}. Апталар {d['weeks']}. Ішкі үзілістер {d['gap_transitions']:,}, сол үзілістердегі жоқ апталар {d['missing_internal_weeks']:,}.")
        lines.append('Алдыңғы бақылау саны бойынша қамту '+str(d['prior_history_coverage'])+'. Бұл қатардағы бақылаулар саны, күнтізбелік қатарлас апталар саны бөлек тексерілуі тиіс.')
lines += ['', 'Негізгі нәтижелер',
'Аномалия файлының revenue және revenue_original бағандарында бос, нөлдік, теріс және шексіз мәндер жоқ. Панельде 127 392 нөлдік түсім бар. Аномалия файлына нөлдік апталардың кірмеуі уақыттық белгілерді есептеуге әсер етеді.',
f'Екі файлдың тікелей салыстыруы {cross}.',
'Аномалия файлы үшін жай shift(1) алдыңғы қолжетімді жазбаны береді. Ол міндетті түрде алдыңғы күнтізбелік апта болмайды. Келесі кезеңде толық панельді уақыт осі ретінде пайдаланып, толтырылған мәндерді жеке жалаушамен сақтау қажет.',
'Енгізілген аномалия саны 17 184, үлесі 1,0081%. Күшті өсім 5 760, орташа өсім 5 734, төмендеу 5 690. Абсолют төзімділік 1e-8 кезінде нақты түсім өзгерісі мен is_injected арасында сәйкессіздік табылмады.',
'Жыл не толық күн бағаны негізгі файлдарда жоқ. 22–35 апта индекстері бір реттік уақыт шкаласы ретінде ғана қабылданды. Күнтізбелік жыл деректен анықталмады.',
'msisdn мәтін түрінде оқылды. Бірегей абоненттердің жеке идентификаторлары есепте жарияланбайды.',
'Бұрынғы ноутбуктер Google Colab/Drive ортасына бағытталған. Аномалия кодында shift(-1) бар, ол кейінгі қолжетімді жазбаны қолданады. Барлық кезеңнен есептелетін статистика да уақыттық сынақ үшін қайта құрылуы қажет. Ноутбуктер бұл аудитте орындалған жоқ.',
'', 'Есептеу ортасы', str(env['cpu']), str(env['memory']),
'RAM шамамен '+str(round(env['memory']['TotalVisibleMemorySize']/1024**2,2))+' GiB. CPU логикалық ағындары '+str(env['cpu_count'])+'.',
'Python '+env['python'], 'pandas '+env['packages'].get('pandas','unknown')+', numpy '+env['packages'].get('numpy','unknown')+'. pyarrow орнатылмаған. Аудит үшін архивтен CSV оқу жеткілікті болды.',
'Диск бос орны мен барлық кітапхана нұсқалары environment.json файлында берілген.',
'', 'Деректер сөздігі',
'data_dictionary.csv барлық CSV бағандарының типін, бос мән санын және рөлін береді. Бағандардың пәндік мағынасының бастапқы көзі document_text.json. Төменде негізгі апталық бағандардың түсіндірмесі берілген.',
'msisdn — абонент идентификаторы. week — апта индексі. revenue — ағымдағы аптаның түсімі. revenue_original — жасанды өзгеріс енгізілгенге дейінгі түсім.',
'is_injected — енгізілген аномалия белгісі. injection_type — оның түрі. outlier_global және spike_local — дайын базалық ережелердің жалаушалары. Бұл бағандар негізгі модельдің кірісіне қосылмайды.',
'is_filled — бастапқыда жоқ абонент–апта жазбасының толтырылғанын көрсететін жалауша. Оның бастапқы операциялық мағынасы деректерді дайындаған тараппен расталуы қажет.',
'lag_1–lag_4 — алдыңғы апталардағы түсім. roll_mean_4 және roll_std_4 — алдыңғы терезенің орташа мәні мен стандартты ауытқуы. cum_mean_prev — бұрынғы апталар бойынша орташа түсім. weeks_since_start — басталу аптасынан өткен уақыт.',
'target_next_week — келесі аптаның мақсатты түсімі. target_next_is_filled — сол мақсатты аптаның толтырылғанын көрсетеді. Олардың нақты формуласының дұрыстығы бұл кезеңде қайта есептелмеді.',
'', 'Қайта орындау', 'audit_stage1.py және finish_audit.py файлдарын көрсетілген bundled Python арқылы ретімен орындау керек. Негізгі аудит бастапқы архивті оқиды және нәтижелерді осы директорияға жазады.']
(out/'audit_report_kk.txt').write_text('\n\n'.join(lines),encoding='utf-8-sig')
for name in ['audit_stage1.py','finish_audit.py']:
    shutil.copy2(root/'work'/name,out/name)
print(json.dumps(cross))
print('Report written')
