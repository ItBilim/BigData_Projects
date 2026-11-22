import json
from pathlib import Path
import pandas as pd
out=Path(__file__).resolve().parent
r=pd.read_csv(out/'validation_results.csv')
s=pd.read_csv(out/'selected_methods.csv').sort_values('mean_weekly_precision_at_1pct',ascending=False)
assert set(r.week)=={30,31}
assert len(r)==448, f'Unexpected record count {len(r)}'
assert len(list(out.glob('scores_*.npz')))==112
assert not r[['precision','recall','average_precision']].isna().any().any()
fits=r.drop_duplicates(['method','candidate','week','seed'])
# Absolute and normalized LightGBM scores reuse the same trained models.
fits=fits.loc[fits.method.ne('lightgbm_normalized')]
time_by_method=fits.groupby('method').agg(fits=('fit_seconds','size'),fit_seconds=('fit_seconds','sum'),score_seconds=('score_seconds','sum'))
time_by_method.to_csv(out/'runtime_summary.csv')
lines=['Бесінші кезеңнің валидация нәтижелері','',
'Параметрлер 30–31 апталардағы апталық Precision@1% орташа көрсеткішімен таңдалды. Үйренетін әдістерде бес seed нәтижесі орташаланды. Төмендегі көрсеткіштер валидацияға тиесілі.',
'',s.to_string(index=False),'',
'Isolation Forest және LightGBM отбасыларының әрқайсысы 30 fit орындады. Әр fit ең көбі 200 000 өткен primary жолды қолданды. Нақты fit және score уақыттары CSV файлдарында сақталған.',
'LightGBM абсолют және нормаланған қалдық нұсқалары бір оқытылған модельден есептелді. Нұсқалар үшін ең жақсы кандидат жеке таңдалған.',
'Нормаланған қалдық міндетті түрде жақсы нәтиже бермейді. Бірдей 1% бюджетте абсолют қалдық жоғары болған жағдайда осы нәтиже сақталды. Жалпы статистикалық әдістің сапасы да барлық үйренетін әдістен төмен болуы міндетті емес.',
'Шектер mean/min/max кестесінде сенімділік интервалдары ретінде түсіндірілмейді. Сынақ апталарында әдістердің реттілігі өзгеруі мүмкін.',
'',time_by_method.to_string(),'',
'Тексерулер: екі валидация аптасы, 112 score файлы, 448 метрика жолы, барлық негізгі метрикалар ақырлы. Әр score файлы ортақ апталық кілттер ретімен жазылған. 32–35 апталардың score нәтижесі есептелмеді.']
(out/'validation_report_kk.txt').write_text('\n\n'.join(lines),encoding='utf-8-sig')
(out/'completion_checks.json').write_text(json.dumps({'metric_rows':len(r),'score_files':112,'weeks':[30,31],'all_complete':True},indent=2),encoding='utf-8')
print(s.to_string(index=False))

