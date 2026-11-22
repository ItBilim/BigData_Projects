from pathlib import Path
import hashlib, importlib.metadata as metadata, json, sys
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
work=ROOT.parent/'work'
versions=json.loads((ROOT/'stage5_methods'/'preregistered_run.json').read_text())['packages']
for path in [work/'ml_dependencies',work/'plot_dependencies']:
    for package in metadata.distributions(path=[str(path)]):
        name=package.metadata['Name']
        if name: versions.setdefault(name.lower(),package.version)
(OUT/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in sorted(versions.items())),encoding='utf-8')
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.suffix in {'.json','.txt','.log','.py'}
             and p.name not in {'run_manifest.json','delivery_run.log'})
(OUT/'run_manifest.json').write_text(json.dumps({
    'python':sys.version,'original_console_logs':'Historical stages retain their lock files, timings and completion checks. Full console capture starts with this delivery runner.',
    'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},
    ensure_ascii=False,indent=2),encoding='utf-8')
s=pd.read_csv(ROOT/'stage8_metrics'/'segment_macro_metrics.csv').query("method=='lightgbm_absolute' and audit_fraction==0.01")
main=pd.read_csv(OUT/'article_main_table.csv').set_index('method')
def segment_line(dimension):
    rows=s.loc[s.dimension.eq(dimension)&s.positive_rows.gt(0)]
    return ', '.join(f'{r.segment} {100*r.recall_within_segment:.2f}%' for r in rows.itertuples())
pair=pd.read_csv(ROOT/'stage9_stability'/'primary_comparison.csv').iloc[0]
assert len(list((OUT/'figures').glob('*.svg')))==7
assert len(list((OUT/'figures').glob('*.png')))==7
scores=pd.read_csv(OUT/'score_manifest.csv')
for row in scores.itertuples():
    with np.load(OUT/'scores'/row.file) as data:
        assert data['score'].shape==(row.series,row.rows_per_series)
        assert len(data['subject'])==row.rows_per_series and len(data['series'])==row.series
        assert data['label'].shape in [(row.rows_per_series,),(row.series,row.rows_per_series)]
report=f"""# Апталық телеком түсіміндегі аномалиялар

32–35 апталардағы негізгі жиында 510 968 абонент–апта және 5 222 енгізілген аномалия бар. Барлық он әдіс нұсқасы бірдей апталық кілттерде бағаланды.

**1. Бірдей тексеру көлемінде қай әдіс көбірек аномалия тапты?** 1% бюджетте LightGBM абсолют қалдығының апталық орташа precision мәні {100*main.loc['lightgbm_absolute','precision_at_k']:.2f}%, recall {100*main.loc['lightgbm_absolute','recall_at_k']:.2f}%. Жалпы статистикалық баллдың precision мәні {100*main.loc['global','precision_at_k']:.2f}%. Абонент бойынша жұпталған айырма {100*pair.difference:.2f} пайыздық тармақ болды, 95% интервал {100*pair.ci_low:.2f}–{100*pair.ci_high:.2f}. Holm түзетілген p={pair.holm_adjusted_p:.4f}. Салыстыру нүктелік бағалар белгілі болғаннан кейін бекітілген.

**2. Қандай аномалиялар жиірек табылды?** LightGBM абсолют қалдығының жалпы апталық top‑1% ішіндегі recall көрсеткіштері төменде берілген.

- Аномалия түрі бойынша {segment_line('injection_type')}.
- Ауытқу шамасы бойынша {segment_line('magnitude_segment')}.
- Тарих ұзақтығы бойынша {segment_line('history_segment')}.
- Тарихи түсім бойынша {segment_line('income_segment')}.

Шектер 26–29 апталардағы оқыту деректерінен есептелген. Тарих сегменттерінің көлемі бірдей емес және тарихы төрт аптадан қысқа жазбаларға бұл көрсеткіштер қолданылмайды.

**3. Кейінгі ақпарат нәтижені қанша өзгертті?** Төрт тарихи көршінің бір бөлігін кейінгі апталармен алмастыру LightGBM абсолют қалдығының precision көрсеткішін +0,19 пайыздық тармаққа өзгертті. Ең үлкен өсім қарапайым нормаланған қалдықта +0,93 тармақ болды, Isolation Forest −0,15 тармаққа өзгерді. 35-аптада кейінгі сынақ көршісі болмағандықтан, сол аптадағы үйренетін модельдер айырмасына оқыту белгілеріндегі кейінгі ақпарат ықпал етті.

Зерттеудің эмпирикалық үлесі — абонент–апта деңгейінде тең тексеру бюджетімен салыстыру, кейінгі ақпаратқа сезімталдықты өлшеу және апта аралық абонент тәуелділігін есепке алу. Төрт сынақ аптасы маусымдық тұрақтылық туралы қорытындыға жеткіліксіз.

Мақала кестелерінің негізгі нәтижелері берілген жасанды инъекцияларға қатысты. Оператор растаған тәуелсіз белгілер қолданылған жоқ. 48 жаңа конфигурацияның ішінен екі есе бір апталық өсім барлық әдіспен бес инъекция seed-інде бағаланды. Екі апталық сценарий үш ережелік әдіспен бір seed-те тексерілді. Қалған сценарийлер мен оқыту ластануының модель сапасына әсері үшін журналдар бар, метрикалар есептелмеген.
"""
(OUT/'analysis_kk.md').write_text(report,encoding='utf-8')
print('Verified seven figures and eight score exports')
