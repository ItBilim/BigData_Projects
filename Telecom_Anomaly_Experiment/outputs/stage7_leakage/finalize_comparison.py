from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
r=pd.read_csv(OUT/'paired_results.csv')
assert len(r)==832
assert set(r.week)=={32,33,34,35}
assert not r[['precision','recall','average_precision']].isna().any().any()
assert len(list(OUT.glob('scores_*.npz')))==208
checks=[]
for week in [32,33,34,35]:
    for method in ['random','global','weekly_absolute','weekly_relative']:
        seeds=[42,43,44,45,46] if method=='random' else [42]
        for seed in seeds:
            a=np.load(OUT/f'scores_{method}_causal_w{week}_s{seed}.npz')['scores']
            b=np.load(OUT/f'scores_{method}_hindsight_w{week}_s{seed}.npz')['scores']
            assert np.array_equal(a,b),(week,method,seed)
checks.append('Unchanged controls identical in both arms')
for method in ['personal_mad','simple_absolute','simple_normalized']:
    a=np.load(OUT/f'scores_{method}_causal_w35_s42.npz')['scores']
    b=np.load(OUT/f'scores_{method}_hindsight_w35_s42.npz')['scores']
    assert np.array_equal(a,b),method
checks.append('Final-week deterministic historical scores identical without future neighbors')
selected=OUT.parent/'stage5_methods'/'selected_parameters.json'
p=json.loads(selected.read_text()); assert p['lightgbm_absolute']['params']==p['lightgbm_normalized']['params']
lock=json.loads((OUT/'run_lock.json').read_text())
assert hashlib.sha256(selected.read_bytes()).hexdigest()==lock['selected_parameters_sha256']
checks.append('Selected parameters unchanged')
table=pd.read_csv(OUT/'information_impact_at_1pct.csv').sort_values('causal_precision',ascending=False)
view=table[['method','causal_precision','hindsight_precision','delta_precision_percentage_points','relative_precision_percent']].copy()
view[['causal_precision','hindsight_precision']]*=100
lines=['Кейінгі ақпараттың әсері бойынша бақылау эксперименті','',
'Төрт сынақ аптасы 32–35. Бес seed. Бекітілген параметрлер. Кестедегі негізгі және бақылау precision мәндері пайызбен, абсолют айырма пайыздық тармақпен, салыстырмалы айырма пайызбен берілген.',
'',view.to_string(index=False),'',
'Негізгі causal нәтижелер ғылыми салыстыруға алынады. Hindsight бақылауында тарихи төрт мәннің орнына өткен және кейінгі көрші апталар қолданылады. Ағымдағы түсім терезелік статистикаға кірмейді. Барлық нақты белгілер README_kk.txt ішінде сипатталған.',
'Екі нұсқаның оқыту кілттері, үлгілері, seed-тері, масштаб floor-ы және модель параметрлері бірдей. Өзгеретін нәрсе — терезелік белгілердің ақпарат көзі. Hindsight fit жолдары сынақ аптасының түсімін де қолдануы мүмкін. Бұл тәжірибеде әдейі енгізілген бұрмалау.',
'35-аптада кейінгі көрші жоқ. Сондықтан ол аптада үйренетін модельдегі айырма fit белгілеріндегі бұрмалауды да қамтиды. Осы шектеумен төрт апталық орташа түсіндіріледі.',
'Сенімділік интервалдары есептелмеді. Теріс айырмалар сақталды. Жаңа синтетикалық сценарийлерге бұл кесте автоматты түрде жалпыланбайды.',
'',*checks]
(OUT/'comparison_report_kk.txt').write_text('\n\n'.join(lines),encoding='utf-8-sig')
(OUT/'completion_checks.json').write_text(json.dumps({'metric_rows':len(r),'score_files':208,'checks':checks},indent=2),encoding='utf-8')
print(view.to_string(index=False))
