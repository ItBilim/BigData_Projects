"""Run selected stages, or package the existing experiment by default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

out=Path(__file__).resolve().parent
root=out.parent
cfg=json.loads((out/'experiment.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('--rerun', action='store_true', help='Recompute the registered stages before exporting')
parser.add_argument('--from-stage', type=int, default=2)
args=parser.parse_args()
if not Path(cfg['archive']).is_file():
    parser.error('The registered input archive is missing')
if args.rerun and Path(cfg['archive']).resolve()!=Path(r'D:\BigData\BigData_Projects\cdr_datasets.tar.gz').resolve():
    parser.error('Historical stage scripts expect the original Windows archive location')
scripts=[f'stage12_delivery/{name}' for name in ['export.py','figures.py','summarize.py']]
if args.rerun:
    scripts=[s for s in cfg['pipeline'] if int(s.split('_',1)[0][5:])>=args.from_stage]+scripts
with (out/'delivery_run.log').open('a',encoding='utf-8') as log:
    for script in scripts:
        stamp=datetime.now(timezone.utc).isoformat()
        print(stamp,script,flush=True)
        log.write(f'{stamp} START {script}\n')
        log.flush()
        with subprocess.Popen([sys.executable,'-B',str(root/script)],stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                              env={**os.environ,'PYTHONIOENCODING':'utf-8',
                                   'TELECOM_FRESH_BOOTSTRAP':'1' if args.rerun else '0'}) as child:
            for line in child.stdout:
                print(line,end='',flush=True)
                log.write(line)
            result=child.wait()
        log.write(f'{script} exit={result}\n')
        log.flush()
        if result: raise SystemExit(result)
