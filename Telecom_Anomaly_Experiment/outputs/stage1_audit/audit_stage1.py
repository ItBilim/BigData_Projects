import collections
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import sys
import tarfile
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

SOURCE = pathlib.Path(r'D:\BigData\BigData_Projects')
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'stage1_audit'
OUT.mkdir(parents=True, exist_ok=True)

def save(name, data):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, np.generic): return value.item()
        return value
    (OUT / name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, default=str), encoding='utf-8')

env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
       'cpu_count': os.cpu_count(), 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
       'disk': {str(p): shutil.disk_usage(p)._asdict() for p in [SOURCE, ROOT]}}
for name in ['memory', 'cpu']:
    env[name] = json.loads((ROOT / 'work' / f'hardware_{name}.json').read_text(encoding='utf-8-sig'))
save('environment.json', env)
context = {}
for path in SOURCE.glob('*.docx'):
    with zipfile.ZipFile(path) as z:
        xml = ET.fromstring(z.read('word/document.xml'))
        context[path.name] = [''.join(p.itertext()) for p in []]
        context[path.name] = [''.join(n.text or '' for n in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')) for p in xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
save('document_text.json', context)
notebooks = {}
for path in SOURCE.glob('*.ipynb'):
    nb = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    notebooks[path.name] = {'cells': len(nb['cells']), 'code': code,
        'saved_errors': [o.get('ename') for c in nb['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error']}
save('notebook_context.json', notebooks)
archive = SOURCE / 'cdr_datasets.tar.gz'
inventory = []
audits = {}
dictionary = []
forbidden = {'revenue_original', 'is_injected', 'injection_type', 'outlier_global', 'spike_local'}
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        inventory.append({'name': member.name, 'bytes': member.size, 'file': member.isfile()})
        if not member.isfile() or not member.name.endswith('.csv'):
            continue
        print('Reading', member.name, flush=True)
        chunks = []
        missing = collections.Counter()
        rows = 0
        types = {}
        for chunk in pd.read_csv(tar.extractfile(member), dtype={'msisdn': 'string'}, chunksize=150000):
            rows += len(chunk)
            missing.update(chunk.isna().sum().to_dict())
            types.update({c: str(t) for c, t in chunk.dtypes.items()})
            if member.name.startswith(('2_', '3_')):
                chunks.append(chunk)
        result = {'rows': rows, 'columns': list(types), 'column_count': len(types), 'inferred_types': types, 'missing': dict(missing)}
        for column in types:
            role = 'identifier' if column == 'msisdn' else 'evaluation_only' if column in forbidden or column.startswith('target_') else 'time' if column == 'week' else 'requires_time_availability_check'
            dictionary.append({'file': member.name, 'column': column, 'inferred_type': types[column], 'missing': missing[column], 'role': role})
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            result['unique_subscribers'] = int(df.msisdn.nunique())
            result['missing_ids'] = int(df.msisdn.isna().sum())
            result['duplicate_keys'] = int(df.duplicated(['msisdn', 'week']).sum())
            result['weeks'] = sorted(df.week.dropna().unique().tolist())
            result['weekly_rows'] = df.groupby('week').size().to_dict()
            result['year_columns'] = [c for c in df if 'year' in c.lower() or 'date' in c.lower()]
            result['leading_zero_ids'] = int(df.msisdn.str.startswith('0').sum())
            result['id_lengths'] = df.msisdn.str.len().value_counts().to_dict()
            result['revenue_stats'] = {}
            for c in ['revenue', 'revenue_original']:
                if c not in df: continue
                v = df[c]
                result['revenue_stats'][c] = {'zero': int(v.eq(0).sum()), 'negative': int(v.lt(0).sum()), 'infinite': int(np.isinf(v).sum()), 'missing': int(v.isna().sum()), 'describe': v.describe(percentiles=[.01,.25,.5,.75,.99]).to_dict()}
            unique = df[['msisdn', 'week']].drop_duplicates().sort_values(['msisdn', 'week'])
            lengths = unique.groupby('msisdn').size()
            result['history_length_distribution'] = lengths.value_counts().sort_index().to_dict()
            unique['previous_week'] = unique.groupby('msisdn').week.shift()
            gaps = unique.week - unique.previous_week
            result['gap_transitions'] = int(gaps.gt(1).sum())
            result['missing_internal_weeks'] = int((gaps[gaps.gt(1)] - 1).sum())
            result['subscribers_with_internal_gaps'] = int(unique.loc[gaps.gt(1), 'msisdn'].nunique())
            unique['prior_observation_count'] = unique.groupby('msisdn').cumcount()
            result['prior_history_coverage'] = {str(n): int(unique.prior_observation_count.ge(n).sum()) for n in [1,2,4,8]}
            if 'is_injected' in df:
                flags = df.is_injected.eq(1)
                changed = ~np.isclose(df.revenue, df.revenue_original, rtol=0, atol=1e-8, equal_nan=True)
                result['injection_labels'] = df.is_injected.value_counts(dropna=False).to_dict()
                result['injection_types'] = df.injection_type.fillna('<missing>').value_counts().to_dict()
                result['injection_rate'] = float(flags.mean())
                result['label_change_mismatch'] = int(np.count_nonzero(changed != flags.to_numpy()))
                result['flagged_unchanged'] = int(np.count_nonzero(flags.to_numpy() & ~changed))
                result['unflagged_changed'] = int(np.count_nonzero(~flags.to_numpy() & changed))
                result['injected_by_week'] = df.groupby('week').is_injected.agg(['sum','mean']).to_dict()
            del df, unique
        audits[member.name] = result
        save('dataset_audit.json', audits)
        print('Completed', member.name, rows, flush=True)
save('archive_inventory.json', inventory)
pd.DataFrame(dictionary).to_csv(OUT / 'data_dictionary.csv', index=False, encoding='utf-8-sig')
save('source_inventory.json', [{'name': str(p.relative_to(SOURCE)), 'bytes': p.stat().st_size} for p in SOURCE.rglob('*') if p.is_file()])
save('provenance.json', {'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_mtime_ns': archive.stat().st_mtime_ns,
    'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), 'completed_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    'parquet_content_audited': False, 'note': 'Archive inventory includes Parquet. Full row audit targets CSV files. Week year is not inferred.'})
print('DONE', OUT, flush=True)
