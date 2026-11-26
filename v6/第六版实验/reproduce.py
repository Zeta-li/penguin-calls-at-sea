"""Recompute the sixth edition against the fifth-edition archived observations and fits."""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digests(paths):
    values={}
    for path in sorted(set(paths)):
        with path.open('rb') as stream:
            values[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(stream,'sha256').hexdigest()
    return values


def main():
    env = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1', SOURCE_DATE_EPOCH='1795651200', PYTHONDONTWRITEBYTECODE='1')
    for folder in (HERE/'results',HERE/'figures'):
        folder.mkdir(exist_ok=True)
    for script in ('acoustic_propagation.py','behavior.py','cue_transfer.py','ecology.py','primary_intervals.py','illustrate.py'):
        print(f'Running {script}',flush=True)
        subprocess.run([sys.executable,str(HERE/script)],cwd=ROOT,env=env,check=True)
    v5 = json.loads((ROOT/'第五版实验/results/manifest.json').read_text(encoding='utf-8'))
    sources=[*HERE.glob('*.py'),*(ROOT/f for f in v5['code'])]
    inputs=[*(ROOT/'date').glob('Data*.csv'),*(ROOT/'第五版实验/inputs').glob('*'),*(ROOT/'第五版实验/results').glob('q*.csv'),*(ROOT/'第四版实验/results').glob('*.csv'),*(ROOT/'第四版实验/results').glob('*.npz'),*(ROOT/'第三版实验/results').glob('*.csv'),*(ROOT/'第二版实验/results').glob('*oof.csv'),*(ROOT/'results').glob('*.csv')]
    outputs=list((HERE/'results').glob('*.csv'))+list((HERE/'results').glob('*.npz'))
    data={'command':'python 第六版实验/reproduce.py','python':sys.version.split()[0], 'versions':{k:version(k) for k in ('numpy','scipy','scikit-learn','pandas','matplotlib','Pillow')},'input_and_reference_files':digests(inputs),'code':digests(sources),'numeric_outputs':digests(outputs)}
    report=HERE/'results/manifest.json'; report.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(outputs)} numerical artifact hashes to {report.relative_to(ROOT)}.')

if __name__=='__main__':main()
