"""Recompute the fourth edition offline, optionally rebuilding its inherited estimates."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def hashes(paths):
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-base",action="store_true",help="Recompute V1 numeric results and V3 conditional rank comparisons")
    p.add_argument("--foundation-cache",type=Path,help="Rebuild and compare all saved foundation OOF probabilities with the pinned model cache")
    args=p.parse_args()
    env=dict(os.environ,MPLBACKEND="Agg",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    def run(*args):
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,check=True)
    if args.rebuild_base:
        # The table path uses only raw numeric inputs, without requiring team notes or the problem PDF.
        scratch=HERE/"base_recheck"
        run(ROOT/"solve.py","--tables-only","--output-dir",scratch)
        run(ROOT/"第三版实验/acoustic_checks.py","--output",scratch/"rank")
    if args.foundation_cache is not None:
        import pandas as pd
        from numpy.testing import assert_allclose
        target=HERE/"foundation_recheck"
        run(ROOT/"第二版实验/evaluate.py","--cache",args.foundation_cache.resolve(),"--output",target)
        for name in ("q1_oof","q3_oof"):
            a=pd.read_csv(target/(name+".csv"))
            b=pd.read_csv(ROOT/"第二版实验/results"/(name+".csv"))
            assert list(a.columns)==list(b.columns)
            for col in a:
                if pd.api.types.is_numeric_dtype(a[col]): assert_allclose(a[col],b[col],rtol=1e-6,atol=1e-7)
                else: assert a[col].equals(b[col]),col
    for name in ("acoustics.py","propagation_v4.py","behavior.py","ecology.py","illustrate.py"):
        run(HERE/name)
    sources=[*sorted((ROOT/"date").glob("Data*.csv")),*(ROOT/"date").glob("Supplementary5*.wav"),
             *(ROOT/"第二版实验/results").glob("*oof.csv")]
    scripts=[*HERE.glob("*.py"),*ROOT.glob("*.py"),*(ROOT/"第三版实验").glob("*.py"),
             *(ROOT/"第二版实验").glob("*.py"),*(ROOT/"utils").glob("*.py")]
    outputs=[*(HERE/"results").glob("*.csv"),*(HERE/"results").glob("*.npz")]
    record={"command":"python 第四版实验/reproduce.py","python":sys.version.split()[0],
            "versions":{s:version(s) for s in ("numpy","scipy","scikit-learn","pandas","matplotlib","Pillow")},
            "method":"V4 core recomputed; archived V2 OOF used as benchmark",
            "inputs":hashes(sources),"code":hashes(scripts),"outputs":hashes(outputs)}
    (HERE/"results/manifest.json").write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print(f"{len(outputs)} numeric outputs; input and code SHA-256 recorded.")
if __name__=="__main__":
    main()
