"""Run the seventh-edition experiments in dependency order, then fingerprint the run."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ("audio_probe.py", "q12.py", "q3.py", "q4.py", "figures.py")


def digests(paths):
    checks = {}
    for path in sorted(set(paths)):
        with path.open("rb") as stream:
            checks[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return checks


def main():
    env = os.environ.copy()
    env.update({k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                  "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE")})
    env.update(MPLBACKEND="Agg", SOURCE_DATE_EPOCH="1795737600", PYTHONUTF8="1")
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    for name in STAGES:
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / name)], cwd=ROOT, env=env, check=True)
    baseline = ROOT / "v6"
    source_files = [*baseline.rglob("*.py"), *HERE.glob("*.py")]
    inputs = [*baseline.rglob("*.csv"), *baseline.rglob("*.npz"),
              *baseline.rglob("manifest.json"), *HERE.glob("audio/*.wav"), *HERE.glob("data/*/*.csv")]
    manifest = {
        "command": "python v7/reproduce.py", "python": sys.version.split()[0],
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow")},
        "stages": STAGES, "source_code": digests(source_files), "inputs_and_frozen_references": digests(inputs),
        "numeric_outputs": digests(results.glob("*.csv")),
        "figures": digests(p for p in (HERE / "figures").glob("*") if p.suffix in (".svg",".png")),
    }
    (results / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"Generated {len(manifest['numeric_outputs'])} tables and {len(manifest['figures'])} figure files.")


if __name__ == "__main__":
    main()
