"""Rebuild edition five, optionally verifying the historical fit and both foundation forwards."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "第四版实验"
INPUTS = HERE / "inputs"
RESULTS = HERE / "results"
STEPS = ("foundation.py", "temporal.py", "classical.py", "downstream.py", "compare.py")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checksums(paths):
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def run(script, *options, cwd=ROOT):
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH="1795651200")
    print(f"Running {script.relative_to(cwd)}", flush=True)
    subprocess.run([sys.executable, str(script), *map(str, options)], cwd=cwd, env=env, check=True)


def compare_v4_outputs(fresh, reference):
    # The original numerical manifest covers 34 CSV tables and one NPZ bootstrap archive.
    if fresh.suffix == ".csv":
        pd.testing.assert_frame_equal(pd.read_csv(fresh), pd.read_csv(reference), check_exact=False,
                                      rtol=1e-6, atol=1e-8)
    elif fresh.suffix == ".npz":
        with np.load(fresh, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f"Array keys differ in {reference.name}")
            for key in a.files:
                if a[key].dtype.kind in "fc":
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-8, err_msg=key)
                else:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    else:
        raise ValueError(f"Unexpected historical result format: {reference.name}")


def rebuild_v4(directory):
    archive = json.loads((V4 / "results/manifest.json").read_text())
    for name, expected in {**archive["inputs"], **archive["code"]}.items():
        source = ROOT / name
        if digest(source) != expected:
            raise ValueError(f"Historical dependency checksum mismatch: {name}")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Start with empty output directories and never call the in-place V4 entry point.
    local = directory / V4.name
    (local / "results").mkdir()
    (local / "figures").mkdir()
    run(local / "reproduce.py", cwd=directory)
    expected_names = set(archive["outputs"])
    computed = {p.relative_to(directory).as_posix() for p in (local / "results").iterdir() if p.suffix in {".csv", ".npz"}}
    if computed != expected_names:
        raise ValueError(f"V4 output inventory differs: {expected_names ^ computed}")
    for name, expected in archive["outputs"].items():
        reference = ROOT / name
        if digest(reference) != expected:
            raise ValueError(f"Historical reference checksum mismatch: {name}")
        compare_v4_outputs(directory / name, reference)
    print(f"Rebuilt and verified {len(computed)} V4 numerical artifacts in isolation.", flush=True)


def rebuild_foundation(cache, directory):
    expected = json.loads((INPUTS / "q1f_inputs.json").read_text())
    weight = cache / expected["weight_file"]
    if not weight.is_file() or digest(weight) != expected["weight_sha256"]:
        raise ValueError(f"Provide the pinned model checkpoint in the cache: {weight.name}")
    for script, prefix, checksum_field in (("foundation.py", "q1f", "probability_sha256"),
                                            ("temporal.py", "q1t", "predictions_sha256")):
        target = directory / prefix
        run(HERE / script, "--forward", "--cache", cache, "--forward-output", target, "--output", target / "scores")
        # Each script checks row/date identities and float probabilities against the frozen archive.
        # A different CPU library can change low-order bits while passing the documented tolerance.
        name = f"{prefix}_inputs.json"
        actual = json.loads((target / name).read_text())
        saved = json.loads((INPUTS / name).read_text())
        prediction = f"{prefix}_held_probabilities.csv"
        if (digest(target / prediction) != actual[checksum_field] or
                digest(INPUTS / prediction) != saved[checksum_field]):
            raise ValueError(f"Invalid generated or archived forward signature: {prefix}")
        for key in saved.keys() - {"versions", checksum_field}:
            if actual[key] != saved[key]:
                raise ValueError(f"Forward signature mismatch: {prefix}.{key}")
    print("Verified date-held and year-held predictions from the pinned weights.", flush=True)


def write_manifest():
    previous = json.loads((V4 / "results/manifest.json").read_text())
    sources = [*(ROOT / "date").glob("Data*.csv"), *INPUTS.glob("*.json"), *INPUTS.glob("*.csv"),
               ROOT / "第二版实验/results/q1_oof.csv", *(V4 / "results" / name for name in (
                   "q1_predictions.csv", "q2_held_distances.csv", "q3v4_timing_oof.csv",
                   "q4_acoustic_shape_sensitivity.csv")), V4 / "results/manifest.json"]
    scripts = [ROOT / s for s in previous["code"]]
    scripts.extend(HERE / name for name in (*STEPS, "reproduce.py", "diagnostics/query_batch.py"))
    outputs = list(RESULTS.glob("q*.csv"))
    record = {"command": "python 第五版实验/reproduce.py",
              "primary_metric": "pooled leave-recording-date-out balanced accuracy",
              "bootstrap": {"unit": "recording date", "paired_draws": 10000, "seed": 20261631},
              "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow")},
              "inputs": checksums(sources), "code": checksums(scripts), "outputs": checksums(outputs)}
    path = RESULTS / "manifest.json"
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=RESULTS, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)
    print(f"Recorded {len(outputs)} numerical tables in {path.relative_to(ROOT)}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-v4", action="store_true", help="Refit archived V4 in an isolated temporary directory.")
    parser.add_argument("--foundation-cache", type=Path, help="Additionally recompute both frozen forwards from a pinned checkpoint.")
    args = parser.parse_args()
    if args.rebuild_v4 or args.foundation_cache:
        with tempfile.TemporaryDirectory(prefix="penguin-v5-verify-") as scratch:
            directory = Path(scratch)
            if args.rebuild_v4:
                rebuild_v4(directory / "v4")
            if args.foundation_cache:
                rebuild_foundation(args.foundation_cache.resolve(), directory / "foundation")
    for name in STEPS:
        run(HERE / name)
    write_manifest()


if __name__ == "__main__":
    main()
