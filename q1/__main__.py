"""Run analyses with python -m q1 --data PATH [--stage statistics|classical]."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

from .data import DATA01, FEATURES, SOURCE_SHA256, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to original Data01 CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/q1"))
    parser.add_argument("--stage", choices=("core", "statistics", "classical"), default="core")
    args = parser.parse_args()
    data = load_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("core", "statistics"):
        from . import statistics

        output = args.out / "statistics"
        output.mkdir(exist_ok=True)
        statistics.run(data, output)
    if args.stage in ("core", "classical"):
        from . import classifiers

        output = args.out / "classical"
        output.mkdir(exist_ok=True)
        classifiers.run(data, output)
    manifest = {
        "source": f"https://zenodo.org/records/21376649/files/{DATA01}",
        "sha256": SOURCE_SHA256,
        "recordings": len(data),
        "dates": data.Date.nunique(),
        "features": FEATURES,
        "row_id": "1-based data record, excluding the CSV header",
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib")},
    }
    (args.out / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
