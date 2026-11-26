"""Compare the reconstructed numerical tables with the archived sixth edition."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "第六版实验" / "results"
REFERENCE = ROOT / "reference" / "numeric_outputs.json"


def main():
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = {path.relative_to(ROOT).as_posix(): path for path in RESULTS.glob("*.csv")}
    problems = []
    for name, digest in expected.items():
        if name not in actual:
            problems.append(f"缺少结果表  {name}")
            continue
        with actual[name].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                problems.append(f"校验值不同  {name}")
    for name in sorted(actual.keys() - expected.keys()):
        problems.append(f"出现额外结果表  {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"已核验 {len(expected)} 张第六版结果表，SHA-256 全部一致。")


if __name__ == "__main__":
    main()
