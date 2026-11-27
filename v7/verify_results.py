"""Check every seventh-edition CSV against the independent release reference."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    expected = json.loads((HERE / "reference" / "numeric_outputs.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(ROOT).as_posix(): p for p in (HERE / "results").glob("*.csv")}
    differences = sorted(set(expected) ^ set(actual))
    for name in sorted(expected.keys() & actual.keys()):
        with actual[name].open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected[name]:
                differences.append(name)
    if differences:
        raise SystemExit("缺失、多出或内容不符的结果表\n" + "\n".join(differences))
    print(f"已核验 {len(expected)} 张第七版数值结果表，SHA-256 全部一致。")


if __name__ == "__main__":
    main()
