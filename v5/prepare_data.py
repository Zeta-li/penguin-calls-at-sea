"""Prepare the fifth-edition experiment's original inputs using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
RECORD = "https://zenodo.org/api/records/21376649"


def fetch(url):
    request = Request(url, headers={"User-Agent": "penguin-calls-at-sea/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def prepare(source_dir):
    manifest = ROOT / "第四版实验/results/manifest.json"
    inputs = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    expected = {Path(p).name: h for p, h in inputs.items() if Path(p).parent == Path("date")}
    extensions = [Path(name).suffix for name in expected]
    if sorted(extensions) != [".csv"] * 11 + [".wav"] * 3:
        raise ValueError(f"Expected 11 CSV and 3 WAV hashes in {manifest}")
    if source_dir is not None and not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    destination = ROOT / "date"
    destination.mkdir(exist_ok=True)
    links = None
    unchanged = 0
    missing_wavs = []
    with TemporaryDirectory(dir=destination) as tmp:
        staged = []
        for name, checksum in sorted(expected.items()):
            target = destination / name
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
                unchanged += 1
                continue
            if source_dir is None:
                if links is None:
                    files = json.loads(fetch(RECORD))["files"]
                    links = {entry["key"]: entry["links"]["self"] for entry in files}
                if name not in links:
                    raise ValueError(f"Zenodo record 21376649 is missing {name}")
                data = fetch(links[name])
            else:
                source = source_dir / name
                if not source.exists() and name.endswith(".wav") and not target.exists():
                    missing_wavs.append(name)
                    continue
                data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch for {name}. Existing files have been preserved.")
            temp_file = Path(tmp) / name
            temp_file.write_bytes(data)
            staged.append(temp_file)

        # Validate every available input before replacing any existing file.
        for temp_file in staged:
            temp_file.replace(destination / temp_file.name)

    print(f"{destination}: 11 verified CSVs, {3 - len(missing_wavs)} verified WAVs; "
          f"{len(staged)} installed, {unchanged} already matched.")
    if missing_wavs:
        print("The default V5 run is ready. --rebuild-v4 also needs these sample WAV files:")
        print("\n".join(missing_wavs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dir", type=Path, metavar="PATH",
                        help="directory containing the original Data01–Data11 CSVs and any sample WAVs")
    source.add_argument("--download", action="store_true",
                        help="download the 11 CSVs and 3 sample WAVs from Zenodo record 21376649")
    args = parser.parse_args()
    try:
        prepare(args.from_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
