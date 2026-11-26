"""Frozen TabICLv2 and Mitra-v2 inference on the same leave-one-date-out folds."""
import argparse
import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut

from .data import CLASSES, FEATURES, SOURCE_SHA256, load_data

SEED = 20261126
TAB_FILE = "tabicl-classifier-v2-20260212.ckpt"
TAB_REV = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
MITRA_REV = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
MITRA_REPO = f"https://huggingface.co/autogluon/mitra-classifier-2/resolve/{MITRA_REV}/"
# Pin the upstream artifact bytes, including the native Mitra inference modules.
ASSETS = {
    TAB_FILE: (
        f"https://huggingface.co/jingang/TabICL/resolve/{TAB_REV}/{TAB_FILE}",
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    ),
    "weights/config.json": (
        MITRA_REPO + "config.json",
        "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df",
    ),
    "weights/model.safetensors": (
        MITRA_REPO + "model.safetensors",
        "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09",
    ),
    "autogluon.whl": (
        "https://files.pythonhosted.org/packages/e2/86/007552ef0077e8fe45a38112fd0c7b67030f5a3576e9d270147169c1e9a8/autogluon_tabular-1.6.3-py3-none-any.whl",
        "6511e7e3edf9a93bf7b214f2f6782c1e00f3755c2150f4863f031fd0b3f44895",
    ),
}
MODELS = ("TabICLv2", "Mitra-v2")
ABLATIONS = {
    "duration": ("Duration",),
    "fundamental": ("F01",),
    "peak": ("Fpeak",),
    "quartiles": ("Q25", "Q50", "Q75"),
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset(cache, name, endpoint, offline):
    target = cache / name
    url, expected = ASSETS[name]
    if target.is_file() and sha256(target) == expected:
        return target
    if offline:
        raise FileNotFoundError(f"Missing or invalid cached asset: {name}. Expected SHA-256 {expected}")
    url = url.replace("https://huggingface.co/", endpoint + "/", 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as staged:
        temp = Path(staged.name)
    try:
        with urlopen(url, timeout=45) as remote, temp.open("wb") as out:
            while chunk := remote.read(1024 * 1024):
                out.write(chunk)
        if sha256(temp) != expected:
            raise ValueError(f"Downloaded bytes failed the SHA-256 check for {name}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def load_mitra(cache, endpoint, offline):
    """Load only the four upstream inference modules needed for the frozen checkpoint."""
    for name in ("weights/config.json", "weights/model.safetensors", "autogluon.whl"):
        asset(cache, name, endpoint, offline)
    # Load wheel code inside its original module namespace without installing all AutoGluon.
    core = cache / "native_core"
    prefix = "autogluon/tabular/models/mitra/_internal/"
    with zipfile.ZipFile(cache / "autogluon.whl") as wheel:
        for relative in ("config/enums.py", "models/base.py", "models/embedding.py", "models/tab2d.py"):
            name = prefix + relative
            path = core / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wheel.read(name))
    sys.path.insert(0, str(core.resolve()))
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    return Tab2D.from_pretrained(str(cache / "weights"), device="cpu").eval()


def predict_mitra(pretrained, train_x, train_y, test_x):
    import torch

    support = torch.as_tensor(train_x, dtype=torch.float32)[None]
    queries = torch.as_tensor(test_x, dtype=torch.float32)[None]
    labels = torch.as_tensor(train_y, dtype=torch.long)[None]
    masks = [torch.zeros(1, n, dtype=torch.bool) for n in (train_x.shape[1], len(train_x), len(test_x))]
    with torch.inference_mode():
        logits = pretrained(support, labels, queries, *masks)[0, :, :len(CLASSES)]
        return logits.softmax(dim=-1).cpu().numpy()


def predict_tabicl(checkpoint, train_x, train_y, test_x):
    from tabicl import TabICLClassifier

    model = TabICLClassifier(
        n_estimators=4, random_state=SEED, device="cpu", n_jobs=4,
        model_path=checkpoint, allow_auto_download=False, use_amp=False, use_fa3=False,
    )
    model.fit(train_x, train_y)
    np.testing.assert_array_equal(model.classes_, np.arange(len(CLASSES)))
    return model.predict_proba(test_x)


def held_date_predictions(data, features, model, pretrained):
    x = data[list(features)].to_numpy()
    y = pd.Categorical(data.CallType, categories=CLASSES).codes
    dates = pd.to_datetime(data.Date, format="%d/%m/%Y")
    p = np.full((len(data), len(CLASSES)), np.nan)
    for train, test in LeaveOneGroupOut().split(x, y, groups=dates):
        if len(np.unique(y[train])) != len(CLASSES):
            raise ValueError("All three classes must be represented in the training context")
        predict = predict_tabicl if model == "TabICLv2" else predict_mitra
        p[test] = predict(pretrained, x[train], y[train], x[test])
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Invalid out-of-fold probabilities")
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
    frame = data[["row_id", "Date", "CallType"]].rename(columns={"CallType": "actual"}).copy()
    frame["model"] = model
    for j, label in enumerate(CLASSES):
        frame["p_" + label] = p[:, j]
    frame["prediction"] = np.asarray(CLASSES)[p.argmax(axis=1)]
    return frame.sort_values("row_id")


def scores(frame):
    actual, predicted = frame["actual"], frame["prediction"]
    prob = frame[["p_" + name for name in CLASSES]].to_numpy()
    # sklearn interprets matrix columns in sorted label order.
    labels = pd.Categorical(actual, categories=CLASSES).codes
    return {
        "macro_f1": f1_score(actual, predicted, labels=CLASSES, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "accuracy": accuracy_score(actual, predicted),
        "log_loss": log_loss(labels, prob / prob.sum(axis=1, keepdims=True), labels=np.arange(len(CLASSES))),
    }


def run(data, out, cache, models=MODELS, endpoint="https://huggingface.co", offline=False, ablate=False):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    for model in models:
        print(f"{model}: verifying checkpoint", flush=True)
        pretrained = (asset(cache, TAB_FILE, endpoint, offline)
                      if model == "TabICLv2" else load_mitra(cache, endpoint, offline))
        frame = held_date_predictions(data, FEATURES, model, pretrained)
        frame.to_csv(out / f"oof_{model}.csv", index=False)
        model_files = [out / f"oof_{name}.csv" for name in MODELS]
        complete = pd.concat([pd.read_csv(path) for path in model_files if path.is_file()], ignore_index=True)
        complete.to_csv(out / "oof.csv", index=False)
        metrics = [{"model": name, **scores(group)} for name, group in complete.groupby("model")]
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        versions = {pkg: version(pkg) for pkg in ("numpy", "pandas", "torch", "scikit-learn", "tabicl", "einops", "safetensors", "huggingface-hub")}
        if model == "Mitra-v2":
            versions["einx"] = version("einx")
        names = [TAB_FILE] if model == "TabICLv2" else [k for k in ASSETS if k != TAB_FILE]
        manifest = {
            "model": model, "source_sha256": SOURCE_SHA256, "features": list(FEATURES),
            "classes": list(CLASSES), "n_records": len(frame), "n_dates": data.Date.nunique(),
            "python": platform.python_version(), "packages": versions, "random_seed": SEED,
            "config": "4-member frozen ensemble" if model == "TabICLv2" else "single frozen forward pass",
            "device": "cpu", "torch_threads": 4, "requested_hf_endpoint": endpoint,
            "assets": {k: {"upstream_url": ASSETS[k][0], "sha256": ASSETS[k][1]} for k in names},
        }
        (out / f"run_{model}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{model}: {scores(frame)}", flush=True)
        if ablate:
            rows = [{"removed": "none", **scores(frame)}]
            for group, excluded in ABLATIONS.items():
                features = [f for f in FEATURES if f not in excluded]
                ablated = held_date_predictions(data, features, model, pretrained)
                ablated.to_csv(out / f"ablation_oof_{model}_{group}.csv", index=False)
                rows.append({"removed": group, **scores(ablated)})
                print(f"{model}: ablation {group} complete", flush=True)
            pd.DataFrame(rows).to_csv(out / f"ablations_{model}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/q1/foundation"))
    parser.add_argument("--cache", type=Path, required=True, help="External directory for pinned checkpoints")
    parser.add_argument("--model", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--hf-endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"), default="https://huggingface.co")
    parser.add_argument("--offline", action="store_true", help="Use only verified files already in the cache")
    parser.add_argument("--ablate", action="store_true", help="Also rerun the four feature group omissions")
    args = parser.parse_args()
    run(load_data(args.data), args.out, args.cache, args.model, args.hf_endpoint, args.offline, args.ablate)


if __name__ == "__main__":
    main()
