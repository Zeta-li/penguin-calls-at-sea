"""Verified upstream checkpoints, and frozen CPU predictors for the two tasks."""
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from tabicl import TabICLClassifier

SEED = 20261126
TAB_FILE = "tabicl-classifier-v2-20260212.ckpt"
TAB_REV = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
MITRA_REV = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
WHEEL = "https://files.pythonhosted.org/packages/e2/86/007552ef0077e8fe45a38112fd0c7b67030f5a3576e9d270147169c1e9a8/autogluon_tabular-1.6.3-py3-none-any.whl"
ASSETS = {
    TAB_FILE: (f"https://huggingface.co/jingang/TabICL/resolve/{TAB_REV}/{TAB_FILE}",
               "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"),
    "weights/model.safetensors": (f"https://huggingface.co/autogluon/mitra-classifier-2/resolve/{MITRA_REV}/model.safetensors",
                                  "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09"),
    "weights/config.json": (f"https://huggingface.co/autogluon/mitra-classifier-2/resolve/{MITRA_REV}/config.json",
                            "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"),
    "autogluon.whl": (WHEEL, "6511e7e3edf9a93bf7b214f2f6782c1e00f3755c2150f4863f031fd0b3f44895"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_asset(cache, name):
    target = cache / name
    url, expected = ASSETS[name]
    if target.is_file() and digest(target) == expected:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        staged = Path(temp.name)
        try:
            with urlopen(url, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)
            temp.flush()
            if digest(staged) != expected:
                raise ValueError(f"Checksum mismatch for {name}")
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)
    return target


def setup(cache):
    for name in ASSETS:
        ensure_asset(cache, name)
    prefix = "autogluon/tabular/models/mitra/_internal/"
    # Import only the model's native torch modules from the verified wheel.
    with zipfile.ZipFile(cache / "autogluon.whl") as wheel:
        for name in ("config/enums.py", "models/base.py", "models/embedding.py", "models/tab2d.py"):
            destination = cache / "core" / (prefix + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(wheel.read(prefix + name))
    sys.path.insert(0, str(cache / "core"))
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    return Tab2D.from_pretrained(str(cache / "weights"), device="cpu").eval()


def tabicl(cache, x, y, query):
    model = TabICLClassifier(n_estimators=4, random_state=SEED, device="cpu", n_jobs=4,
                             model_path=cache / TAB_FILE, allow_auto_download=False,
                             use_amp=False, use_fa3=False)
    model.fit(x, y)
    np.testing.assert_array_equal(model.classes_, np.arange(len(np.unique(y))))
    return model.predict_proba(query)


def mitra(model, x, y, query):
    support = torch.as_tensor(x, dtype=torch.float32)[None]
    test = torch.as_tensor(query, dtype=torch.float32)[None]
    labels = torch.as_tensor(y, dtype=torch.long)[None]
    masks = [torch.zeros(1, n, dtype=torch.bool) for n in (x.shape[1], len(x), len(query))]
    with torch.inference_mode():
        scores = model(support, labels, test, *masks)[0, :, :len(np.unique(y))]
        return torch.softmax(scores, dim=-1).numpy()
