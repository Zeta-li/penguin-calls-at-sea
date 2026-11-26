"""Validate and identify the versioned, read-only Zenodo Data01 input."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("Duration", "F01", "Fpeak", "Q25", "Q50", "Q75")
CLASSES = ("flat", "mod", "tv")
DATA01 = "Data01_AfricanPenguins_seasurface_calls_acoustics.csv"
SOURCE_SHA256 = "1328f92601a84d81dbf8e1404fa0eeabc36cdf17a7626d9746aa62794827d02e"


def load_data(path: Path) -> pd.DataFrame:
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Input differs from Zenodo record 21376649/Data01. Use the original CSV.")
    data = pd.read_csv(path)
    if data[list(FEATURES)].isna().any().any() or not np.isfinite(data[list(FEATURES)]).all().all():
        raise ValueError("The six common acoustic features must be complete and finite.")
    if data.CallType.value_counts().to_dict() != {"flat": 44, "mod": 40, "tv": 24}:
        raise ValueError("Unexpected call labels or sample counts.")
    pd.to_datetime(data["Date"], format="%d/%m/%Y", errors="raise")
    data.insert(0, "row_id", np.arange(1, len(data) + 1))
    return data
