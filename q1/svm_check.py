"""Post-selection sensitivity of SVM hard decisions to probability calibration.

Reuses recorded C and gamma for each held-out day. Both predictions use the
same training recordings and standardization. This diagnostic is separate
from nested model selection and does not enter the candidate ranking.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classifiers import PROB_COLS, _predict
from .data import CLASSES, FEATURES, load_data


def run(source: Path, results: Path) -> pd.DataFrame:
    data = load_data(source)
    selected = pd.read_csv(results / "selection.csv")
    settings = selected.loc[selected.model.eq("svm") & selected.selected].set_index("Date")
    if not settings.index.is_unique or set(settings.index) != set(data.Date):
        raise ValueError("Expected exactly one recorded SVM setting for every test date.")
    recorded = pd.read_csv(results / "oof.csv").query("model == 'svm'").set_index("row_id")
    if not recorded.index.is_unique or set(recorded.index) != set(data.row_id):
        raise ValueError("Expected one SVM prediction per Data01 recording.")

    x = data.loc[:, list(FEATURES)].to_numpy(dtype=float)
    y = data.CallType.to_numpy()
    dates = data.Date.to_numpy()
    rows = []
    for train, test in LeaveOneGroupOut().split(x, y, dates):
        date = dates[test[0]]
        params = json.loads(settings.at[date, "parameters"])
        reference = recorded.loc[data.row_id.iloc[test]]
        np.testing.assert_array_equal(reference["actual"], y[test])
        np.testing.assert_array_equal(reference["Date"], dates[test])
        p = _predict("svm", params, x[train], y[train], dates[train], x[test])
        np.testing.assert_allclose(p, reference[PROB_COLS], atol=1e-9, rtol=1e-8)
        calibrated = np.take(CLASSES, p.argmax(axis=1))
        np.testing.assert_array_equal(calibrated, reference.prediction)

        # Same SVC that ensemble=False refits on all outer-training dates.
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", **params)).fit(x[train], y[train])
        native = svm.predict(x[test])
        rows.append(pd.DataFrame({
            "row_id": data.row_id.iloc[test], "Date": date, "actual": y[test], **params,
            "calibrated_argmax": calibrated, "native_svc": native, "changed": native != calibrated,
        }))
    comparison = pd.concat(rows).sort_values("row_id")
    comparison.to_csv(results / "svm_decision_sensitivity.csv", index=False)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results/q1/classical"))
    args = parser.parse_args()
    frame = run(args.data, args.results)
    print(f"Changed decisions: {frame.changed.sum()}/{len(frame)}")
    for name in ("calibrated_argmax", "native_svc"):
        macro = f1_score(frame.actual, frame[name], labels=CLASSES, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(frame.actual, frame[name])
        tv = frame.loc[frame.actual.eq("tv"), name].eq("tv")
        print(f"{name}: macro F1={macro:.5f}, balanced accuracy={balanced:.5f}, tv={tv.sum()}/{len(tv)}")
