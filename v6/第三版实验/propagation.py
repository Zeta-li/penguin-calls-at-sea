"""Monotone B-spline sensitivity analysis for the Q2 propagation data."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from models_acoustic import (BLOCK_SAMPLES, EQUAL_WEIGHTS, THRESHOLDS, acoustic_range,
                             block_means, evaluate_nodes, monotone_nodes, smooth_fit, smooth_value)
from penguin_data import CALLS, DISTANCES, METRICS, read_inputs

# One prespecified internal knot at the midpoint of the observation domain on log2 scale.
DEGREE = 3
KNOTS = [2.] * 4 + [5.] + [8.] * 4
BASIS = BSpline(KNOTS, np.eye(5), DEGREE, extrapolate=False)
SMOOTHNESS = 0.1
D1 = np.diff(np.eye(5), axis=0)
D2 = np.diff(np.eye(5), n=2, axis=0)


def basis(distances):
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    if not np.all((distances >= DISTANCES[0]) & (distances <= DISTANCES[-1])):
        raise ValueError("Predictions must stay within 4 to 256 m")
    return BASIS(np.log2(distances))


def fit_one(design_matrix, values):
    augmented = np.vstack((design_matrix, np.sqrt(SMOOTHNESS) * D2))
    target = np.r_[values, 0., 0., 0.]
    def objective(coefs):
        residual = augmented @ coefs - target
        return .5 * residual @ residual, augmented.T @ residual
    solution = minimize(objective, np.full(5, np.mean(values)), jac=True, method="SLSQP",
                        bounds=[(-1., 1.)] * 5, constraints=LinearConstraint(D1, -np.inf, 0.),
                        options={"ftol": 1e-13, "maxiter": 200})
    if not solution.success or np.any(np.diff(solution.x) > 1e-8):
        raise ValueError(f"Invalid monotone fit: {solution.message}")
    return solution.x


def fit_curves(means=None, distances=DISTANCES):
    """Return coefficients (call, representation, basis) from equal working-block means."""
    if means is None:
        blocks, _ = block_means(read_inputs()[2])
        means = blocks.mean(axis=2)
    means = np.asarray(means)
    if means.shape != (len(CALLS), len(METRICS), len(distances)) or not np.isfinite(means).all():
        raise ValueError("Expected a finite cell mean at every modeled distance")
    design_matrix = basis(distances)
    return np.array([[fit_one(design_matrix, row) for row in group] for group in means])


def fidelity(coefficients, distances):
    """Equal-weight smooth fidelity at in-domain distances, shape (3, n)."""
    return np.mean(coefficients, axis=1) @ basis(distances).T


def spline_range(composite, threshold):
    """Largest distance satisfying the threshold, with observable-domain censoring."""
    lo, hi = DISTANCES[0], DISTANCES[-1]
    if composite[0] < threshold:
        return lo, "below_4m"
    if composite[-1] >= threshold:
        return hi, "right_boundary"
    for _ in range(45):
        mid = .5 * (lo + hi)
        if BASIS(np.log2(mid)) @ composite >= threshold:
            lo = mid
        else:
            hi = mid
    return lo, "crossing"


def validate(means, held):
    records = []
    for j in held:
        keep = np.arange(7) != j
        train_d, query = DISTANCES[keep], DISTANCES[j]
        reduced = means[:, :, keep]
        coefs = fit_curves(reduced, train_d)
        iso_nodes = monotone_nodes(reduced, train_d)
        for k, call in enumerate(CALLS):
            for m, group in enumerate((*METRICS, "equal_weight")):
                if group == "equal_weight":
                    target = means[k].mean(axis=0)
                    iso = EQUAL_WEIGHTS @ iso_nodes[k]
                    c = EQUAL_WEIGHTS @ coefs[k]
                else:
                    target, iso, c = means[k, m], iso_nodes[k, m], coefs[k, m]
                # The V2 parametric comparator fits the composite directly at this level.
                params = smooth_fit(train_d, target[keep])
                records.append((call, group, query, target[j], evaluate_nodes(iso, query, train_d),
                                smooth_value(params, query), (basis([query]) @ c).item()))
    return pd.DataFrame(records, columns=["call", "representation", "distance_m", "observed",
                                          "pava", "parametric", "monotone_spline"])


def scores(diagnostics):
    records = []
    for group, chunk in diagnostics.groupby("representation", sort=False):
        subsets = [("all", chunk), *chunk.groupby("call", sort=False)]
        for name, subset in subsets:
            for model in ("pava", "parametric", "monotone_spline"):
                diff = subset[model] - subset.observed
                records.append((group, name, model, len(diff), np.sqrt(np.mean(diff**2)),
                                np.mean(np.abs(diff))))
    return pd.DataFrame(records, columns=["representation", "call", "model", "n", "rmse", "mae"])


def evaluate_ranges(blocks, means, fitted):
    nodes = monotone_nodes(means)
    # Repeated block-count configurations have identical means. Keep multiplicities
    # of ordered draws when evaluating the same conditional percentile as in V2.
    draws, multiplicities = np.unique(np.sort(BLOCK_SAMPLES, axis=1), axis=0, return_counts=True)
    sampled = [fit_curves(np.take(blocks, draw, axis=2).mean(axis=2)).mean(axis=1) for draw in draws]
    records = []
    for k, call in enumerate(CALLS):
        for threshold in THRESHOLDS:
            pava_distance, pava_status = acoustic_range(EQUAL_WEIGHTS @ nodes[k], threshold)
            est, status = spline_range(fitted[k].mean(axis=0), threshold)
            sample = [spline_range(one[k], threshold) for one in sampled]
            radii = np.repeat([row[0] for row in sample], multiplicities)
            states = np.repeat([row[1] for row in sample], multiplicities)
            lo, hi = np.quantile(radii, [.025, .975])
            records.append((call, threshold, pava_distance, pava_status, est, status,
                            est - pava_distance, lo, hi, sum(states == "below_4m"),
                            sum(states == "right_boundary")))
    return pd.DataFrame(records, columns=["call","threshold","pava_m","pava_status","spline_m",
                                         "spline_status","model_difference_m","spline_lower_m",
                                         "spline_upper_m","draws_below_4m","draws_right_boundary"])


def run(output, smoke=False):
    blocks, _ = block_means(read_inputs()[2])
    means = blocks.mean(axis=2)
    held = [3] if smoke else range(1, len(DISTANCES)-1)
    diagnostics = validate(means, held)
    shared = diagnostics.loc[diagnostics.representation == "equal_weight"]
    old = pd.read_csv(ROOT/"results/q2_held_out_distances.csv")
    keys = ["call", "distance_m"]
    both = shared.merge(old, on=keys, validate="one_to_one")
    assert len(both) == len(shared)
    np.testing.assert_allclose(both[["observed","pava","parametric"]].to_numpy(),
                               both[["held_out_block_mean","isotonic_prediction","smooth_prediction"]].to_numpy(),
                               rtol=1e-7, atol=1e-9)
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        diagnostics.to_csv(output/"q2v3_smoke.csv", index=False, float_format="%.12g")
        return
    coef = fit_curves(means)
    ranges = evaluate_ranges(blocks, means, coef)
    grid = 2.**np.linspace(2,8,241)
    values = coef @ basis(grid).T
    assert np.max(np.diff(values,axis=-1)) < 1e-8 and np.max(np.abs(values)) <= 1.+1e-8
    curves = pd.DataFrame([(call, metric, d, f) for k,call in enumerate(CALLS)
        for metric, series in zip((*METRICS, "equal_weight"), [*values[k],values[k].mean(axis=0)])
        for d,f in zip(grid, series)], columns=["call","representation","distance_m","spline"])
    coefficients = pd.DataFrame([(call, group, *coef[k,m]) for k,call in enumerate(CALLS)
        for m,group in enumerate(METRICS)], columns=["call","representation",*(f"c{i}" for i in range(5))])
    for name, table in (("held_out",diagnostics), ("scores",scores(diagnostics)),
                        ("ranges",ranges), ("curves",curves), ("coefficients",coefficients)):
        table.to_csv(output/f"q2v3_{name}.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output,args.smoke)
