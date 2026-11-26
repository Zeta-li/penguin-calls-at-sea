"""Classifying acoustic recordings and fitting monotone propagation curves."""

from itertools import product

import numpy as np
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from penguin_data import CALLS, DISTANCES

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
THRESHOLDS = (0.5, 0.6, 0.7)
EQUAL_WEIGHTS = np.full(3, 1 / 3)
BLOCK_SAMPLES = np.array(list(product(range(3), repeat=3)), dtype=int)


def train_classifier(x, y, c):
    if set(y) != set(CALLS):
        raise ValueError("The classifier training fold must contain all three calls")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-9),
    ).fit(x, y)


def block_means(correlation):
    by_block = correlation.reshape(3, 3, 3, 100, 7)
    counts = np.isfinite(by_block).sum(axis=3)
    totals = np.nansum(by_block, axis=3)
    means = np.divide(
        totals, counts, out=np.full_like(totals, np.nan), where=counts > 0
    )
    return means, counts


def monotone_nodes(means, distances=DISTANCES):
    estimates = np.empty_like(means, dtype=float)
    for idx in np.ndindex(means.shape[:-1]):
        fitter = IsotonicRegression(increasing=False, y_min=-1, y_max=1)
        estimates[idx] = fitter.fit_transform(np.log2(distances), means[idx])
    assert np.all(np.diff(estimates, axis=-1) <= 1e-12)
    return estimates


def smooth_value(params, d):
    a, midpoint_log, power = params
    return a / (1 + np.exp(power * (np.log(d) - midpoint_log)))


def smooth_fit(distances, measured):
    def residual(params):
        return smooth_value(params, distances) - measured

    candidates = [
        least_squares(
            residual,
            [0.9, np.log(midpoint), 1.5],
            bounds=([0, 0, 0.05], [1, np.log(2048), 6]),
            ftol=1e-10,
        )
        for midpoint in (32, 128, 512)
    ]
    viable = [fit for fit in candidates if fit.success]
    if not viable:
        raise RuntimeError("Every smooth propagation fit failed")
    return min(viable, key=lambda fit: fit.cost).x


def evaluate_nodes(nodes, d, distances=DISTANCES):
    return np.interp(np.log2(d), np.log2(distances), nodes)


def acoustic_range(nodes, threshold):
    reached = np.flatnonzero(nodes >= threshold)
    if reached.size == 0:
        return DISTANCES[0], "below_4m"
    if reached[-1] == len(DISTANCES) - 1:
        return DISTANCES[-1], "right_boundary"
    i = reached[-1]
    ratio = (nodes[i] - threshold) / (nodes[i] - nodes[i + 1])
    return DISTANCES[i] * 2.0**ratio, "crossing"
