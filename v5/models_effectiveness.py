"""Transfer observed group response distributions into noise scenarios."""

import numpy as np
from scipy.special import expit

from penguin_data import STATES

ECO_WEIGHTS = np.array([[0.25, 0.55, 0.20], [0.15, 0.35, 0.50], [0.15, 0.50, 0.35]])
SCENARIOS = (
    ("Baseline", 0.50, 0.00),
    ("Threshold 0.60", 0.60, 0.00),
    ("Threshold 0.70", 0.70, 0.00),
    ("Fidelity loss 5%", 0.50, 0.05),
    ("Fidelity loss 10%", 0.50, 0.10),
    ("Fidelity loss 20%", 0.50, 0.20),
    ("Combined stress", 0.70, 0.20),
)
BASE_WIDTH = 0.05
UTILITY_THRESHOLD = 0.05
ADDITIONAL_FRACTIONS = (0.0, 0.25, 0.5, 1.0)


def ecological_values(weights=ECO_WEIGHTS, bonus=0.2):
    return (np.asarray(weights) @ STATES.T + bonus * STATES[:, 1] * STATES[:, 2]) / (1 + bonus)


def scenario_response(fidelity, natural, control, utilities, threshold, loss, width):
    baseline_gate = expit((fidelity - 0.50) / width)
    gate = expit(((1 - loss) * fidelity - threshold) / width)
    retained = np.minimum(1, gate / baseline_gate)
    change = natural - control
    observed = control + retained[..., None] * change
    index = np.minimum(gate, baseline_gate) * np.sum(change * utilities, axis=-1)
    return observed, index, retained


def functional_radius(distance_grid, values, threshold=UTILITY_THRESHOLD):
    good = np.asarray(values) >= threshold
    if not good[0]:
        return distance_grid[0], "no_segment_at_lower_bound", bool(good.any())
    first_failure = np.flatnonzero(~good)
    if not len(first_failure):
        return distance_grid[-1], "right_boundary", False
    i = first_failure[0]
    fraction = (values[i - 1] - threshold) / (values[i - 1] - values[i])
    crossing = distance_grid[i - 1] + fraction * (distance_grid[i] - distance_grid[i - 1])
    return crossing, "crossing", bool(good[i:].any())
