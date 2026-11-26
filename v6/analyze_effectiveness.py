"""Noise, functional radius and group-response scenarios for Q4."""

import numpy as np
import pandas as pd

from analyze_acoustics import write_table
from models_acoustic import evaluate_nodes
from models_effectiveness import (
    ADDITIONAL_FRACTIONS, BASE_WIDTH, ECO_WEIGHTS, SCENARIOS, UTILITY_THRESHOLD,
    ecological_values, functional_radius, scenario_response,
)
from models_playback import design, response_probs
from penguin_data import CALLS, CONTEXTS, REPRESENTATIVE_N, SEED, SHARED_RANGE, STIMULI

GRID_STEP = 0.25
EVALUATION_DISTANCE = 100.0
DESIGN = np.arange(SHARED_RANGE[0], SHARED_RANGE[1] + GRID_STEP / 2, GRID_STEP)
ALL_CONTEXTS = CONTEXTS + ("pooled",)


def probabilities(coefficients, distances=DESIGN, n=REPRESENTATIVE_N):
    return response_probs(
        coefficients,
        design(np.asarray(STIMULI)[:, None, None], np.asarray(CONTEXTS)[None, :, None],
               distances[None, None, :], n),
    )


def effective_point(fidelity, p, context, scenario, utilities, width):
    _, threshold, loss = scenario
    retained_results = [
        scenario_response(fidelity[k], p[k + 1], p[0], utilities[:, None, :], threshold, loss, width)
        for k, _ in enumerate(CALLS)
    ]
    scenario_prob, index, retention = map(np.asarray, zip(*retained_results))
    assert np.allclose(scenario_prob.sum(axis=-1), 1)
    assert np.all(scenario_prob >= -1e-12)
    group = 1 - scenario_prob[..., 0]
    group = np.concatenate((group, (context @ group)[:, None, :]), axis=1)
    index = np.concatenate((index, (context @ index)[:, None, :]), axis=1)
    return group, index, retention


def compute_radii(index, threshold=UTILITY_THRESHOLD):
    return [[functional_radius(DESIGN, curve, threshold) for curve in by_type] for by_type in index]


def scenario_results(propagation, playback, results):
    rng = np.random.default_rng(SEED + 4)
    context = playback["context"]
    p = probabilities(playback["parameter"])
    fidelity = np.stack([evaluate_nodes(row, DESIGN) for row in propagation["node_fit"]])
    assert fidelity.min() > 0
    utilities = ecological_values()
    distance_index = int(round((EVALUATION_DISTANCE - DESIGN[0]) / GRID_STEP))
    point = [effective_point(fidelity, p, context, sc, utilities, BASE_WIDTH) for sc in SCENARIOS]
    np.testing.assert_allclose(point[0][0][:, :3], 1 - p[1:, ..., 0])
    np.testing.assert_allclose(point[0][2], 1)
    n_fits = len(playback["boot_parameter"])
    sampled_nodes = propagation["sampled_nodes"]
    draws = rng.integers(len(sampled_nodes), size=(n_fits, len(CALLS)))
    prob_samples = np.empty((n_fits, len(SCENARIOS), len(CALLS), len(ALL_CONTEXTS)))
    rad_samples = np.empty_like(prob_samples)
    left_limit = np.zeros_like(prob_samples, dtype=bool)
    right_limit = np.zeros_like(prob_samples, dtype=bool)
    for i, beta in enumerate(playback["boot_parameter"]):
        prob = probabilities(beta)
        resampled_fidelity = np.stack([
            evaluate_nodes(sampled_nodes[idx, k], DESIGN) for k, idx in enumerate(draws[i])
        ])
        for a, sc in enumerate(SCENARIOS):
            response, indices, _ = effective_point(resampled_fidelity, prob, context, sc, utilities, BASE_WIDTH)
            for k, by_type in enumerate(compute_radii(indices)):
                for c, (radius, status, _) in enumerate(by_type):
                    rad_samples[i, a, k, c] = radius
                    left_limit[i, a, k, c] = status == "no_segment_at_lower_bound"
                    right_limit[i, a, k, c] = status == "right_boundary"
            prob_samples[i, a] = response[:, :, distance_index]
    r_bounds = np.quantile(rad_samples, (0.025, 0.975), axis=0)
    p_bounds = np.quantile(prob_samples, (0.025, 0.975), axis=0)
    p_delta_bounds = np.quantile(prob_samples - prob_samples[:, :1], (0.025, 0.975), axis=0)

    scenario_rows, curve_records, count_records = [], [], []
    for a, ((name, tau, loss), (response, indices, retain)) in enumerate(zip(SCENARIOS, point)):
        radii = compute_radii(indices)
        for k, call in enumerate(CALLS):
            for c, ctxt in enumerate(ALL_CONTEXTS):
                radius, status, disconnected = radii[k][c]
                index_100m = indices[k, c, distance_index]
                at_100m = response[k, c, distance_index]
                scenario_rows.append((
                    name, tau, loss, call, ctxt, REPRESENTATIVE_N, radius, status, disconnected,
                    *r_bounds[:, a, k, c], left_limit[:, a, k, c].sum(), right_limit[:, a, k, c].sum(),
                    at_100m, *p_bounds[:, a, k, c],
                    at_100m - point[0][0][k, c, distance_index], *p_delta_bounds[:, a, k, c], index_100m,
                ))
                for i in range(0, len(DESIGN), 4):
                    curve_records.append((name, call, ctxt, DESIGN[i], indices[k, c, i], response[k, c, i],
                                          (1 - loss) * fidelity[k, i], retain[k, i]))
                for fraction in ADDITIONAL_FRACTIONS:
                    factor = 1 + fraction * (REPRESENTATIVE_N - 1)
                    expected = at_100m * factor
                    assert 0 <= expected <= REPRESENTATIVE_N
                    count_records.append((name, call, ctxt, EVALUATION_DISTANCE, REPRESENTATIVE_N,
                                          fraction, expected, *(p_bounds[:, a, k, c] * factor)))
    scenarios = pd.DataFrame(scenario_rows, columns=[
        "scenario", "recognition_threshold", "relative_fidelity_loss", "call", "context", "group_size",
        "functional_radius_m", "radius_status", "disconnected_segment", "radius_lower_m", "radius_upper_m",
        "draws_no_segment_at_lower_bound", "draws_right_boundary", "any_at_100m", "p_lower", "p_upper",
        "p_change_at_100m", "p_change_lower", "p_change_upper", "index_at_100m",
    ])
    curves = pd.DataFrame(curve_records, columns=[
        "scenario", "call", "context", "distance_m", "effectiveness", "any_response", "fidelity", "retained_excess",
    ])
    counts = pd.DataFrame(count_records, columns=[
        "scenario", "call", "context", "distance_m", "group_size", "additional_fraction", "expected_responders",
        "expected_lower", "expected_upper",
    ])

    higher_approach = ECO_WEIGHTS * 1.0
    higher_approach[:, 2] *= 2.0
    higher_approach /= higher_approach.sum(axis=1, keepdims=True)
    checks = [
        ("base", utilities, BASE_WIDTH, UTILITY_THRESHOLD),
        *((f"epsilon_{eps:g}", utilities, BASE_WIDTH, eps) for eps in (0.025, 0.075, 0.10)),
        *((f"width_{width:g}", utilities, width, UTILITY_THRESHOLD) for width in (0.025, 0.10)),
        ("equal_ecological_weights", ecological_values(np.full((3, 3), 1 / 3)), BASE_WIDTH, UTILITY_THRESHOLD),
        ("no_cooccurrence_premium", ecological_values(bonus=0), BASE_WIDTH, UTILITY_THRESHOLD),
        ("upweighted_recruitment", ecological_values(higher_approach), BASE_WIDTH, UTILITY_THRESHOLD),
    ]
    sens = []
    for label, alternative, width, epsilon in checks:
        for scenario in SCENARIOS:
            _, scores, _ = effective_point(fidelity, p, context, scenario, alternative, width)
            for k, by_call in enumerate(compute_radii(scores, epsilon)):
                for c, (radius, state, disjoint) in enumerate(by_call):
                    sens.append((label, scenario[0], CALLS[k], ALL_CONTEXTS[c], width, epsilon,
                                 radius, state, disjoint, scores[k, c, distance_index]))
    sensitivity = pd.DataFrame(sens, columns=[
        "variant", "scenario", "call", "context", "width", "effectiveness_cutoff", "radius_m",
        "status", "disconnected_segment", "index_at_100m",
    ])
    scenario_grid = playback["raw"].loc[playback["raw"].Type.eq("call"), ["expID", "Context", "Npenguins"]]
    for table, name in (
        (scenarios, "q4_scenario_results"), (curves, "q4_curves"),
        (counts, "q4_expected_responders"), (sensitivity, "q4_sensitivity"),
        (scenario_grid, "q4_group_context"),
    ):
        write_table(table, results, name)
    return {
        "scenarios": scenarios, "curves": curves, "counts": counts,
        "sensitivity": sensitivity, "raw_groups": scenario_grid,
    }
