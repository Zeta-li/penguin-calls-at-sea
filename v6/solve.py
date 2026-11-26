"""Run the 2026 APMCM C reference analysis from its project directory."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from models_acoustic import (
    EQUAL_WEIGHTS, acoustic_range, block_means, evaluate_nodes, monotone_nodes,
    smooth_fit, smooth_value, train_classifier,
)
from models_effectiveness import BASE_WIDTH, ecological_values, functional_radius, scenario_response
from models_playback import design, response_fit, response_probs, time_cdf, time_fit
from penguin_data import (
    BEHAVIOURS, DISTANCES, FEATURES, HORIZON, PROJECT, REPRESENTATIVE_N, STATES,
    event_flags, latency_values, read_inputs, state_index,
)


def smoke_test(acoustics, playback, correlation):
    """One real-data pass through each model family without resampling or plots."""
    test = acoustics.session_date.eq(acoustics.session_date.max())
    classifier = train_classifier(acoustics.loc[~test, list(FEATURES)], acoustics.loc[~test].CallType, 1)
    class_probs = classifier.predict_proba(acoustics.loc[test, list(FEATURES)])
    assert np.allclose(class_probs.sum(axis=1), 1)
    means, coverage = block_means(correlation)
    measured = np.nanmean(means, axis=2)
    nodes = monotone_nodes(measured)
    fidelity = EQUAL_WEIGHTS @ nodes
    flat_range, _ = acoustic_range(fidelity[0], 0.5)
    alternate = smooth_fit(DISTANCES, EQUAL_WEIGHTS @ measured[0])
    assert np.all(np.diff(smooth_value(alternate, DISTANCES)) <= 0)
    x = design(playback.CallType, playback.Context, playback.Distance, playback.Npenguins)
    flags = event_flags(playback)
    beta = response_fit(x, state_index(flags), 4)
    p = response_probs(beta, x)
    assert np.allclose(p.sum(axis=1), 1) and p.min() > 0
    onset = latency_values(playback)
    median_check = {}
    representative = design("tv", "searching", 100.0, REPRESENTATIVE_N)
    marginals = response_probs(beta, representative) @ STATES
    for j, behaviour in enumerate(BEHAVIOURS):
        time_model = time_fit(x[flags[:, j]], onset[flags[:, j], j])
        at_zero = time_cdf(time_model, representative, 0)
        at_end = time_cdf(time_model, representative, HORIZON)
        assert np.isclose(at_zero, 0) and np.isclose(at_end, 1)
        median_check[behaviour] = float(marginals[j] * time_cdf(time_model, representative, 10))
    distances = np.arange(20.0, 151)
    controls = response_probs(beta, design("control", "searching", distances, REPRESENTATIVE_N))
    calls = response_probs(beta, design("tv", "searching", distances, REPRESENTATIVE_N))
    score = evaluate_nodes(fidelity[2], distances)
    utilities = ecological_values()[1]
    ref, effective, keep = scenario_response(score, calls, controls, utilities, 0.5, 0, BASE_WIDTH)
    noisy, _, retention = scenario_response(score, calls, controls, utilities, 0.6, 0.1, BASE_WIDTH)
    assert np.allclose(ref, calls) and np.allclose(keep, 1)
    assert np.allclose(noisy.sum(axis=-1), 1) and np.all(retention <= 1)
    radius, status, _ = functional_radius(distances, effective)
    output = {
        "recordings": len(acoustics), "held_out_date_records": int(test.sum()),
        "playback_records": len(playback), "experiment_pairs": int(playback.expID.nunique()),
        "smallest_assumed_propagation_block_n": int(coverage.min()),
        "flat_acoustic_R50_m": flat_range, "tv_any_response_at_100m_searching": float(1 - calls[80, 0]),
        "tv_unconditional_behaviour_probability_by_10s_at_100m_searching": median_check,
        "tv_functional_radius_searching_m": radius, "tv_functional_radius_status": status,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def run_full(output, tables_only):
    from analyze_acoustics import quantify_acoustics, quantify_propagation
    from analyze_effectiveness import scenario_results
    from analyze_playback import analyze_playbacks

    tables = output / "results"
    tables.mkdir(parents=True, exist_ok=True)
    acoustics, playback, correlation = read_inputs()
    q1 = quantify_acoustics(acoustics, tables)
    print("Q1 complete: date-grouped classification and rank tests", flush=True)
    q2 = quantify_propagation(correlation, playback, tables)
    print("Q2 complete: monotone curves and block sensitivity", flush=True)
    q3 = analyze_playbacks(playback, tables)
    print("Q3 complete: group cross-validation and paired bootstrap", flush=True)
    q4 = scenario_results(q2, q3, tables)
    print("Q4 complete: conditional scenarios and functional radii", flush=True)
    if not tables_only:
        from figures_apmcm import render_figures

        render_figures(output / "figures", q1, q2, q3, q4)
        write_manifest(tables)


def write_manifest(tables):
    from analyze_acoustics import N_BOOT, N_PERMUTATIONS, SCAN_THRESHOLDS, WEIGHTS
    from analyze_effectiveness import GRID_STEP
    from models_acoustic import BLOCK_SAMPLES, C_GRID, THRESHOLDS
    from models_effectiveness import ADDITIONAL_FRACTIONS, ECO_WEIGHTS, SCENARIOS, UTILITY_THRESHOLD
    from models_playback import LAMBDAS, LATENCY_PENALTY, LATENCY_SHIFT
    from penguin_data import SEED, SHARED_RANGE, SOURCE
    from utils.repro_manifest import build_manifest, sha256_file

    inputs = sorted(p for p in SOURCE.iterdir() if p.is_file() and not p.name.startswith("."))
    inputs.append(PROJECT / "2026 APMCM Problem C.pdf")
    inputs.extend(PROJECT / name for name in ("题目分析报告.md", "术语表格.md"))
    parameters = {
        "q1": {"dated_permutations": N_PERMUTATIONS, "date_bootstraps": N_BOOT, "ridge_C": C_GRID},
        "q2": {"distances_m": DISTANCES.tolist(), "block_draws": len(BLOCK_SAMPLES),
               "scan_thresholds": SCAN_THRESHOLDS.tolist(), "levels": THRESHOLDS,
               "weights": {label: value.tolist() for label, value in WEIGHTS.items()}},
        "q3": {"pair_bootstraps": N_BOOT, "swap_permutations": N_PERMUTATIONS, "lambda_candidates": LAMBDAS,
               "latency_penalty": LATENCY_PENALTY, "latency_shift_s": LATENCY_SHIFT, "horizon_s": HORIZON},
        "q4": {"scenarios": SCENARIOS, "distance_step_m": GRID_STEP, "common_domain_m": SHARED_RANGE,
               "ecological_weights": ECO_WEIGHTS.tolist(), "initial_gate_width": BASE_WIDTH,
               "minimum_effectiveness": UTILITY_THRESHOLD, "conditional_additional_fractions": ADDITIONAL_FRACTIONS,
               "representative_group_size": REPRESENTATIVE_N},
    }
    command = shlex.join([sys.executable, "solve.py"])
    target = build_manifest(inputs, SEED, parameters, command, (
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow",
    ))
    source = sorted([*PROJECT.glob("solve.py"), *PROJECT.glob("penguin_data.py"),
                     *PROJECT.glob("models_*.py"), *PROJECT.glob("analyze_*.py"),
                     PROJECT / "figures_apmcm.py", *PROJECT.glob("utils/*.py")])
    target["code_files"] = {str(p.relative_to(PROJECT)): sha256_file(p) for p in source}
    target["output_tables"] = {p.name: sha256_file(p) for p in sorted(tables.glob("*.csv"))}
    (tables / "复现清单.json").write_text(json.dumps(target, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the real-data P1 checks")
    parser.add_argument("--tables-only", action="store_true", help="Run all four analyses without rendering")
    parser.add_argument("--output-dir", type=Path, default=PROJECT, help="Output directory within the project")
    args = parser.parse_args()
    if args.smoke:
        smoke_test(*read_inputs())
        return
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("--output-dir must be inside the project")
    run_full(output, args.tables_only)


if __name__ == "__main__":
    main()
