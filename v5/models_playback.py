"""Structured eight-state response probabilities and truncated onset timing."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr

from penguin_data import CALLS, CONTEXTS, HORIZON, STATES

LAMBDAS = (0.5, 1, 2, 4, 8, 16)
LATENCY_PENALTY = 4.0
LATENCY_SHIFT = 0.5
FEATURE_NAMES = (
    "intercept", "flat", "mod", "tv", "searching", "after",
    "log2(distance/50)", "log2(group_size/8)", "call:searching", "call:after",
)
INTERACTIONS = np.column_stack(
    (
        STATES[:, 0] * STATES[:, 1],
        STATES[:, 0] * STATES[:, 2],
        STATES[:, 1] * STATES[:, 2],
        np.prod(STATES, axis=1),
    )
)
NORMAL_DENSITY = 1.0 / np.sqrt(2.0 * np.pi)


def design(stimulus, context, distance, n):
    stimulus, context, distance, n = np.broadcast_arrays(stimulus, context, distance, n)
    is_call = stimulus != "control"
    return np.stack(
        (
            np.ones_like(distance, dtype=float),
            *(stimulus == call for call in CALLS),
            *(context == context_name for context_name in CONTEXTS[1:]),
            np.log2(distance / 50.0),
            np.log2(n / 8.0),
            *(is_call & (context == context_name) for context_name in CONTEXTS[1:]),
        ),
        axis=-1,
    )


def response_log_probs(coeffs, x):
    unary = coeffs[: len(FEATURE_NAMES) * 3].reshape(-1, 3)
    state_scores = (x @ unary) @ STATES.T
    pair_coefficients = coeffs[len(FEATURE_NAMES) * 3:]
    if len(pair_coefficients):
        state_scores += INTERACTIONS @ pair_coefficients
    return state_scores - logsumexp(state_scores, axis=-1, keepdims=True)


def response_probs(coeffs, x):
    return np.exp(response_log_probs(coeffs, x))


def response_fit(x, y, penalty, dependent=True):
    penalties = np.ones(x.shape[1] * 3 + dependent * INTERACTIONS.shape[1])
    penalties[:3] = 0.1
    rows = np.arange(len(y))
    target_unary = STATES[y]
    target_dependence = INTERACTIONS[y]

    def loss_gradient(beta):
        logp = response_log_probs(beta, x)
        probs = np.exp(logp)
        risk = -logp[rows, y].sum() + 0.5 * penalty * np.dot(penalties * beta, beta)
        unary_gradient = (x.T @ (probs @ STATES - target_unary)).ravel()
        if dependent:
            dependence_gradient = (probs @ INTERACTIONS - target_dependence).sum(axis=0)
            gradient = np.concatenate((unary_gradient, dependence_gradient))
        else:
            gradient = unary_gradient
        return risk, gradient + penalty * penalties * beta

    optimum = minimize(
        loss_gradient, np.zeros(len(penalties)), method="L-BFGS-B", jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not optimum.success or not np.isfinite(optimum.x).all():
        raise RuntimeError(f"Eight-state fit failed: {optimum.message}")
    return optimum.x


def time_cdf(params, x, t, shift=LATENCY_SHIFT):
    mu = x @ params[:-1]
    sigma = np.exp(params[-1])
    low = ndtr((np.log(shift) - mu) / sigma)
    high = ndtr((np.log(HORIZON + shift) - mu) / sigma)
    bounded_t = np.clip(t, 0, HORIZON)
    q = ndtr((np.log(bounded_t + shift) - mu) / sigma)
    return np.clip((q - low) / np.maximum(high - low, 1e-14), 0, 1)


def time_fit(x, t, shift=LATENCY_SHIFT):
    if len(t) == 0:
        return None
    logs = np.log(t + shift)
    anchor_low = np.log(shift)
    anchor_high = np.log(HORIZON + shift)

    def loss_gradient(params):
        location = x @ params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)
        z = (logs - location) / sigma
        a = (anchor_low - location) / sigma
        b = (anchor_high - location) / sigma
        mass = np.maximum(ndtr(b) - ndtr(a), 1e-14)
        pdf_a = NORMAL_DENSITY * np.exp(-a**2 / 2)
        pdf_b = NORMAL_DENSITY * np.exp(-b**2 / 2)
        penalty = LATENCY_PENALTY * params[1:-1]
        val = 0.5 * z**2 + log_sigma + np.log(mass) + logs
        val = val.sum() + 0.5 * penalty @ params[1:-1]
        slope = x.T @ ((-z + (pdf_a - pdf_b) / mass) / sigma)
        slope[1:] += penalty
        spread = np.sum(1.0 - z**2 + (a * pdf_a - b * pdf_b) / mass)
        return val, np.r_[slope, spread]

    initial = np.zeros(x.shape[1] + 1)
    initial[0] = np.median(logs)
    initial[-1] = np.log(np.clip(np.std(logs), 0.2, 2.0))
    fit = minimize(
        loss_gradient, initial, jac=True, method="L-BFGS-B",
        bounds=[(-6, 6)] * x.shape[1] + [(np.log(0.1), np.log(3.0))],
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Time fit failed with n={len(t)}: {fit.message}")
    return fit.x
