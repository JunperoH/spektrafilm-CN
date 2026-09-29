"""Phase 10B: fit a norm_cdfs DensityCurvesModel to sampled density curves.

Custom film stocks (recipes) generate their curves from the parametric
hyperbolic-log family (model/parametric.py) or, later, from drawn spline
points (Phase 10D). The morph engines — push/pull chemistry and expiration —
operate THROUGH DensityCurvesModel (sums of Gaussian CDFs per emulsion
sublayer), so every custom stock must carry a valid norm_cdfs decomposition.
This module provides that fit-back.

The fit is per channel: n_layers Gaussian-CDF layers

    D_c(x) = sum_i A_i * Phi(s * (x - c_i) / sigma_i),   s = -1 for positives

solved with scipy.optimize.least_squares under physical bounds (amplitudes
and sigmas positive). Deterministic initialization from the cumulative
density rise, so the same curve always fits to the same model (no RNG).
"""

from __future__ import annotations

import numpy as np

from spektrafilm.profiles.io import DensityCurvesModel

__all__ = ["fit_density_curves_model"]

_MIN_SIGMA = 0.05
_MAX_SIGMA = 5.0


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _channel_model(x: np.ndarray, centers, amplitudes, sigmas, sign: float) -> np.ndarray:
    total = np.zeros_like(x)
    for c, a, s in zip(centers, amplitudes, sigmas):
        total += a * _norm_cdf(sign * (x - c) / s)
    return total


def _initial_guess(x: np.ndarray, y: np.ndarray, n_layers: int, sign: float):
    """Centers at the quantiles of the density-rise mass, equal amplitude
    split, sigmas from the active span — deterministic."""
    rise = np.diff(y * sign)
    rise = np.maximum(rise, 0.0)
    mass = np.cumsum(rise)
    span = float(x[-1] - x[0])
    if mass[-1] <= 0.0:
        centers = np.linspace(x[0] + 0.25 * span, x[-1] - 0.25 * span, n_layers)
    else:
        mass = mass / mass[-1]
        x_mid = 0.5 * (x[1:] + x[:-1])
        quantiles = (np.arange(n_layers) + 0.5) / n_layers
        centers = np.interp(quantiles, mass, x_mid)
    amplitude_total = float(abs(y[-1] - y[0]))
    amplitudes = np.full(n_layers, max(amplitude_total, 1e-3) / n_layers)
    sigmas = np.full(n_layers, max(span / (3.0 * n_layers), 2.0 * _MIN_SIGMA))
    return centers, amplitudes, sigmas


def fit_density_curves_model(
    log_exposure,
    density_curves,
    *,
    n_layers: int = 3,
    profile_type: str = "negative",
) -> tuple[DensityCurvesModel, np.ndarray]:
    """Fit an n_layers norm_cdfs model to (n_samples, n_channels) curves.

    Returns ``(model, rms_residual_per_channel)``. The model's baseline is
    the curve MINIMUM per channel: Gaussian-CDF sums start at 0, so a fog
    floor in the sampled curve is absorbed into the layer amplitudes only
    where the fit can reach it — the residual reports what is left. This is
    the same contract the shipped profiles live with (their sampled totals
    carry measurement ripples the layer sum does not).
    """
    from scipy.optimize import least_squares

    x = np.asarray(log_exposure, dtype=float)
    curves = np.asarray(density_curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError("density_curves must be (n_samples, n_channels)")
    n_channels = curves.shape[1]
    sign = -1.0 if profile_type == "positive" else 1.0

    centers = np.empty((n_channels, n_layers))
    amplitudes = np.empty((n_channels, n_layers))
    sigmas = np.empty((n_channels, n_layers))
    residuals = np.empty(n_channels)

    for channel in range(n_channels):
        y = curves[:, channel]
        floor = float(y.min())
        target = y - floor
        c0, a0, s0 = _initial_guess(x, target + floor, n_layers, sign)

        def pack(c, a, s):
            return np.concatenate([c, a, s])

        def unpack(v):
            return v[:n_layers], v[n_layers:2 * n_layers], v[2 * n_layers:]

        def residual(v):
            c, a, s = unpack(v)
            return _channel_model(x, c, a, s, sign) - target

        span = float(x[-1] - x[0])
        lower = pack(np.full(n_layers, x[0] - span),
                     np.zeros(n_layers),
                     np.full(n_layers, _MIN_SIGMA))
        upper = pack(np.full(n_layers, x[-1] + span),
                     np.full(n_layers, max(float(target.max()), 1e-3) * 2.0 + 1e-6),
                     np.full(n_layers, _MAX_SIGMA))
        guess = np.clip(pack(c0, a0, s0), lower, upper)
        solution = least_squares(residual, guess, bounds=(lower, upper), max_nfev=4000)
        c, a, s = unpack(solution.x)
        order = np.argsort(c)
        centers[channel] = c[order]
        amplitudes[channel] = a[order]
        sigmas[channel] = s[order]
        residuals[channel] = float(np.sqrt(np.mean(solution.fun ** 2)))

    model = DensityCurvesModel(
        model_type="norm_cdfs",
        centers=centers,
        amplitudes=amplitudes,
        sigmas=sigmas,
    )
    return model, residuals
