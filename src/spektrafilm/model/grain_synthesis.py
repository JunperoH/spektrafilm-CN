"""Phase 11C: the 'synthesis' grain model — an inhomogeneous Boolean grain
field rendered by Monte Carlo (the OFX plugin's research tier, manual 4.8 +
7.15).

Model: silver grains are disks whose centers form a Poisson process in the
film plane (intensity lam0 per um^2) with lognormal radii; a grain DEVELOPS
with the local development probability p = (D + d_min)/(D_max + d_min)
(thinning -> the inhomogeneous Boolean model). A point is covered when any
developed grain contains it; density follows the manual's formula

    coverage = P(grain covers the observation point)
    density ~= -log10(max(1 - coverage, epsilon))

For a thinned Boolean model E[coverage] = 1 - exp(-p*lam0*E[pi R^2]), so
choosing lam0 * E[pi R^2] = (D_max + d_min) * ln(10) makes the EXPECTED
density exactly p * (D_max + d_min) — the model reproduces the input
density in the mean and generates the texture stochastically around it.

Rendering: per output pixel, `samples` Monte Carlo points drawn from a
Gaussian observation aperture (sigma = half the pixel size by default,
divided by Synthesis Sharpness) test coverage against the grains of the
surrounding cells. EVERYTHING is counter-based hash RNG (cell/pixel/sample
coordinates + seed): the result is bit-deterministic regardless of thread
scheduling — unlike the production model's thread-unordered numba RNG.

This tier is EXPENSIVE by design (research / final-quality); the simplified
Size / Amount / Sharpness / Quality knobs drive it day-to-day and the
advanced Boolean controls are a collapsed group. The OFX's 'Layered' depth
split (three stacked fields, radii scaled per layer, each a third of D_max)
landed in the 2026-07-21 polish pass.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from spektrafilm.runtime.params_schema import GrainParams

_LN10 = math.log(10.0)


@njit(cache=True, inline='always')
def _mix64(x):
    x = (x ^ (x >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
    x = (x ^ (x >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53)
    return x ^ (x >> np.uint64(33))


@njit(cache=True, inline='always')
def _hash01(a, b, c, seed):
    """Deterministic uniform in (0, 1) from three counters + seed."""
    h = _mix64(np.uint64(a) * np.uint64(0x9E3779B97F4A7C15)
               ^ np.uint64(b) * np.uint64(0xBF58476D1CE4E5B9)
               ^ np.uint64(c) * np.uint64(0x94D049BB133111EB)
               ^ np.uint64(seed))
    return (float(h >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline='always')
def _cell_poisson(ci, cj, lam, max_count, seed):
    """Knuth Poisson from the cell's hash stream, capped at max_count."""
    limit = math.exp(-lam)
    product = 1.0
    count = 0
    while count <= max_count:
        product *= _hash01(ci, cj, 900000 + count, seed)
        if product <= limit:
            break
        count += 1
    return min(count, max_count)


@njit(parallel=True, cache=True)
def _synthesis_channel(p_map, pixel_um, mu_r, sigma_r, r_max, lam_area,
                       aperture_sigma_um, samples, cell_um, max_grains,
                       epsilon, seed):
    height, width = p_map.shape
    out = np.empty((height, width))
    inv_cell = 1.0 / cell_um
    rings = 1 + int(r_max * inv_cell)
    for py in prange(height):
        for px in range(width):
            center_x = (px + 0.5) * pixel_um
            center_y = (py + 0.5) * pixel_um
            covered = 0
            for s in range(samples):
                u1 = _hash01(px, py, s * 2, seed + 7)
                u2 = _hash01(px, py, s * 2 + 1, seed + 7)
                radius_n = math.sqrt(-2.0 * math.log(u1))
                sample_x = center_x + radius_n * math.cos(2.0 * math.pi * u2) * aperture_sigma_um
                sample_y = center_y + radius_n * math.sin(2.0 * math.pi * u2) * aperture_sigma_um
                cell_i = int(math.floor(sample_x * inv_cell))
                cell_j = int(math.floor(sample_y * inv_cell))
                hit = False
                for di in range(-rings, rings + 1):
                    if hit:
                        break
                    for dj in range(-rings, rings + 1):
                        if hit:
                            break
                        ci = cell_i + di
                        cj = cell_j + dj
                        lam_cell = lam_area * cell_um * cell_um
                        count = _cell_poisson(ci, cj, lam_cell, max_grains, seed)
                        for g in range(count):
                            gx = (ci + _hash01(ci, cj, g * 8, seed)) * cell_um
                            gy = (cj + _hash01(ci, cj, g * 8 + 1, seed)) * cell_um
                            # lognormal radius, quantile-clamped
                            v1 = _hash01(ci, cj, g * 8 + 2, seed)
                            v2 = _hash01(ci, cj, g * 8 + 3, seed)
                            z = math.sqrt(-2.0 * math.log(v1)) * math.cos(2.0 * math.pi * v2)
                            radius = math.exp(mu_r + sigma_r * z)
                            if radius > r_max:
                                radius = r_max
                            dx = sample_x - gx
                            dy = sample_y - gy
                            if dx * dx + dy * dy > radius * radius:
                                continue
                            # develops with the LOCAL probability (thinning)
                            gpx = int(gx / pixel_um)
                            gpy = int(gy / pixel_um)
                            if gpx < 0:
                                gpx = 0
                            elif gpx >= width:
                                gpx = width - 1
                            if gpy < 0:
                                gpy = 0
                            elif gpy >= height:
                                gpy = height - 1
                            if _hash01(ci, cj, g * 8 + 4, seed) < p_map[gpy, gpx]:
                                hit = True
                                break
                if hit:
                    covered += 1
            remainder = 1.0 - covered / samples
            if remainder < epsilon:
                remainder = epsilon
            out[py, px] = -math.log10(remainder)
    return out


def coverage_to_density(coverage, epsilon):
    """The manual's formula, exposed for the test pin."""
    return -np.log10(np.maximum(1.0 - np.asarray(coverage, dtype=float),
                                float(epsilon)))


def apply_grain_synthesis(density_cmy, pixel_size_um, grain: GrainParams,
                          density_curves):
    density_min = np.array(grain.density_min, dtype=float)
    density_max = np.nanmax(np.asarray(density_curves, dtype=float), axis=0) + density_min
    size = max(float(grain.synthesis_size), 0.05)
    amount = float(grain.synthesis_amount)
    sharpness = max(float(grain.synthesis_sharpness), 0.05)
    quality = max(float(grain.synthesis_quality), 0.05)
    samples = max(1, int(round(grain.synthesis_samples * quality)))
    stddev_ratio = max(float(grain.synthesis_radius_stddev_ratio), 1e-3)
    aperture = float(grain.synthesis_aperture_sigma_um)
    if aperture <= 0.0:            # auto: half the pixel footprint
        aperture = 0.5 * float(pixel_size_um)
    aperture /= sharpness
    epsilon = max(float(grain.synthesis_coverage_epsilon), 1e-9)
    max_grains = max(1, int(grain.synthesis_max_grains_per_cell))

    from scipy.stats import norm

    z_quantile = float(norm.ppf(np.clip(
        float(grain.synthesis_max_radius_quantile), 0.5, 1 - 1e-9)))

    # Polish 2026-07-21: the OFX 'Layered' depth split. Each depth layer is
    # an independent Boolean field with its own radius scale, calibrated to
    # a THIRD of the channel's D_max — stacked emulsion layers' densities
    # add, so the expected total still reproduces the input exactly.
    layered = bool(getattr(grain, 'synthesis_layered', False))
    layer_scales = (tuple(getattr(grain, 'synthesis_layer_scale', (2.0, 1.0, 0.5)))
                    if layered else (1.0,))

    out = np.empty_like(density_cmy)
    for ch in range(3):
        base_radius = (float(grain.synthesis_mean_radius_um) * size
                       * float(grain.synthesis_radius_scale[ch]))
        p_map = np.clip(
            (density_cmy[:, :, ch] + density_min[ch]) / density_max[ch],
            1e-6, 1.0 - 1e-6)
        p_contiguous = np.ascontiguousarray(p_map)

        gross = np.zeros(density_cmy.shape[:2])
        for layer, layer_scale in enumerate(layer_scales):
            mean_radius = max(base_radius * float(layer_scale), 1e-3)
            std_radius = stddev_ratio * mean_radius
            sigma_r = math.sqrt(math.log(1.0 + (std_radius / mean_radius) ** 2))
            mu_r = math.log(mean_radius) - 0.5 * sigma_r ** 2
            r_max = math.exp(mu_r + sigma_r * z_quantile)
            expected_area = math.pi * (mean_radius ** 2 + std_radius ** 2)
            # calibration: full exposure saturates at this layer's share
            d_max_layer = float(density_max[ch]) / len(layer_scales)
            lam_area = d_max_layer * _LN10 / expected_area
            cell_um = max(float(grain.synthesis_cell_size_ratio) * mean_radius, 1e-2)
            # NAT 2026-07-22: GPU fast path (one dispatch per channel/layer
            # field; statistical parity, see gpu/grain_synthesis.py). None
            # falls through to the numba path unchanged.
            _gpu_field = None
            try:
                from spektrafilm.gpu.grain_synthesis import synthesis_grain_dispatch
                _gpu_field = synthesis_grain_dispatch(
                    p_contiguous, pixel_um=float(pixel_size_um), mu_r=mu_r,
                    sigma_r=sigma_r, r_max=r_max, lam_area=lam_area,
                    aperture_sigma_um=aperture, samples=samples,
                    cell_um=cell_um, max_grains=max_grains, epsilon=epsilon,
                    seed=11 + ch + 1000 * layer)
            except Exception:
                _gpu_field = None
            if _gpu_field is not None:
                gross += _gpu_field
                continue
            gross += _synthesis_channel(
                p_contiguous, float(pixel_size_um), mu_r, sigma_r,
                r_max, lam_area, aperture, samples, cell_um, max_grains,
                epsilon, 11 + ch + 1000 * layer)
        synth = gross - density_min[ch]
        out[:, :, ch] = (density_cmy[:, :, ch]
                         + amount * (synth - density_cmy[:, :, ch]))
    return np.maximum(out, -density_min[None, None, :])
