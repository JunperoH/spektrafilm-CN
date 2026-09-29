"""Film expiration (emulsion aging) — Phase 7.

The aged / expired-film look, built on the SAME density-curve machinery as
push/pull (utils/morph_curves.py), not a parallel system. The s023 morph
preserves D(0) and D_max per channel exactly by construction; the two
signature axes of aging are precisely what it forbids, and they are what this
module adds, applied to the SAMPLED per-layer density curves at pipeline init
(same gated-on-active, deep-copied, idempotent baking pattern as
FilmChemistryParams):

1. Per-channel additive BASE FOG (D(0) / toe lift), headroom-weighted so each
   layer saturates at its own amplitude: the colored shadow cast, lifted
   blacks, and shadow contrast loss no gamma change can make. Fog leaves
   D_max untouched (a fully-developed layer has no headroom left to fog).
2. DYE-DENSITY / D_max scale: uniform amplitude reduction -> the muddy,
   desaturated aged look.

Reused, not rebuilt (wired through the same bake):
- Per-channel gamma (crossover-by-slope) and global gamma: composed
  MULTIPLICATIVELY into the existing FilmChemistryParams s023 morph, so
  expiration STACKS with push/pull (pull + expired works).
- Speed loss: a rigid shift of the sensitometric curves along log-exposure
  (the film needs more light; auto-exposure meters the scene, not the film,
  so the negative honestly thins).
- Grain coarsening: multiplies the existing grain particle area.

The Age + Storage macro is a GENERIC expired-colour-negative model — expired
film has no datasheets, so the constants below are empirically-tuned,
plausible starting points meant to be calibrated BY EYE against real expired
scans and biased per stock by the user with the manual knobs on top. It makes
no claim of physical accuracy.

Print behavior falls out for free: the print filters (or auto print balance)
can compensate the GLOBAL part of the cast, but not the per-channel slope
crossover — exactly like printing a real aged negative.

Default inactive (and active-but-all-neutral) is a strict no-op: the bake is
skipped entirely and the shipped profile arrays pass through byte-identical
(test-pinned, same discipline as chemistry inactive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from spektrafilm.utils.morph_curves import FilmChemistryParams

__all__ = [
    "ExpirationParams",
    "ExpirationEffect",
    "STORAGE_CONDITIONS",
    "expiration_is_identity",
    "resolve_expiration",
    "compose_chemistry_gamma",
    "apply_expiration_to_curves",
]


# Aging-rate multipliers per storage condition (cold storage slows aging).
STORAGE_CONDITIONS = ("frozen", "cold", "room", "hot-humid")
_STORAGE_RATE = {"frozen": 0.05, "cold": 0.3, "room": 1.0, "hot-humid": 2.5}

# Saturating age response: x = 1 - exp(-effective_years / TAU). Degradation
# grows quickly at first, then levels off (fog and dye loss are bounded).
_AGING_TAU_YEARS = 12.0

# Generic expired colour negative at full degradation (x = 1). Eye-calibration
# targets, NOT datasheet values (none exist for expired film):
_FOG_FULL_RGB = (0.10, 0.14, 0.22)   # blue record fogs fastest -> colored shadows
_GAMMA_LOSS_FULL = 0.18             # global contrast loss
_CROSSOVER_FULL_RGB = (0.04, 0.0, -0.10)  # slope tilt: red up, blue down
_DENSITY_LOSS_FULL = 0.22           # D_max / dye loss -> desaturation
_GRAIN_GAIN_FULL = 0.6              # aging coarsens grain
_SPEED_LOSS_EV_PER_YEAR = 0.1       # the "one stop per decade" rule of thumb
_SPEED_LOSS_EV_MAX = 3.0


@dataclass(frozen=True)
class ExpirationParams:
    """User-facing film expiration controls.

    ``age_years`` + ``storage`` drive the generic macro; the remaining fields
    are the underlying manual knobs, NEUTRAL by default, composed on top of
    the macro (fog adds, gammas/scales multiply, EVs add) so the user can bias
    the generic model per stock.
    """

    active: bool = False
    # -- creative macro ------------------------------------------------------
    age_years: float = 0.0
    storage: str = "room"               # frozen | cold | room | hot-humid
    # -- manual knobs (biases on top of the macro; defaults are neutral) -----
    fog_rgb: tuple = (0.0, 0.0, 0.0)    # extra per-channel base fog, density
    gamma: float = 1.0                  # extra global gamma (contrast trim)
    gamma_rgb: tuple = (1.0, 1.0, 1.0)  # extra per-channel gamma (crossover)
    density_scale: float = 1.0          # extra D_max / dye-density scale
    grain_gain: float = 1.0             # extra grain particle-area gain
    speed_loss_ev: float = 0.0          # extra speed loss, EV


@dataclass(frozen=True)
class ExpirationEffect:
    """Macro + manual knobs resolved to the concrete degradation numbers."""

    fog_rgb: tuple
    gamma: float
    gamma_rgb: tuple
    density_scale: float
    grain_gain: float
    speed_loss_ev: float


def _degradation_fraction(age_years: float, storage: str) -> tuple[float, float]:
    rate = _STORAGE_RATE.get(str(storage).strip().lower(), _STORAGE_RATE["room"])
    effective_years = max(float(age_years), 0.0) * rate
    return 1.0 - math.exp(-effective_years / _AGING_TAU_YEARS), effective_years


def expiration_is_identity(p: ExpirationParams | None) -> bool:
    if p is None or not p.active:
        return True
    return (
        float(p.age_years) <= 0.0
        and tuple(float(v) for v in p.fog_rgb) == (0.0, 0.0, 0.0)
        and float(p.gamma) == 1.0
        and tuple(float(v) for v in p.gamma_rgb) == (1.0, 1.0, 1.0)
        and float(p.density_scale) == 1.0
        and float(p.grain_gain) == 1.0
        and float(p.speed_loss_ev) == 0.0
    )


def resolve_expiration(p: ExpirationParams) -> ExpirationEffect | None:
    """Resolve macro + manual knobs to concrete numbers. None when identity."""
    if expiration_is_identity(p):
        return None
    x, effective_years = _degradation_fraction(p.age_years, p.storage)

    fog = tuple(
        max(x * full + float(extra), 0.0)
        for full, extra in zip(_FOG_FULL_RGB, p.fog_rgb)
    )
    gamma = (1.0 - _GAMMA_LOSS_FULL * x) * max(float(p.gamma), 0.05)
    gamma_rgb = tuple(
        max((1.0 + tilt * x) * max(float(extra), 0.05), 0.05)
        for tilt, extra in zip(_CROSSOVER_FULL_RGB, p.gamma_rgb)
    )
    density_scale = (1.0 - _DENSITY_LOSS_FULL * x) * min(
        max(float(p.density_scale), 0.05), 2.0)
    grain_gain = (1.0 + _GRAIN_GAIN_FULL * x) * min(
        max(float(p.grain_gain), 0.05), 8.0)
    speed_loss_ev = (
        min(_SPEED_LOSS_EV_PER_YEAR * effective_years, _SPEED_LOSS_EV_MAX)
        + float(p.speed_loss_ev)
    )
    return ExpirationEffect(
        fog_rgb=fog,
        gamma=max(gamma, 0.05),
        gamma_rgb=gamma_rgb,
        density_scale=max(density_scale, 0.05),
        grain_gain=max(grain_gain, 0.05),
        speed_loss_ev=speed_loss_ev,
    )


def compose_chemistry_gamma(
    chemistry: FilmChemistryParams | None,
    effect: ExpirationEffect,
) -> FilmChemistryParams:
    """Fold the expiration gamma axes into the s023 chemistry morph params.

    Multiplicative composition onto the (possibly inactive/default) push-pull
    chemistry, so expiration and push/pull STACK through the one existing
    morph engine. Returns an ACTIVE params object for the bake."""
    base = chemistry if chemistry is not None else FilmChemistryParams(active=False)
    return replace(
        base,
        active=True,
        gamma_factor=float(base.gamma_factor) * float(effect.gamma),
        gamma_factor_red=float(base.gamma_factor_red) * float(effect.gamma_rgb[0]),
        gamma_factor_green=float(base.gamma_factor_green) * float(effect.gamma_rgb[1]),
        gamma_factor_blue=float(base.gamma_factor_blue) * float(effect.gamma_rgb[2]),
    )


def _shift_curve(log_exposure: np.ndarray, curve: np.ndarray, delta: float) -> np.ndarray:
    """D'(x) = D(x - delta): rigid shift toward higher exposure (slower film),
    edge-clamped (both curve ends are saturated plateaus)."""
    if delta == 0.0:
        return curve
    return np.interp(log_exposure - delta, log_exposure, curve)


def apply_expiration_to_curves(
    log_exposure,
    density_curves,
    density_curves_layers,
    effect: ExpirationEffect,
):
    """Apply speed loss, base fog, and dye-density loss to the SAMPLED curves.

    Order: speed shift (sensitometric, fog-independent) -> fog + scale.

    Per layer i of channel c with saturation amplitude A_i (its own sampled
    max) and channel D_max = sum_i A_i:

        layer' = scale * (layer + fog_c * (A_i / D_max) * (1 - layer / A_i))

    The headroom weight (1 - layer/A_i) makes fog a TOE-WEIGHTED additive
    floor: full fog where the layer is undeveloped, zero at saturation. It
    keeps every layer monotone (slope scaled by 1 - fog_c/D_max > 0 since fog
    is clamped below D_max) and sums to the total-curve form

        D' = scale * (D + fog_c * (1 - D / D_max)),

    so D(0) lifts by ~scale*fog_c while D_max maps to scale*D_max — fog and
    dye loss stay independent axes.

    The shipped profiles sample density_curves and density_curves_layers from
    SEPARATE fits (the total carries small measurement ripples the layer sum
    does not), so the two are NOT numerically equal at rest. To preserve that
    baseline relationship the total is transformed from the LAYER SUM only
    when it already equals it (the chemistry-morph-regenerated case, keeping
    the grain sublayers exactly consistent); otherwise the same total-form
    transform is applied to the shipped total independently, with its own
    sampled D_max. Returns (total', layers' | None).
    """
    x = np.asarray(log_exposure, dtype=float)
    total = np.asarray(density_curves, dtype=float).copy()
    layers = None
    if density_curves_layers is not None:
        layers_arr = np.asarray(density_curves_layers, dtype=float)
        if layers_arr.ndim == 3 and layers_arr.size:
            layers = layers_arr.copy()

    delta = float(effect.speed_loss_ev) * math.log10(2.0)
    scale = float(effect.density_scale)
    n_channels = total.shape[1]

    for c in range(n_channels):
        fog_c = max(float(effect.fog_rgb[c]) if c < len(effect.fog_rgb) else 0.0, 0.0)
        total_consistent = layers is not None and np.allclose(
            total[:, c], layers[:, :, c].sum(axis=1), atol=1e-9)
        if layers is not None:
            n_layers = layers.shape[1]
            amplitudes = np.array(
                [max(layers[:, i, c].max(), 0.0) for i in range(n_layers)])
            d_sum = float(amplitudes.sum())
            fog_layers = min(fog_c, 0.9 * d_sum) if d_sum > 0.0 else 0.0
            for i in range(n_layers):
                a_i = amplitudes[i]
                layer = _shift_curve(x, layers[:, i, c], delta)
                if a_i > 0.0 and fog_layers > 0.0:
                    layer = layer + fog_layers * (a_i / d_sum) * (1.0 - layer / a_i)
                layers[:, i, c] = scale * layer
        if total_consistent:
            total[:, c] = layers[:, :, c].sum(axis=1)
        else:
            d_max = float(total[:, c].max())
            curve = _shift_curve(x, total[:, c], delta)
            if d_max > 0.0 and fog_c > 0.0:
                fog_total = min(fog_c, 0.9 * d_max)
                curve = curve + fog_total * (1.0 - curve / d_max)
            total[:, c] = scale * curve

    return total, layers
