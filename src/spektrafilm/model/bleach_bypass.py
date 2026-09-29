"""Phase 13B: bleach bypass — retained-silver behavior (modeled extension).

Skipping (or weakening) the bleach leaves developed silver in the emulsion
alongside the dyes. The model, matching the OFX manual's description (7.14)
and its honesty framing ('a physically motivated model extension, not
measured stock-specific bleach data'):

  total_density(lambda) = dye_spectral_density(lambda) + silver_density

with silver_density = amount * mean(developed CMY density) per pixel — the
silver image the bleach would have removed is proportional to development.
The silver term is SPECTRALLY FLAT (conservative neutral model), which
makes it exact through every spectral integral as a light factor:

  light(lambda) = 10^-(spectral + S) = 10^-S * 10^-spectral
  =>  raw/XYZ   = 10^-S * integral(...)

so callers apply ``10 ** -silver`` AFTER their (GPU-fused) spectral reduce
— no spectral kernel changes, and the transform stays a pure per-CMY-value
function (LUT-compatible).

The NEGATIVE side additionally carries the leuco-cyan component: without
the bleach's oxidation some cyan dye never forms (stays leuco/colorless).
``leuco_cyan_coupling`` scales it; 1.0 is the intended path (manual 6.7).
LEUCO_CYAN_LOSS is this model's documented coefficient, not measured data.
"""
from __future__ import annotations

import numpy as np

# Fraction of the cyan dye lost at bleach_bypass_amount=1, coupling=1.
# A modeled coefficient (see module docstring) — deliberately conservative.
LEUCO_CYAN_LOSS = 0.2

__all__ = ["LEUCO_CYAN_LOSS", "bypass_effective_cmy", "retained_silver_density"]


def retained_silver_density(density_cmy, amount: float):
    """Per-pixel neutral silver density (..., 1): amount x mean developed
    density. Zero amount -> zeros (callers usually gate first)."""
    if amount <= 0.0:
        return np.zeros(density_cmy.shape[:-1] + (1,))
    return float(amount) * np.mean(np.fmax(density_cmy, 0.0), axis=-1,
                                   keepdims=True)


def bypass_effective_cmy(density_cmy, amount: float,
                         leuco_cyan_coupling: float = 1.0):
    """Negative-side leuco-cyan: reduce the cyan dye that never formed.
    Returns the input untouched when inactive."""
    loss = LEUCO_CYAN_LOSS * max(float(amount), 0.0) * max(float(leuco_cyan_coupling), 0.0)
    if loss <= 0.0:
        return density_cmy
    out = np.array(density_cmy, dtype=float, copy=True)
    out[..., 0] = out[..., 0] * (1.0 - min(loss, 0.9))
    return out
