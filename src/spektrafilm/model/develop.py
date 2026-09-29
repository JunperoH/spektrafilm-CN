from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from opt_einsum import contract
from spektrafilm.model.couplers import apply_density_correction_dir_couplers
from spektrafilm.model.density_curves import interpolate_exposure_to_density
from spektrafilm.model.grain import apply_grain, resolve_grain_model
from spektrafilm.profiles.io import DensityCurvesModel
from spektrafilm.runtime.params_schema import DirCouplersParams, GrainParams
from spektrafilm.utils.morph_curves import apply_print_curves_morph, PrintCurvesMorphParams

FloatArray: TypeAlias = NDArray[np.float64]
ProfileType: TypeAlias = Literal['negative', 'positive']

################################################################################
# Emulsion helpers

def compute_density_spectral(
    channel_density,
    density_cmy,
    base_density=None,
):
    density_spectral = contract('ijk, lk->ijl', density_cmy, np.asarray(channel_density))
    if base_density is not None:
        density_spectral += np.asarray(base_density)
    return density_spectral


# Keep black/white reference correction anchored to the stock density curves.
# Creative print-curve morphing belongs in develop_print_morph().
def develop_simple(
    log_raw,
    log_exposure,
    density_curves,
    gamma_factor=1.0,
):
    density_cmy = interpolate_exposure_to_density(log_raw, density_curves, log_exposure, gamma_factor)
    return density_cmy

def develop(
    log_raw: FloatArray,
    pixel_size_um: float,
    log_exposure: FloatArray,
    density_curves: FloatArray,
    density_curves_layers: FloatArray,
    dir_couplers: DirCouplersParams,
    grain: GrainParams,
    profile_type: ProfileType,
    gamma_factor: float = 1.0,
    bypass_grain: bool = False,
    use_fast_stats: bool = False,
) -> FloatArray:
    density_curves = np.asarray(density_curves)
    normalized_density_curves = density_curves - np.nanmin(density_curves, axis=0)

    # Phase 11B: 'rms' resolves HERE into production-equivalent params
    # (granularity target -> derived particle areas), so it rides the fused
    # GPU kernel below exactly like production. Other models pass through.
    grain = resolve_grain_model(grain, normalized_density_curves, density_curves_layers)

    # Phase 6D fused fast path: develop AND grain as ONE resident submit (one
    # upload of log_raw, one readback of the grained density). Eligibility is
    # the intersection of both islands' gates; None -> the separate-island /
    # CPU path below runs unchanged. Phase 11A: the fused kernel implements
    # the PRODUCTION grain model only - any other model choice routes through
    # apply_grain's model dispatch below.
    # (Polish 2026-07-21: monochrome B&W now RIDES the fused kernel — the
    # kernel draws independent per-channel noise, and channel 0 is replicated
    # after, which IS the single-emulsion semantic since the broadcast
    # channels carry identical densities and channel-0 params.)
    if not bypass_grain and getattr(grain, 'model', 'production') == 'production':
        try:
            from spektrafilm.gpu.develop_grain_resident import develop_grain_fused_dispatch
            _fused = develop_grain_fused_dispatch(
                log_raw, pixel_size_um, log_exposure, normalized_density_curves,
                dir_couplers, grain, density_curves_layers, profile_type,
                gamma_factor=gamma_factor, use_fast_stats=use_fast_stats,
            )
            if _fused is not None:
                if getattr(grain, 'monochrome', False):
                    _fused[:, :, 1] = _fused[:, :, 0]
                    _fused[:, :, 2] = _fused[:, :, 0]
                return _fused
        except ImportError:
            pass

    # GPU fast path: develop_simple + apply_density_correction_dir_couplers run as
    # ONE resident chain (single upload / single readback). Returns None -> the
    # unchanged CPU/per-stage path below. Grain is applied identically either way.
    density_cmy = None
    try:
        from spektrafilm.gpu.develop_resident import develop_deterministic_dispatch
        density_cmy = develop_deterministic_dispatch(
            log_raw,
            pixel_size_um,
            log_exposure,
            normalized_density_curves,
            dir_couplers,
            profile_type,
            gamma_factor=gamma_factor,
        )
    except ImportError:
        density_cmy = None

    if density_cmy is None:
        density_cmy = develop_simple(
            log_raw,
            log_exposure,
            normalized_density_curves,
            gamma_factor=gamma_factor,
        )
        density_cmy = apply_density_correction_dir_couplers(
            density_cmy,
            log_raw,
            pixel_size_um,
            log_exposure,
            normalized_density_curves,
            dir_couplers,
            profile_type,
            gamma_factor=gamma_factor,
        )
    return apply_grain(
        density_cmy,
        pixel_size_um,
        grain,
        normalized_density_curves,
        density_curves_layers,
        profile_type,
        bypass_grain=bypass_grain,
        use_fast_stats=use_fast_stats,
    )

def develop_print_morph(
    log_raw: FloatArray,
    log_exposure: FloatArray,
    density_curves_model: DensityCurvesModel,
    density_curves_morph: PrintCurvesMorphParams,
    profile_type: ProfileType = 'negative',
):
    density_curves_morphed = apply_print_curves_morph(
        log_exposure,
        density_curves_model,
        density_curves_morph,
        profile_type=profile_type,
    )
    # Phase 6D tail: the print-develop interp through the validated GPU curve
    # kernel (the same one the film develop chain uses); None -> CPU numba.
    try:
        from spektrafilm.gpu.density_curve import interpolate_exposure_to_density_dispatch
        _gpu = interpolate_exposure_to_density_dispatch(
            log_raw, density_curves_morphed, log_exposure, 1.0
        )
        if _gpu is not None:
            return _gpu
    except ImportError:
        pass
    density_cmy = interpolate_exposure_to_density(
        log_raw,
        density_curves_morphed,
        log_exposure,
        gamma_factor=1.0,
    )
    return density_cmy


################################################################################
# Phase 13A: Shadow Shape / Highlight Shape — paper density shaping

# Strength scale for a knob value of +-1. Chosen so the combined remap stays
# STRICTLY MONOTONE for any |shadow|,|highlight| <= 1 (max |d(delta)/du| =
# 0.12 * 6.75 = 0.81 < 1) — the print response can bend but never fold.
SHAPE_STRENGTH = 0.12


def apply_print_shape(density_cmy, density_curves,
                      shadow_shape=0.0, highlight_shape=0.0):
    """Endpoint-preserving remap of the developed PRINT density (Phase 13A).

    Applied after develop_print_morph, which is mathematically identical to
    warping the print response curve itself (a monotone density remap
    composes with the curve lookup) — so it IS shaping 'through the paper
    density response', composes with print push/pull (curve-side) and
    preflash (exposure-side), and preserves the paper D-min/D-max exactly
    (the shaping weights vanish at both endpoints).

    On a print, image SHADOWS are the DENSE end and image highlights the
    thin end. Signs follow the OFX manual: positive shadow_shape opens
    rendered shadows (reduces density near D-max); negative highlight_shape
    gives gentler highlight rolloff (adds density near D-min)."""
    if shadow_shape == 0.0 and highlight_shape == 0.0:
        return density_cmy
    curves = np.asarray(density_curves, dtype=float)
    d_min = np.nanmin(curves, axis=0)
    d_max = np.nanmax(curves, axis=0)
    span = np.maximum(d_max - d_min, 1e-6)
    u = np.clip((density_cmy - d_min) / span, 0.0, 1.0)
    s = float(np.clip(shadow_shape, -1.0, 1.0)) * SHAPE_STRENGTH
    h = float(np.clip(highlight_shape, -1.0, 1.0)) * SHAPE_STRENGTH
    # 6.75 = 27/4 normalizes each weight's peak to 1
    weight_shadow = 6.75 * u * u * (1.0 - u)          # peaked toward D-max
    weight_highlight = 6.75 * u * (1.0 - u) * (1.0 - u)   # peaked toward D-min
    return density_cmy + (-s * weight_shadow - h * weight_highlight) * span


# Some future work notes:
# Add print dye shift in nanometers for dye absorption peaks.
# Investigate how density curves change with development conditions.
# Add a gray card border to check white balance.

if __name__ == '__main__':
    pass

