from __future__ import annotations

import copy
from dataclasses import fields
from functools import lru_cache

from spektrafilm.profiles.io import load_profile
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.io import read_neutral_print_filters


# Heavy fields shared by reference when cloning: the two Profiles carry the
# spectral arrays and density-curve models. They are swapped at selection
# time, never mutated during ordinary parameter edits, so a snapshot can
# share them safely.
_SHARED_PROFILE_FIELDS = ("film", "print")


def clone_runtime_params(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Cheap snapshot of runtime params for GUI undo / preview.

    Deep-copies the whole parameter subtree so edits to the clone never
    leak into the original, but shares the two ``Profile`` objects
    (``film``, ``print``) by reference. This keeps a snapshot O(params)
    instead of O(spectral data), the same property that made the GUI's
    old ``clone_gui_state`` cheap — without maintaining a mirror state.
    """
    kwargs = {
        f.name: getattr(params, f.name)
        if f.name in _SHARED_PROFILE_FIELDS
        else copy.deepcopy(getattr(params, f.name))
        for f in fields(params)
    }
    return RuntimePhotoParams(**kwargs)


@lru_cache(maxsize=1)
def _get_neutral_print_filters():
    try:
        return read_neutral_print_filters()
    except FileNotFoundError:
        return {}


def apply_database_neutral_print_filters(
    params: RuntimePhotoParams,
    *,
    database=None,
    warn_missing: bool = True,
) -> RuntimePhotoParams:
    if not params.settings.neutral_print_filters_from_database:
        return params

    # Phase 12A: a B&W print material has ONE sensitivity record, so CMY
    # filtration can only scale its overall exposure — which the print
    # exposure compensation normalizes away. 'Neutral filters' are therefore
    # meaningless for B&W: pin the neutral position to zero, silently.
    if getattr(params.print, 'is_bw', False):
        params.enlarger.c_filter_neutral = 0.0
        params.enlarger.m_filter_neutral = 0.0
        params.enlarger.y_filter_neutral = 0.0
        return params

    filters = _get_neutral_print_filters() if database is None else database
    per_film = (
        filters
        .get(params.print.info.stock, {})
        .get(params.enlarger.illuminant, {})
    )
    stock_filters = per_film.get(params.film.info.stock)
    if stock_filters is None:
        # Custom stocks (Phase 10B) inherit the base stock's neutral filters:
        # their camera-side spectra are the base's, so the base balance is
        # the right starting point.
        custom_base = getattr(params.film.info, 'custom_base', None)
        if custom_base:
            stock_filters = per_film.get(custom_base)
    if stock_filters is not None:
        c_filter, m_filter, y_filter = (float(value) for value in stock_filters)
        params.enlarger.c_filter_neutral = c_filter
        params.enlarger.m_filter_neutral = m_filter
        params.enlarger.y_filter_neutral = y_filter
    elif warn_missing:
        print(
            f"Warning: No neutral print filters found in database for print stock {params.print.info.stock} "
            f"with illuminant {params.enlarger.illuminant} and film stock {params.film.info.stock}. "
            "Using defaults."
        )
    return params


def digest_params(params: RuntimePhotoParams, apply_stocks_specifics=True) -> RuntimePhotoParams:
    """Digest the params to prepare for use in the runtime pipeline.
    In the pipeline params should be static and not be changed.
    params.settings and params.debug should contain all the switching logic for the digesting.
    """
    params = apply_database_neutral_print_filters(params)

    if params.settings.preview_mode:
        params.enlarger.lens_blur = 0.0
        params.film_render.dir_couplers.diffusion_size_um = 0.0
        params.film_render.grain.active = False
        params.film_render.grain.particle_area_um2 = 0.0
        params.film_render.grain.blur = 0.0
        # scatter/halation kernel sigmas are preserved in preview mode
        params.print_render.glare.blur = 0.0
        params.camera.lens_blur_um = 0.0
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)
    
    if apply_stocks_specifics:
        params = _apply_film_specifics(params)
        params = _apply_print_specifics(params)
    
    # debug switches
    if params.debug.lut_mode:
        # LUT-sampling regime: force the pipeline into a deterministic
        # per-pixel transform. Enabling lut_mode promotes spatial and
        # stochastic deactivation and disables image-aware adjustments.
        params.debug.deactivate_spatial_effects = True
        params.debug.deactivate_stochastic_effects = True
        # exposure control
        params.camera.auto_exposure = False
        params.camera.exposure_compensation_ev = 0.0
        params.enlarger.print_exposure_compensation = False
        params.enlarger.print_exposure = 1.0
        # Highlight boost normalizes by the image-wide max (np.max(x)), so it is
        # an image-global transform — the same input value maps to different
        # outputs depending on the rest of the frame. That cannot be represented
        # by a static 3D LUT (a bake would freeze in the cube grid's max), so it
        # must be off in lut_mode, exactly like auto_exposure above.
        params.film_render.halation.boost_ev = 0.0
        params.scanner.white_correction = False
        params.scanner.black_correction = False
        params.scanner.unsharp_mask = (0.0, 0.0)

    if params.debug.deactivate_spatial_effects:
        # Halation is fully spatial (scatter + back-reflection blurs); kill it
        # at the active flag as well as zeroing the kernel sigmas, so it stays
        # a no-op even if a future sigma-independent term is added inside
        # apply_halation_um.
        params.film_render.halation.active = False
        params.film_render.halation.scatter_core_um = (0.0, 0.0, 0.0)
        params.film_render.halation.scatter_tail_um = (0.0, 0.0, 0.0)
        params.film_render.halation.halation_first_sigma_um = (0.0, 0.0, 0.0)
        params.film_render.dir_couplers.diffusion_size_um = 0
        params.film_render.grain.blur = 0.0
        params.film_render.grain.blur_dye_clouds_um = 0.0
        params.print_render.glare.blur = 0
        params.camera.lens_blur_um = 0.0
        params.enlarger.lens_blur = 0.0
        params.enlarger.diffusion_filter.active = False
        params.camera.diffusion_filter.active = False
        params.scanner.lens_blur = 0.0
        params.scanner.unsharp_mask = (0.0, 0.0)

    if params.debug.deactivate_stochastic_effects:
        params.film_render.grain.active = False
        params.print_render.glare.active = False
        
    return params


def init_params(
    film_profile: str = "kodak_portra_400",
    print_profile: str = "kodak_portra_endura",
    film_development_time: float | None = None,
) -> RuntimePhotoParams:
    """Simple helper to build a RuntimePhotoParams with just film and print profiles specified.
    Build a runtime parameter object.
    It needs to be digested with digest_params before being used in the runtime pipeline.

    Phase 12B: ``film_development_time`` (minutes) selects the nearest member
    of a B&W profile's development-time family; None = the profile default
    (floor-middle). Ignored by profiles without a family (all color stocks)."""

    params = RuntimePhotoParams(
        film=load_profile(film_profile, development_time=film_development_time),
        print=load_profile(print_profile),
    )
    return params

def _first_triplet(obj, name):
    """Channel-0-broadcast for per-channel halation tuples; scalar fields
    (older schemas) pass through unchanged."""
    value = getattr(obj, name)
    try:
        return (value[0],) * 3
    except TypeError:
        return value


def _apply_film_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply film specific settings to the params."""
    # film overrides
    # define here all the specifics to stocks that should be applied in params.film_render
    if params.film.is_positive:
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.12, 0.08, 0.06)
        params.film_render.dir_couplers.gamma_interlayer_r_to_gb = (0.12, 0.06)
        params.film_render.dir_couplers.gamma_interlayer_g_to_rb = (0.08, 0.06)
        params.film_render.dir_couplers.gamma_interlayer_b_to_rg = (0.06, 0.06) # just eyeballed, optimize!
        
    if params.film.is_negative:
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.336, 0.319, 0.273)
        params.film_render.dir_couplers.gamma_interlayer_r_to_gb = (0.353, 0.302)
        params.film_render.dir_couplers.gamma_interlayer_g_to_rb = (0.154, 0.353)
        params.film_render.dir_couplers.gamma_interlayer_b_to_rg = (0.168, 0.226) # just eyeballed, optimize!

    # Phase 12A B&W (line-for-line from spektrafilm.rs pipeline.rs:68-93,
    # itself mirroring upstream's n_channels==1 semantics): one emulsion —
    # the DIR matrix is 1x1 self-inhibition with no interlayer terms, grain
    # is one shared neutral noise field, and every per-channel tuple is
    # forced to channel 0 so the broadcast 3-channel engine stays
    # numerically identical to the upstream single-channel run. Monochrome
    # is derived from the film, never user-set — cleared for color so a
    # stray params file can't correlate the color channels' noise.
    params.film_render.grain.monochrome = params.film.is_bw
    # NOTE the channel-0 forcing itself runs at the END of this function
    # (see _force_bw_channel0) — the halation preset and per-stock overrides
    # below would overwrite it otherwise.

# skin - forest colors - cc blue optimization
# 0.42 alpha 
# [[0.336423, 0.353654, 0.302163],
#  [0.154796, 0.319218, 0.353513],
#  [0.168943, 0.226796, 0.273107]]
# 0.5 alpha high saturation
# [[0.341559, 0.355603, 0.305212],
#  [0.154542, 0.324590, 0.358254],
#  [0.171108, 0.225493, 0.273080]]
# reference matrix cc loss
# [[0.344306, 0.324515, 0.305428],
#  [0.142589, 0.345450, 0.360369],
#  [0.161963, 0.241596, 0.263818]]
# gamma 1.2 sigma 1.55 loss 1.37
#  [[0.53402457 0.43368797 0.23228746]
#  [0.37136105 0.4572779  0.37136105]
#  [0.23228746 0.43368797 0.53402457]]
# gamma 1.0 sigma 1.3
# [[0.48777655 0.36285359 0.14936985]
#  [0.29901809 0.40196381 0.29901809]
#  [0.14936985 0.36285359 0.48777655]]

    _apply_halation_preset(params)

    # stock specifics overrides
    if params.film.info.stock == "fujifilm_velvia_100":
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.108, 0.072, 0.054)
        params.film_render.dir_couplers.gamma_interlayer_r_to_gb = (0.108, 0.054)
        params.film_render.dir_couplers.gamma_interlayer_g_to_rb = (0.072, 0.054)
        params.film_render.dir_couplers.gamma_interlayer_b_to_rg = (0.054, 0.054)
    if params.film.info.stock == "fujifilm_provia_100f":
        params.film_render.dir_couplers.gamma_samelayer_rgb = (0.156, 0.104, 0.078)
        params.film_render.dir_couplers.gamma_interlayer_r_to_gb = (0.156, 0.078)
        params.film_render.dir_couplers.gamma_interlayer_g_to_rb = (0.104, 0.078)
        params.film_render.dir_couplers.gamma_interlayer_b_to_rg = (0.078, 0.078)
        
    # if params.film.info.stock == "kodak_portra_400":
    #     params.film_render.halation.scatter_core_um = (3.5, 2.2, 1.9)

    # Phase 10B custom stocks: a baked recipe can carry partial look
    # overrides (grain / halation / dir_couplers deltas). They apply LAST so
    # the recipe wins over the type/use/antihalation presets above, exactly
    # like the per-stock overrides do.
    _apply_custom_look_overrides(params)
    # Phase 12A: the B&W channel-0 forcing runs LAST of all — whatever the
    # presets/overrides above set, one emulsion means one set of params.
    _force_bw_channel0(params)
    return params


def _force_bw_channel0(params: RuntimePhotoParams) -> None:
    if not params.film.is_bw:
        return
    dir_couplers = params.film_render.dir_couplers
    same = dir_couplers.gamma_samelayer_rgb[0]
    dir_couplers.gamma_samelayer_rgb = (same, same, same)
    dir_couplers.gamma_interlayer_r_to_gb = (0.0, 0.0)
    dir_couplers.gamma_interlayer_g_to_rb = (0.0, 0.0)
    dir_couplers.gamma_interlayer_b_to_rg = (0.0, 0.0)
    grain = params.film_render.grain
    grain.particle_scale = (grain.particle_scale[0],) * 3
    grain.density_min = (grain.density_min[0],) * 3
    grain.uniformity = (grain.uniformity[0],) * 3
    halation = params.film_render.halation
    halation.halation_strength = (halation.halation_strength[0],) * 3
    halation.scatter_core_um = _first_triplet(halation, 'scatter_core_um')
    halation.scatter_tail_um = _first_triplet(halation, 'scatter_tail_um')
    halation.scatter_tail_weight = _first_triplet(halation, 'scatter_tail_weight')
    halation.halation_first_sigma_um = _first_triplet(halation, 'halation_first_sigma_um')


def _apply_custom_look_overrides(params: RuntimePhotoParams) -> None:
    custom_look = getattr(params.film.info, 'custom_look', None)
    if not custom_look:
        return
    for target_name, overrides in custom_look.items():
        target = getattr(params.film_render, target_name, None)
        if target is None or not isinstance(overrides, dict):
            continue
        for field_name, value in overrides.items():
            if not hasattr(target, field_name):
                continue
            current = getattr(target, field_name)
            if isinstance(current, tuple) and isinstance(value, (list, tuple)):
                setattr(target, field_name, tuple(value))
            elif isinstance(current, bool):
                setattr(target, field_name, bool(value))
            elif isinstance(current, (int, float)) and isinstance(value, (int, float)):
                setattr(target, field_name, type(current)(value))
            else:
                setattr(target, field_name, value)


# Halation low-level presets keyed by (use, antihalation). These set the
# physical baselines from the private halation implementation notes §5-§6.1;
# the user-facing knobs (scatter_amount, scatter_spatial_scale,
# halation_amount, halation_spatial_scale) remain at 1.0 and let the user
# push the effect stronger or weaker without editing the low-level
# parameters.
#
# sigma_h is set by the base material:
#   still -> triacetate, 120-140 um thick -> sigma_h ~= 65 um
#   cine  -> PET,        95-125 um thick -> sigma_h ~= 50 um
# halation_strength is set by the antihalation layer, from §5 ranges:
#   strong -> Vision3, modern colour neg: a1^R ~ 0.005-0.02
#   weak   -> older / mismatched AH:       a1^R ~ 0.02-0.08
#   no     -> rem-jet removed / redscale:  a1^R ~ 0.08-0.25
# Strength values below are a1 midpoints * 1/(1-rho) with rho=0.5 (~2x a1).
_HALATION_PRESETS: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {
    ('still', 'strong'): {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.015, 0.005, 0.0)},
    ('still', 'weak'):   {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.08,  0.02,  0.0)},
    ('still', 'no'):     {'sigma_h': (65.0, 65.0, 65.0), 'strength': (0.30,  0.10,  0.015)},
    ('cine',  'strong'): {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.015, 0.005, 0.0)},
    ('cine',  'weak'):   {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.08,  0.02,  0.0)},
    ('cine',  'no'):     {'sigma_h': (50.0, 50.0, 50.0), 'strength': (0.30,  0.10,  0.015)},
}


def _apply_halation_preset(params: RuntimePhotoParams) -> None:
    """Seed low-level halation parameters from the profile's use/antihalation tags."""
    if not params.film.is_film:
        return
    info = params.film.info
    preset = _HALATION_PRESETS.get((info.use, info.antihalation))
    if preset is None:
        return
    params.film_render.halation.halation_first_sigma_um = preset['sigma_h']
    params.film_render.halation.halation_strength = preset['strength']

def _apply_print_specifics(params: RuntimePhotoParams) -> RuntimePhotoParams:
    """Apply print specific settings to the params."""
    # define here all the specifics to stocks that should be applied in params.print_render
    return params


__all__ = [
    "apply_database_neutral_print_filters",
    "clone_runtime_params",
    "digest_params",
    "init_params",
]
