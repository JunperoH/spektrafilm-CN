from __future__ import annotations

from spektrafilm_gui.state import GuiState
from spektrafilm.runtime.api import init_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec, OutputGamutCompressSpec
from spektrafilm.utils.morph_curves import FilmChemistryParams, PrintCurvesMorphParams

# Standard push/pull: development gamma multiplier per stop. Calibrated on the
# Kodak portra_800 push1/push2 datasheet family shipped with the app (measured
# mid-curve gamma ratios: push1 x1.22, push2 x1.39 ~= 1.18^2/stop).
PUSH_PULL_GAMMA_PER_STOP = 1.2


def build_params_from_state(state: GuiState) -> RuntimePhotoParams:
    # Phase 12B: 0.0 = profile default; a positive value picks the nearest
    # member of a B&W stock's development-time family (color stocks ignore it).
    development_time = getattr(state.simulation, 'film_development_time', 0.0)
    params = init_params(
        film_profile=state.simulation.film_stock,
        print_profile=state.simulation.print_paper,
        film_development_time=development_time if development_time > 0 else None,
    )

    _apply_special(params, state)
    _apply_glare(params, state)
    _apply_chemistry(params, state)
    _apply_push_pull(params, state)  # after _apply_special: composes into density_curve_gamma
    _apply_aging(params, state)
    _apply_scan_finishing(params, state)
    _apply_gamut_compression(params, state)
    _apply_camera(params, state)
    _apply_io(params, state)
    _apply_halation(params, state)
    _apply_grain(params, state)
    _apply_couplers(params, state)
    _apply_enlarger(params, state)
    _apply_scanner(params, state)
    _apply_settings(params, state)
    return params


def configure_export_params(
    params: RuntimePhotoParams,
    *,
    saving_color_space: str,
    saving_cctf_encoding: bool,
    full_precision: bool,
) -> RuntimePhotoParams:
    """Reconfigure params for a file-export render (the 'film scanner' path).

    The pipeline renders directly into the saving colour space and encoding, so
    nothing is clipped to a narrower intermediate gamut and no post-render
    RGB_to_RGB conversion is needed before writing. ``full_precision`` turns the
    enlarger/scanner LUTs OFF: the spectral integration runs exactly per pixel
    instead of through the LUT approximation. Fast-stats grain is kept ON either
    way: grain is a stochastic texture and the fast sampler is validated
    statistically equivalent to the exact scipy sampler (moments/PSD parity
    tests), while the exact sampler runs CPU-only and costs ~30 s extra at
    24 MP (measured 2026-07-06: export 49.4 s vs 21.5 s, develop stage 72%).
    Bit depth and latitude are unaffected by this flag (chosen at write time
    from the float buffer); only spectral fidelity and speed differ.

    This does not change WHAT the pipeline computes (parity-safe); it only
    selects the same approximation level used by the interactive scan vs the
    full-precision reference.
    """
    params.io.output_color_space = saving_color_space
    params.io.output_cctf_encoding = saving_cctf_encoding
    settings = params.settings
    settings.preview_mode = False
    if full_precision:
        settings.use_enlarger_lut = False
        settings.use_scanner_lut = False
    # NAT 2026-07-21 (v1.0.0): the Grain model dropdown drives EVERYTHING —
    # exports render whatever model is selected (the old 'Grain override'
    # forcing is gone; picking 'preview' for a fast export is a feature).
    return params


def _apply_special(params: RuntimePhotoParams, state: GuiState) -> None:
    def swap_channels(profile, new_cmy_order=(0,2,1)):
        profile.data.channel_density = profile.data.channel_density[:,new_cmy_order]
        return profile
    if state.special.film_channel_swap != (0, 1, 2):
        params.film = swap_channels(params.film, state.special.film_channel_swap)
    if state.special.print_channel_swap != (0, 1, 2):
        params.print = swap_channels(params.print, state.special.print_channel_swap)

    params.film_render.density_curve_gamma = state.special.film_gamma_factor
    # print_render.density_curve_gamma is gone since the 0.3.4 base; the Tune
    # knob now composes into the chemistry morph (see _apply_chemistry).


def _apply_glare(params: RuntimePhotoParams, state: GuiState) -> None:
    params.print_render.glare.active = state.glare.active
    params.print_render.glare.percent = state.glare.percent
    params.print_render.glare.roughness = state.glare.roughness
    params.print_render.glare.blur = state.glare.blur


def _apply_chemistry(params: RuntimePhotoParams, state: GuiState) -> None:
    c = state.chemistry
    # The 0.3.4 base develops the print through the curve morph only, and an
    # identity morph reproduces the unmorphed curves exactly (pinned by test).
    # The Tune tab's Print gamma factor therefore composes multiplicatively
    # into the morph's coupled gamma, replacing the removed
    # print_render.density_curve_gamma control; the morph activates whenever
    # either control is non-neutral. Clamped to the morph's >0 validation.
    print_gamma = max(float(state.special.print_gamma_factor), 0.05)
    params.print_render.density_curves_morph = PrintCurvesMorphParams(
        active=bool(c.active) or print_gamma != 1.0,
        gamma_factor=float(c.gamma_factor) * print_gamma,
        gamma_factor_fast=float(c.gamma_factor_fast),
        gamma_factor_slow=float(c.gamma_factor_slow),
        gamma_factor_red=float(c.gamma_factor_red),
        gamma_factor_green=float(c.gamma_factor_green),
        gamma_factor_blue=float(c.gamma_factor_blue),
        developer_exhaustion=float(c.developer_exhaustion),
    )


def _apply_push_pull(params: RuntimePhotoParams, state: GuiState) -> None:
    pp = state.push_pull
    if not bool(getattr(pp, 'active', True)):
        params.film_render.chemistry = FilmChemistryParams(active=False)
        return
    stops = float(pp.stops)
    stops_gamma = PUSH_PULL_GAMMA_PER_STOP ** stops
    if pp.mode == 'experimental':
        # Film-side s023 chemistry morph: baked into the film curves + grain
        # sublayers at pipeline init. Activates only when non-identity, so the
        # default state keeps the shipped profile arrays byte-identical.
        gamma_factor_fast = max(float(pp.gamma_factor_fast), 0.05)
        gamma_factor_slow = max(float(pp.gamma_factor_slow), 0.05)
        developer_exhaustion = min(max(float(pp.developer_exhaustion), 0.0), 1.0)
        non_identity = (
            stops != 0.0
            or gamma_factor_fast != 1.0
            or gamma_factor_slow != 1.0
            or developer_exhaustion > 0.0
        )
        params.film_render.chemistry = FilmChemistryParams(
            active=non_identity,
            gamma_factor=max(stops_gamma, 0.05),
            gamma_factor_fast=gamma_factor_fast,
            gamma_factor_slow=gamma_factor_slow,
            developer_exhaustion=developer_exhaustion,
        )
    else:
        # Standard: the stable timing/gamma model — stops map onto the plain
        # density-curve gamma exactly like the Tune tab's Film gamma factor
        # (multiplicative composition; _apply_special ran first).
        params.film_render.density_curve_gamma = (
            float(params.film_render.density_curve_gamma) * stops_gamma
        )
        params.film_render.chemistry = FilmChemistryParams(active=False)


def _apply_aging(params: RuntimePhotoParams, state: GuiState) -> None:
    from spektrafilm.utils.expiration import ExpirationParams

    a = getattr(state, 'aging', None)
    if a is None:
        params.film_render.expiration = ExpirationParams(active=False)
        return
    # Film expiration (emulsion aging): baked at pipeline init like chemistry.
    # The engine skips the bake when everything is neutral, so an 'active'
    # section with default knobs keeps the render bit-identical.
    params.film_render.expiration = ExpirationParams(
        active=bool(a.active),
        age_years=min(max(float(a.age_years), 0.0), 100.0),
        storage=str(a.storage),
        fog_rgb=tuple(float(v) for v in a.fog_rgb),
        gamma=min(max(float(a.gamma), 0.05), 4.0),
        gamma_rgb=tuple(min(max(float(v), 0.05), 4.0) for v in a.gamma_rgb),
        density_scale=min(max(float(a.density_scale), 0.05), 2.0),
        grain_gain=min(max(float(a.grain_gain), 0.05), 8.0),
        speed_loss_ev=min(max(float(a.speed_loss_ev), -6.0), 6.0),
    )


def _apply_scan_finishing(params: RuntimePhotoParams, state: GuiState) -> None:
    from spektrafilm.model.scan_finishing import ScanFinishingParams

    sf = state.scan_finishing
    params.scanner.finishing = ScanFinishingParams(
        active=bool(sf.active),
        crop=bool(sf.crop),
        crop_center=tuple(float(v) for v in sf.crop_center),
        crop_size=tuple(min(max(float(v), 0.005), 1.0) for v in sf.crop_size),
        skew_deg=min(max(float(sf.skew_deg), -45.0), 45.0),
        desqueeze=min(max(float(sf.desqueeze), -2.0), 2.0),
        exposure_ev=float(sf.exposure_ev),
        gain_red_ev=float(sf.gain_red_ev),
        gain_green_ev=float(sf.gain_green_ev),
        gain_blue_ev=float(sf.gain_blue_ev),
        color_separation=min(max(float(sf.color_separation), 0.0), 1.0),
        black_point=float(sf.black_point),
        white_point=float(sf.white_point),
        black_point_rgb=tuple(float(v) for v in sf.black_point_rgb),
        white_point_rgb=tuple(float(v) for v in sf.white_point_rgb),
        gamma=max(float(sf.gamma), 0.05),
        contrast=min(max(float(sf.contrast), -1.0), 1.0),
        toe=min(max(float(sf.toe), 0.0), 1.0),
        shoulder=min(max(float(sf.shoulder), 0.0), 1.0),
        micro_contrast=min(max(float(getattr(sf, 'micro_contrast', 0.0)), -1.0), 1.0),
        saturation=max(float(sf.saturation), 0.0),
        vibrance=max(float(sf.vibrance), 0.0),
        hue_saturation_ryg=tuple(max(float(v), 0.0) for v in sf.hue_saturation_ryg),
        hue_saturation_cbm=tuple(max(float(v), 0.0) for v in sf.hue_saturation_cbm),
        shadow_tint_hue=float(sf.shadow_tint_hue) % 360.0,
        shadow_tint_strength=min(max(float(sf.shadow_tint_strength), 0.0), 1.0),
        highlight_tint_hue=float(sf.highlight_tint_hue) % 360.0,
        highlight_tint_strength=min(max(float(sf.highlight_tint_strength), 0.0), 1.0),
        curve_luma=_curve_points(getattr(sf, 'curve_luma', None)),
        curve_r=_curve_points(getattr(sf, 'curve_r', None)),
        curve_g=_curve_points(getattr(sf, 'curve_g', None)),
        curve_b=_curve_points(getattr(sf, 'curve_b', None)),
    )


def _curve_points(points) -> tuple:
    # JSON round-trips hand back lists of lists; clamp into the unit square.
    if points is None:
        return ((0.0, 0.0), (1.0, 1.0))
    normalized = tuple(
        (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)) for x, y in points
    )
    return normalized if len(normalized) >= 2 else ((0.0, 0.0), (1.0, 1.0))


def _valid_knee(knee: tuple[float, float, float]) -> tuple[float, float, float]:
    # The frozen specs validate threshold in [0, 1) and limit/power > 0; clamp
    # spinbox values into the valid domain so the GUI can never raise here.
    threshold, limit, power = (float(value) for value in knee)
    return (min(max(threshold, 0.0), 0.99), max(limit, 0.01), max(power, 0.01))


def _apply_gamut_compression(params: RuntimePhotoParams, state: GuiState) -> None:
    igc = state.input_gamut_compress
    params.io.input_gamut_compress = InputGamutCompressSpec(
        active=bool(igc.active),
        algorithm=str(igc.algorithm),
        knee=_valid_knee(igc.knee),
    )
    ogc = state.output_gamut_compress
    params.io.output_gamut_compress = OutputGamutCompressSpec(
        algorithm=str(ogc.algorithm),
        knee=_valid_knee(ogc.knee),
    )


def _apply_camera(params: RuntimePhotoParams, state: GuiState) -> None:
    params.camera.lens_blur_um = state.simulation.camera_lens_blur_um
    params.camera.diffusion_filter.active = bool(state.simulation.camera_diffusion_filter_active)
    params.camera.diffusion_filter.filter_family = state.simulation.camera_diffusion_filter_family
    params.camera.diffusion_filter.strength = float(state.simulation.camera_diffusion_filter_strength)
    params.camera.diffusion_filter.spatial_scale = float(state.simulation.camera_diffusion_filter_spatial_scale)
    params.camera.diffusion_filter.halo_warmth = float(state.simulation.camera_diffusion_filter_halo_warmth)
    params.camera.diffusion_filter.core_intensity = float(state.simulation.camera_diffusion_filter_core_intensity)
    params.camera.diffusion_filter.core_size = float(state.simulation.camera_diffusion_filter_core_size)
    params.camera.diffusion_filter.halo_intensity = float(state.simulation.camera_diffusion_filter_halo_intensity)
    params.camera.diffusion_filter.halo_size = float(state.simulation.camera_diffusion_filter_halo_size)
    params.camera.diffusion_filter.bloom_intensity = float(state.simulation.camera_diffusion_filter_bloom_intensity)
    params.camera.diffusion_filter.bloom_size = float(state.simulation.camera_diffusion_filter_bloom_size)
    params.camera.exposure_compensation_ev = state.simulation.exposure_compensation_ev
    params.camera.auto_exposure = state.simulation.auto_exposure
    params.camera.auto_exposure_method = state.simulation.auto_exposure_method
    params.camera.film_format_mm = state.simulation.film_format_mm
    params.camera.filter_uv = state.input_image.filter_uv
    params.camera.filter_ir = state.input_image.filter_ir
    params.camera.color_filter = state.input_image.color_filter


def _apply_io(params: RuntimePhotoParams, state: GuiState) -> None:
    params.io.upscale_factor = state.input_image.upscale_factor
    params.io.crop = state.input_image.crop
    params.io.crop_center = state.input_image.crop_center
    params.io.crop_size = state.input_image.crop_size
    params.io.input_color_space = state.input_image.input_color_space
    params.io.input_cctf_decoding = state.input_image.apply_cctf_decoding
    params.io.output_color_space = state.simulation.output_color_space
    params.io.output_cctf_encoding = True
    params.io.scan_film = state.simulation.scan_film


def _apply_halation(params: RuntimePhotoParams, state: GuiState) -> None:
    h = state.halation
    p = params.film_render.halation
    p.active = h.active
    p.scatter_amount = h.scatter_amount
    p.scatter_spatial_scale = h.scatter_spatial_scale
    p.halation_amount = h.halation_amount
    p.halation_spatial_scale = h.halation_spatial_scale
    p.boost_ev = h.boost_ev
    p.protect_ev = h.protect_ev
    p.boost_range = h.boost_range
    p.scatter_core_um = tuple(h.scatter_core_um)
    p.scatter_tail_um = tuple(h.scatter_tail_um)
    p.scatter_tail_weight = tuple(float(value) / 100.0 for value in h.scatter_tail_weight)
    p.halation_strength = tuple(float(value) / 100.0 for value in h.halation_strength)
    p.halation_first_sigma_um = tuple(h.halation_first_sigma_um)
    p.halation_n_bounces = int(h.halation_n_bounces)
    p.halation_bounce_decay = float(h.halation_bounce_decay)
    p.halation_renormalize = bool(h.halation_renormalize)


def _apply_grain(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.grain.active = state.grain.active
    params.film_render.grain.model = state.grain.model
    params.film_render.grain.sublayers_active = state.grain.sublayers_active
    params.film_render.grain.particle_area_um2 = state.grain.particle_area_um2
    params.film_render.grain.particle_scale = state.grain.particle_scale
    params.film_render.grain.particle_scale_layers = state.grain.particle_scale_layers
    params.film_render.grain.density_min = state.grain.density_min
    params.film_render.grain.uniformity = state.grain.uniformity
    params.film_render.grain.blur = state.grain.blur
    params.film_render.grain.blur_dye_clouds_um = state.grain.blur_dye_clouds_um
    params.film_render.grain.micro_structure = state.grain.micro_structure
    # Phase 11B/11C model params (inert unless their model is selected)
    params.film_render.grain.rms_granularity = state.grain.rms_granularity
    params.film_render.grain.rms_strength = float(state.grain.rms_strength)
    params.film_render.grain.synthesis_size = float(state.grain.synthesis_size)
    params.film_render.grain.synthesis_amount = float(state.grain.synthesis_amount)
    params.film_render.grain.synthesis_sharpness = float(state.grain.synthesis_sharpness)
    params.film_render.grain.synthesis_quality = float(state.grain.synthesis_quality)
    params.film_render.grain.synthesis_samples = max(1, int(round(state.grain.synthesis_samples)))
    params.film_render.grain.synthesis_mean_radius_um = float(state.grain.synthesis_mean_radius_um)
    params.film_render.grain.synthesis_radius_stddev_ratio = float(state.grain.synthesis_radius_stddev_ratio)
    params.film_render.grain.synthesis_aperture_sigma_um = float(state.grain.synthesis_aperture_sigma_um)
    params.film_render.grain.synthesis_cell_size_ratio = float(state.grain.synthesis_cell_size_ratio)
    params.film_render.grain.synthesis_max_radius_quantile = float(state.grain.synthesis_max_radius_quantile)
    params.film_render.grain.synthesis_coverage_epsilon = float(state.grain.synthesis_coverage_epsilon)
    params.film_render.grain.synthesis_max_grains_per_cell = max(1, int(round(state.grain.synthesis_max_grains_per_cell)))
    params.film_render.grain.synthesis_radius_scale = state.grain.synthesis_radius_scale
    params.film_render.grain.synthesis_layered = bool(state.grain.synthesis_layered)
    params.film_render.grain.synthesis_layer_scale = state.grain.synthesis_layer_scale


def _apply_couplers(params: RuntimePhotoParams, state: GuiState) -> None:
    params.film_render.dir_couplers.active = state.couplers.active
    params.film_render.dir_couplers.amount = state.couplers.amount
    params.film_render.dir_couplers.inhibition_samelayer = state.couplers.inhibition_samelayer
    params.film_render.dir_couplers.inhibition_interlayer = state.couplers.inhibition_interlayer
    params.film_render.dir_couplers.gamma_samelayer_rgb = tuple(state.couplers.gamma_samelayer_rgb)
    params.film_render.dir_couplers.gamma_interlayer_r_to_gb = tuple(state.couplers.gamma_interlayer_r_to_gb)
    params.film_render.dir_couplers.gamma_interlayer_g_to_rb = tuple(state.couplers.gamma_interlayer_g_to_rb)
    params.film_render.dir_couplers.gamma_interlayer_b_to_rg = tuple(state.couplers.gamma_interlayer_b_to_rg)
    params.film_render.dir_couplers.diffusion_size_um = state.couplers.diffusion_size_um


def _apply_enlarger(params: RuntimePhotoParams, state: GuiState) -> None:
    params.enlarger.illuminant = state.simulation.print_illuminant
    params.enlarger.print_exposure = state.simulation.print_exposure
    params.enlarger.print_exposure_compensation = state.simulation.print_exposure_compensation
    params.enlarger.y_filter_shift = state.simulation.print_y_filter_shift
    params.enlarger.m_filter_shift = state.simulation.print_m_filter_shift
    params.enlarger.c_filter_shift = state.simulation.print_c_filter_shift
    # Phase 13A/13B
    params.print_render.shadow_shape = float(state.simulation.print_shadow_shape)
    params.print_render.highlight_shape = float(state.simulation.print_highlight_shape)
    params.print_render.bleach_bypass_amount = float(state.simulation.print_bleach_bypass)
    params.film_render.bleach_bypass_amount = float(state.simulation.negative_bleach_bypass)
    params.film_render.bleach_bypass_leuco_cyan = float(state.simulation.negative_leuco_cyan_coupling)
    params.enlarger.diffusion_filter.active = bool(state.simulation.diffusion_filter_active)
    params.enlarger.diffusion_filter.filter_family = state.simulation.diffusion_filter_family
    params.enlarger.diffusion_filter.strength = float(state.simulation.diffusion_filter_strength)
    params.enlarger.diffusion_filter.spatial_scale = float(state.simulation.diffusion_filter_spatial_scale)
    params.enlarger.diffusion_filter.halo_warmth = float(state.simulation.diffusion_filter_halo_warmth)
    params.enlarger.diffusion_filter.core_intensity = float(state.simulation.diffusion_filter_core_intensity)
    params.enlarger.diffusion_filter.core_size = float(state.simulation.diffusion_filter_core_size)
    params.enlarger.diffusion_filter.halo_intensity = float(state.simulation.diffusion_filter_halo_intensity)
    params.enlarger.diffusion_filter.halo_size = float(state.simulation.diffusion_filter_halo_size)
    params.enlarger.diffusion_filter.bloom_intensity = float(state.simulation.diffusion_filter_bloom_intensity)
    params.enlarger.diffusion_filter.bloom_size = float(state.simulation.diffusion_filter_bloom_size)
    # Preflash active toggle: off = zero exposure (the engine's no-op), so the
    # dialed-in values survive in the GUI state for A/B without re-entry.
    if bool(getattr(state.preflashing, 'active', True)):
        params.enlarger.preflash_exposure = state.preflashing.exposure
    else:
        params.enlarger.preflash_exposure = 0.0
    params.enlarger.preflash_y_filter_shift = state.preflashing.y_filter_shift
    params.enlarger.preflash_m_filter_shift = state.preflashing.m_filter_shift
    params.enlarger.preflash_c_filter_shift = state.preflashing.c_filter_shift


def _apply_scanner(params: RuntimePhotoParams, state: GuiState) -> None:
    params.scanner.lens_blur = state.simulation.scan_lens_blur
    params.scanner.white_correction = state.simulation.scan_white_correction
    params.scanner.white_level = state.simulation.scan_white_level
    params.scanner.black_correction = state.simulation.scan_black_correction
    params.scanner.black_level = state.simulation.scan_black_level
    params.scanner.unsharp_mask = state.simulation.scan_unsharp_mask


def _apply_settings(params: RuntimePhotoParams, state: GuiState) -> None:
    params.settings.rgb_to_raw_method = state.input_image.spectral_upsampling_method
    params.settings.apply_hanatos2025_adaptation_window = bool(state.input_image.apply_hanatos2025_adaptation_window)
    params.settings.apply_hanatos2025_adaptation_surface = bool(state.input_image.apply_hanatos2025_adaptation_surface)
    params.settings.spectral_gaussian_blur = float(state.input_image.spectral_gaussian_blur)
    params.settings.preview_max_size = state.display.preview_max_size
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    params.settings.use_fast_stats = True
