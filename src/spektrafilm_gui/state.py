from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import TypeVar

from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime.api import digest_params, init_params
from spektrafilm.runtime.params_schema import RuntimePhotoParams
from spektrafilm_gui.options import ExportFormats


StateSection = TypeVar('StateSection')


@dataclass(slots=True)
class InputImageState:
    upscale_factor: float
    crop: bool
    crop_center: tuple[float, float]
    crop_size: tuple[float, float]
    input_color_space: str
    apply_cctf_decoding: bool
    spectral_upsampling_method: str
    apply_hanatos2025_adaptation_window: bool
    apply_hanatos2025_adaptation_surface: bool
    spectral_gaussian_blur: float
    filter_uv: tuple[float, float, float]
    filter_ir: tuple[float, float, float]
    color_filter: str


@dataclass(slots=True)
class LoadRawState:
    white_balance: str
    temperature: float
    tint: float
    lens_correction: bool


@dataclass(slots=True)
class LoadRGBState:
    white_balance: str
    temperature: float
    tint: float
    recover_16bit: bool = False


@dataclass(slots=True)
class GrainState:
    active: bool
    sublayers_active: bool
    # Phase 11A: which grain model renders. A USER choice - deliberately NOT
    # in the profile-sync GRAIN list, so a stock switch never resets it.
    # (Row order = NAT's: model sits under 'sublayers active'.)
    model: str
    particle_area_um2: float
    particle_scale: tuple[float, float, float]
    particle_scale_layers: tuple[float, float, float]
    density_min: tuple[float, float, float]
    uniformity: tuple[float, float, float]
    blur: float
    blur_dye_clouds_um: float
    micro_structure: tuple[float, float]
    # Phase 11B RMS model (used when model='rms')
    rms_granularity: tuple[float, float, float]
    rms_strength: float
    # Phase 11C Grain Synthesis: simplified controls (used when
    # model='synthesis')...
    synthesis_size: float
    synthesis_amount: float
    synthesis_sharpness: float
    synthesis_quality: float
    # ...and the advanced Boolean-model group (hidden here, surfaced by the
    # collapsed Grain Synthesis section).
    synthesis_samples: float
    synthesis_mean_radius_um: float
    synthesis_radius_stddev_ratio: float
    synthesis_aperture_sigma_um: float
    synthesis_cell_size_ratio: float
    synthesis_max_radius_quantile: float
    synthesis_coverage_epsilon: float
    synthesis_max_grains_per_cell: float
    synthesis_radius_scale: tuple[float, float, float]
    synthesis_layered: bool
    synthesis_layer_scale: tuple[float, float, float]


@dataclass(slots=True)
class PreflashingState:
    active: bool
    exposure: float
    y_filter_shift: float
    m_filter_shift: float
    c_filter_shift: float   # NAT 2026-07-22: preflash C trim (default 0)


@dataclass(slots=True)
class HalationState:
    active: bool
    # high-level knobs (intended for the simplified GUI)
    scatter_amount: float
    scatter_spatial_scale: float
    halation_amount: float
    halation_spatial_scale: float
    # highlight boost
    boost_ev: float
    protect_ev: float
    boost_range: float
    # scatter (low-level, developer GUI)
    scatter_core_um: tuple[float, float, float]
    scatter_tail_um: tuple[float, float, float]
    scatter_tail_weight: tuple[float, float, float]  # stored 0-100 percent
    # halation (low-level, developer GUI)
    halation_strength: tuple[float, float, float]  # stored 0-100 percent
    halation_first_sigma_um: tuple[float, float, float]
    halation_n_bounces: int
    halation_bounce_decay: float
    halation_renormalize: bool


@dataclass(slots=True)
class CouplersState:
    active: bool
    amount: float
    inhibition_samelayer: float
    inhibition_interlayer: float
    gamma_samelayer_rgb: tuple[float, float, float]
    gamma_interlayer_r_to_gb: tuple[float, float]
    gamma_interlayer_g_to_rb: tuple[float, float]
    gamma_interlayer_b_to_rg: tuple[float, float]
    diffusion_size_um: float


@dataclass(slots=True)
class GlareState:
    active: bool
    percent: float
    roughness: float
    blur: float


@dataclass(slots=True)
class ChemistryState:
    # Mirrors spektrafilm.utils.morph_curves.PrintCurvesMorphParams (the s023
    # print density-curve morph). Inactive keeps the classic develop path and
    # the Print gamma factor control untouched.
    active: bool
    gamma_factor: float
    gamma_factor_fast: float
    gamma_factor_slow: float
    gamma_factor_red: float
    gamma_factor_green: float
    gamma_factor_blue: float
    developer_exhaustion: float


@dataclass(slots=True)
class ScanFinishingState:
    # Mirrors spektrafilm.model.scan_finishing.ScanFinishingParams (SCAN tab):
    # display-referred scanner processing on the digitized output. All
    # defaults are identity (bit-identical render).
    active: bool
    # scan-time geometry (cutout of the rendered film)
    crop: bool
    crop_center: tuple[float, float]
    crop_size: tuple[float, float]
    skew_deg: float
    desqueeze: float
    exposure_ev: float
    gain_red_ev: float
    gain_green_ev: float
    gain_blue_ev: float
    color_separation: float
    black_point: float
    white_point: float
    black_point_rgb: tuple[float, float, float]
    white_point_rgb: tuple[float, float, float]
    gamma: float
    contrast: float
    toe: float
    shoulder: float
    micro_contrast: float
    saturation: float
    vibrance: float
    hue_saturation_ryg: tuple[float, float, float]
    hue_saturation_cbm: tuple[float, float, float]
    shadow_tint_hue: float
    shadow_tint_strength: float
    highlight_tint_hue: float
    highlight_tint_strength: float
    # Phase 10A point curve: (x, y) control points in [0,1]^2 per channel;
    # ((0,0),(1,1)) is identity. Kept LAST with defaults so pre-10A saved
    # states merge-load cleanly.
    curve_luma: tuple = ((0.0, 0.0), (1.0, 1.0))
    curve_r: tuple = ((0.0, 0.0), (1.0, 1.0))
    curve_g: tuple = ((0.0, 0.0), (1.0, 1.0))
    curve_b: tuple = ((0.0, 0.0), (1.0, 1.0))


@dataclass(slots=True)
class PushPullState:
    # Film push/pull development, FILM tab. 'standard' maps stops onto the
    # plain density-curve gamma (1.2 ** stops, calibrated on the Kodak
    # portra_800 push1/push2 datasheet family), composed with the Tune tab's
    # Film gamma factor. 'experimental' drives the film-side s023 chemistry
    # morph instead (per-sublayer coupled gamma + developer exhaustion, baked
    # into curves + grain sublayers before develop).
    active: bool
    mode: str
    stops: float
    gamma_factor_fast: float
    gamma_factor_slow: float
    developer_exhaustion: float


@dataclass(slots=True)
class AgingState:
    # Film expiration (emulsion aging), FILM tab — the sibling of push/pull.
    # Age + Storage drive the generic expired-negative macro; the manual knobs
    # below bias it per stock (they stack: fog adds, gammas/scales multiply).
    # Maps onto film_render.expiration (see params_mapper._apply_aging).
    active: bool
    age_years: float
    storage: str
    fog_rgb: tuple[float, float, float]
    gamma: float
    gamma_rgb: tuple[float, float, float]
    density_scale: float
    grain_gain: float
    speed_loss_ev: float


@dataclass(slots=True)
class InputGamutCompressState:
    # Mirrors spektrafilm.utils.gamut_compression.InputGamutCompressSpec.
    active: bool
    algorithm: str
    knee: tuple[float, float, float]


@dataclass(slots=True)
class OutputGamutCompressState:
    # Mirrors spektrafilm.utils.gamut_compression.OutputGamutCompressSpec
    # (algorithm + knee only; lightness_compression keeps the engine default).
    algorithm: str
    knee: tuple[float, float, float]


@dataclass(slots=True)
class SpecialState:
    film_channel_swap: tuple[int, int, int]
    film_gamma_factor: float
    print_channel_swap: tuple[int, int, int]
    print_gamma_factor: float


@dataclass(slots=True)
class SimulationState:
    film_stock: str
    film_format_mm: float
    camera_lens_blur_um: float
    camera_diffusion_filter_active: bool
    camera_diffusion_filter_family: str
    camera_diffusion_filter_strength: float
    camera_diffusion_filter_spatial_scale: float
    camera_diffusion_filter_halo_warmth: float
    camera_diffusion_filter_core_intensity: float
    camera_diffusion_filter_core_size: float
    camera_diffusion_filter_halo_intensity: float
    camera_diffusion_filter_halo_size: float
    camera_diffusion_filter_bloom_intensity: float
    camera_diffusion_filter_bloom_size: float
    exposure_compensation_ev: float
    auto_exposure: bool
    auto_exposure_method: str
    print_paper: str
    print_illuminant: str
    print_exposure: float
    print_exposure_compensation: bool
    print_y_filter_shift: float
    print_m_filter_shift: float
    print_c_filter_shift: float
    # Phase 12B: minutes; 0 = profile default. Only B&W stocks carry a
    # development-time family (nearest member selected).
    film_development_time: float
    # Phase 13A: paper shaping (PRINT tab wrapper section)
    print_shadow_shape: float
    print_highlight_shape: float
    # Phase 13B: bleach bypass (modeled extension; FILM + PRINT wrappers)
    negative_bleach_bypass: float
    negative_leuco_cyan_coupling: float
    print_bleach_bypass: float
    diffusion_filter_active: bool
    diffusion_filter_family: str
    diffusion_filter_strength: float
    diffusion_filter_spatial_scale: float
    diffusion_filter_halo_warmth: float
    diffusion_filter_core_intensity: float
    diffusion_filter_core_size: float
    diffusion_filter_halo_intensity: float
    diffusion_filter_halo_size: float
    diffusion_filter_bloom_intensity: float
    diffusion_filter_bloom_size: float
    scan_lens_blur: float
    scan_white_correction: bool
    scan_white_level: float
    scan_black_correction: bool
    scan_black_level: float
    scan_unsharp_mask: tuple[float, float]
    output_color_space: str
    saving_color_space: str
    saving_cctf_encoding: bool
    export_format: str
    export_full_precision: bool
    auto_preview: bool
    scan_film: bool


@dataclass(slots=True)
class DisplayState:
    # gray_18_canvas removed 2026-07-10: the canvas background is the nav-bar
    # swatch row now (persisted via QSettings), old saved keys merge-ignore.
    use_display_transform: bool
    white_padding: float
    preview_max_size: int
    output_interpolation: str = 'spline36'


@dataclass(slots=True)
class PaperState:
    # Paper-texture finish (ADVANCED tab, NAT 2026-07-26 v1.0.2): display-
    # referred paper-scan textures blended onto the RENDERED output. NOT film
    # science — runs after the pipeline, so all-default = bit-identical.
    # ``textures`` is an ordered tuple of (path, fusion_mode). Kept a plain
    # tuple (like the finishing-curve points) so it clones/persists cleanly and
    # old saved states merge-load with paper inactive.
    active: bool = False
    padding_burn: bool = False
    textures: tuple = ()


@dataclass(slots=True)
class GuiState:
    input_image: InputImageState
    load_raw: LoadRawState
    load_rgb: LoadRGBState
    grain: GrainState
    preflashing: PreflashingState
    halation: HalationState
    couplers: CouplersState
    glare: GlareState
    chemistry: ChemistryState
    push_pull: PushPullState
    aging: AgingState
    scan_finishing: ScanFinishingState
    input_gamut_compress: InputGamutCompressState
    output_gamut_compress: OutputGamutCompressState
    special: SpecialState
    simulation: SimulationState
    display: DisplayState
    paper: PaperState = field(default_factory=PaperState)


def clone_state_section(section: StateSection) -> StateSection:
    if not is_dataclass(section):
        raise TypeError('Expected a dataclass instance to clone.')
    return replace(section)


def clone_gui_state(state: GuiState) -> GuiState:
    return GuiState(
        input_image=clone_state_section(state.input_image),
        load_raw=clone_state_section(state.load_raw),
        load_rgb=clone_state_section(state.load_rgb),
        grain=clone_state_section(state.grain),
        preflashing=clone_state_section(state.preflashing),
        halation=clone_state_section(state.halation),
        couplers=clone_state_section(state.couplers),
        glare=clone_state_section(state.glare),
        chemistry=clone_state_section(state.chemistry),
        push_pull=clone_state_section(state.push_pull),
        aging=clone_state_section(state.aging),
        scan_finishing=clone_state_section(state.scan_finishing),
        input_gamut_compress=clone_state_section(state.input_gamut_compress),
        output_gamut_compress=clone_state_section(state.output_gamut_compress),
        special=clone_state_section(state.special),
        simulation=clone_state_section(state.simulation),
        display=clone_state_section(state.display),
        paper=clone_state_section(state.paper),
    )


def _expiration(params: RuntimePhotoParams):
    return getattr(params.film_render, 'expiration', None)


def _curve_points_state(points) -> tuple:
    """Finishing-curve points as a tuple of (x, y) float tuples; identity for
    missing/degenerate values (pre-10A params objects have no curve fields)."""
    if points is None:
        return ((0.0, 0.0), (1.0, 1.0))
    normalized = tuple((float(x), float(y)) for x, y in points)
    return normalized if len(normalized) >= 2 else ((0.0, 0.0), (1.0, 1.0))


def gui_state_from_params(
    params: RuntimePhotoParams,
    *,
    film_stock: str,
    print_paper: str,
) -> GuiState:
    return GuiState(
        input_image=InputImageState(
            upscale_factor=params.io.upscale_factor,
            crop=params.io.crop,
            crop_center=tuple(params.io.crop_center),
            crop_size=tuple(params.io.crop_size),
            input_color_space=params.io.input_color_space,
            apply_cctf_decoding=params.io.input_cctf_decoding,
            spectral_upsampling_method=params.settings.rgb_to_raw_method,
            apply_hanatos2025_adaptation_window=params.settings.apply_hanatos2025_adaptation_window,
            apply_hanatos2025_adaptation_surface=params.settings.apply_hanatos2025_adaptation_surface,
            spectral_gaussian_blur=params.settings.spectral_gaussian_blur,
            filter_uv=tuple(params.camera.filter_uv),
            filter_ir=tuple(params.camera.filter_ir),
            color_filter=params.camera.color_filter,
        ),
        load_raw=LoadRawState(
            white_balance='as_shot',
            temperature=5500.0,
            tint=1.0,
            lens_correction=False,
        ),
        load_rgb=LoadRGBState(
            white_balance='as_shot',
            temperature=5500.0,
            tint=1.0,
            recover_16bit=False,
        ),
        grain=GrainState(
            active=params.film_render.grain.active,
            model=getattr(params.film_render.grain, 'model', 'production'),
            sublayers_active=params.film_render.grain.sublayers_active,
            particle_area_um2=params.film_render.grain.particle_area_um2,
            particle_scale=tuple(params.film_render.grain.particle_scale),
            particle_scale_layers=tuple(params.film_render.grain.particle_scale_layers),
            density_min=tuple(params.film_render.grain.density_min),
            uniformity=tuple(params.film_render.grain.uniformity),
            blur=params.film_render.grain.blur,
            blur_dye_clouds_um=params.film_render.grain.blur_dye_clouds_um,
            micro_structure=tuple(params.film_render.grain.micro_structure),
            # 11B/11C fields: getattr-tolerant for pre-11B params objects
            rms_granularity=tuple(getattr(
                params.film_render.grain, 'rms_granularity', (5.0, 5.0, 5.0))),
            rms_strength=float(getattr(
                params.film_render.grain, 'rms_strength', 1.0)),
            synthesis_size=float(getattr(
                params.film_render.grain, 'synthesis_size', 1.0)),
            synthesis_amount=float(getattr(
                params.film_render.grain, 'synthesis_amount', 1.0)),
            synthesis_sharpness=float(getattr(
                params.film_render.grain, 'synthesis_sharpness', 1.0)),
            synthesis_quality=float(getattr(
                params.film_render.grain, 'synthesis_quality', 1.0)),
            synthesis_samples=float(getattr(
                params.film_render.grain, 'synthesis_samples', 16)),
            synthesis_mean_radius_um=float(getattr(
                params.film_render.grain, 'synthesis_mean_radius_um', 0.6)),
            synthesis_radius_stddev_ratio=float(getattr(
                params.film_render.grain, 'synthesis_radius_stddev_ratio', 0.4)),
            synthesis_aperture_sigma_um=float(getattr(
                params.film_render.grain, 'synthesis_aperture_sigma_um', 0.0)),
            synthesis_cell_size_ratio=float(getattr(
                params.film_render.grain, 'synthesis_cell_size_ratio', 4.0)),
            synthesis_max_radius_quantile=float(getattr(
                params.film_render.grain, 'synthesis_max_radius_quantile', 0.999)),
            synthesis_coverage_epsilon=float(getattr(
                params.film_render.grain, 'synthesis_coverage_epsilon', 1e-4)),
            synthesis_max_grains_per_cell=float(getattr(
                params.film_render.grain, 'synthesis_max_grains_per_cell', 32)),
            synthesis_radius_scale=tuple(getattr(
                params.film_render.grain, 'synthesis_radius_scale', (1.0, 1.0, 1.0))),
            synthesis_layered=bool(getattr(
                params.film_render.grain, 'synthesis_layered', False)),
            synthesis_layer_scale=tuple(getattr(
                params.film_render.grain, 'synthesis_layer_scale', (2.0, 1.0, 0.5))),
        ),
        preflashing=PreflashingState(
            active=True,
            exposure=params.enlarger.preflash_exposure,
            y_filter_shift=params.enlarger.preflash_y_filter_shift,
            m_filter_shift=params.enlarger.preflash_m_filter_shift,
            c_filter_shift=getattr(params.enlarger, 'preflash_c_filter_shift', 0.0),
        ),
        halation=HalationState(
            active=params.film_render.halation.active,
            scatter_amount=params.film_render.halation.scatter_amount,
            scatter_spatial_scale=params.film_render.halation.scatter_spatial_scale,
            halation_amount=params.film_render.halation.halation_amount,
            halation_spatial_scale=params.film_render.halation.halation_spatial_scale,
            boost_ev=params.film_render.halation.boost_ev,
            protect_ev=params.film_render.halation.protect_ev,
            boost_range=params.film_render.halation.boost_range,
            scatter_core_um=tuple(params.film_render.halation.scatter_core_um),
            scatter_tail_um=tuple(params.film_render.halation.scatter_tail_um),
            scatter_tail_weight=tuple(value * 100.0 for value in params.film_render.halation.scatter_tail_weight),
            halation_strength=tuple(value * 100.0 for value in params.film_render.halation.halation_strength),
            halation_first_sigma_um=tuple(params.film_render.halation.halation_first_sigma_um),
            halation_n_bounces=params.film_render.halation.halation_n_bounces,
            halation_bounce_decay=params.film_render.halation.halation_bounce_decay,
            halation_renormalize=params.film_render.halation.halation_renormalize,
        ),
        couplers=CouplersState(
            active=params.film_render.dir_couplers.active,
            amount=params.film_render.dir_couplers.amount,
            inhibition_samelayer=params.film_render.dir_couplers.inhibition_samelayer,
            inhibition_interlayer=params.film_render.dir_couplers.inhibition_interlayer,
            gamma_samelayer_rgb=tuple(params.film_render.dir_couplers.gamma_samelayer_rgb),
            gamma_interlayer_r_to_gb=tuple(params.film_render.dir_couplers.gamma_interlayer_r_to_gb),
            gamma_interlayer_g_to_rb=tuple(params.film_render.dir_couplers.gamma_interlayer_g_to_rb),
            gamma_interlayer_b_to_rg=tuple(params.film_render.dir_couplers.gamma_interlayer_b_to_rg),
            diffusion_size_um=params.film_render.dir_couplers.diffusion_size_um,
        ),
        glare=GlareState(
            active=params.print_render.glare.active,
            percent=params.print_render.glare.percent,
            roughness=params.print_render.glare.roughness,
            blur=params.print_render.glare.blur,
        ),
        chemistry=ChemistryState(
            active=params.print_render.density_curves_morph.active,
            gamma_factor=params.print_render.density_curves_morph.gamma_factor,
            gamma_factor_fast=params.print_render.density_curves_morph.gamma_factor_fast,
            gamma_factor_slow=params.print_render.density_curves_morph.gamma_factor_slow,
            gamma_factor_red=params.print_render.density_curves_morph.gamma_factor_red,
            gamma_factor_green=params.print_render.density_curves_morph.gamma_factor_green,
            gamma_factor_blue=params.print_render.density_curves_morph.gamma_factor_blue,
            developer_exhaustion=params.print_render.density_curves_morph.developer_exhaustion,
        ),
        push_pull=PushPullState(
            # Pure GUI state: 'standard' composes into density_curve_gamma and
            # 'experimental' constructs film_render.chemistry at params-build
            # time (see params_mapper._apply_push_pull). Defaults = no-op.
            active=True,
            mode='standard',
            stops=0.0,
            gamma_factor_fast=params.film_render.chemistry.gamma_factor_fast,
            gamma_factor_slow=params.film_render.chemistry.gamma_factor_slow,
            developer_exhaustion=params.film_render.chemistry.developer_exhaustion,
        ),
        aging=AgingState(
            # Film expiration: read from params with getattr fallbacks so
            # params trees predating the feature map to the inactive default.
            active=bool(getattr(_expiration(params), 'active', False)),
            age_years=float(getattr(_expiration(params), 'age_years', 0.0)),
            storage=str(getattr(_expiration(params), 'storage', 'room')),
            fog_rgb=tuple(getattr(_expiration(params), 'fog_rgb', (0.0, 0.0, 0.0))),
            gamma=float(getattr(_expiration(params), 'gamma', 1.0)),
            gamma_rgb=tuple(getattr(_expiration(params), 'gamma_rgb', (1.0, 1.0, 1.0))),
            density_scale=float(getattr(_expiration(params), 'density_scale', 1.0)),
            grain_gain=float(getattr(_expiration(params), 'grain_gain', 1.0)),
            speed_loss_ev=float(getattr(_expiration(params), 'speed_loss_ev', 0.0)),
        ),
        scan_finishing=ScanFinishingState(
            active=params.scanner.finishing.active,
            crop=params.scanner.finishing.crop,
            crop_center=tuple(params.scanner.finishing.crop_center),
            crop_size=tuple(params.scanner.finishing.crop_size),
            skew_deg=params.scanner.finishing.skew_deg,
            desqueeze=params.scanner.finishing.desqueeze,
            exposure_ev=params.scanner.finishing.exposure_ev,
            gain_red_ev=params.scanner.finishing.gain_red_ev,
            gain_green_ev=params.scanner.finishing.gain_green_ev,
            gain_blue_ev=params.scanner.finishing.gain_blue_ev,
            color_separation=params.scanner.finishing.color_separation,
            black_point=params.scanner.finishing.black_point,
            white_point=params.scanner.finishing.white_point,
            black_point_rgb=tuple(params.scanner.finishing.black_point_rgb),
            white_point_rgb=tuple(params.scanner.finishing.white_point_rgb),
            gamma=params.scanner.finishing.gamma,
            contrast=params.scanner.finishing.contrast,
            toe=params.scanner.finishing.toe,
            shoulder=params.scanner.finishing.shoulder,
            micro_contrast=getattr(params.scanner.finishing, 'micro_contrast', 0.0),
            saturation=params.scanner.finishing.saturation,
            vibrance=params.scanner.finishing.vibrance,
            hue_saturation_ryg=tuple(params.scanner.finishing.hue_saturation_ryg),
            hue_saturation_cbm=tuple(params.scanner.finishing.hue_saturation_cbm),
            shadow_tint_hue=params.scanner.finishing.shadow_tint_hue,
            shadow_tint_strength=params.scanner.finishing.shadow_tint_strength,
            highlight_tint_hue=params.scanner.finishing.highlight_tint_hue,
            highlight_tint_strength=params.scanner.finishing.highlight_tint_strength,
            curve_luma=_curve_points_state(getattr(params.scanner.finishing, 'curve_luma', None)),
            curve_r=_curve_points_state(getattr(params.scanner.finishing, 'curve_r', None)),
            curve_g=_curve_points_state(getattr(params.scanner.finishing, 'curve_g', None)),
            curve_b=_curve_points_state(getattr(params.scanner.finishing, 'curve_b', None)),
        ),
        input_gamut_compress=InputGamutCompressState(
            active=params.io.input_gamut_compress.active,
            algorithm=params.io.input_gamut_compress.algorithm,
            knee=tuple(params.io.input_gamut_compress.knee),
        ),
        output_gamut_compress=OutputGamutCompressState(
            # Shipped GUI default stays 'off': since the 0.3.4 base the ENGINE
            # default is cam16ucs, a per-pixel CPU op (~30 s at 24 MP) that
            # would sink the GPU render times. Users opt in from ADVANCED.
            algorithm='off',
            knee=tuple(params.io.output_gamut_compress.knee),
        ),
        special=SpecialState(
            film_channel_swap=(0, 1, 2),
            film_gamma_factor=params.film_render.density_curve_gamma,
            print_channel_swap=(0, 1, 2),
            # The 0.3.4 base dropped print_render.density_curve_gamma; the knob
            # is GUI state composed into the chemistry morph's gamma_factor at
            # params-build time (see params_mapper._apply_chemistry).
            print_gamma_factor=1.0,
        ),
        simulation=SimulationState(
            film_stock=film_stock,
            film_format_mm=float(params.camera.film_format_mm),
            camera_lens_blur_um=params.camera.lens_blur_um,
            camera_diffusion_filter_active=bool(params.camera.diffusion_filter.active),
            camera_diffusion_filter_family=params.camera.diffusion_filter.filter_family,
            camera_diffusion_filter_strength=float(params.camera.diffusion_filter.strength),
            camera_diffusion_filter_spatial_scale=float(params.camera.diffusion_filter.spatial_scale),
            camera_diffusion_filter_halo_warmth=float(params.camera.diffusion_filter.halo_warmth),
            camera_diffusion_filter_core_intensity=float(params.camera.diffusion_filter.core_intensity),
            camera_diffusion_filter_core_size=float(params.camera.diffusion_filter.core_size),
            camera_diffusion_filter_halo_intensity=float(params.camera.diffusion_filter.halo_intensity),
            camera_diffusion_filter_halo_size=float(params.camera.diffusion_filter.halo_size),
            camera_diffusion_filter_bloom_intensity=float(params.camera.diffusion_filter.bloom_intensity),
            camera_diffusion_filter_bloom_size=float(params.camera.diffusion_filter.bloom_size),
            exposure_compensation_ev=params.camera.exposure_compensation_ev,
            auto_exposure=params.camera.auto_exposure,
            auto_exposure_method=params.camera.auto_exposure_method,
            print_paper=print_paper,
            print_illuminant=params.enlarger.illuminant,
            print_exposure=params.enlarger.print_exposure,
            print_exposure_compensation=params.enlarger.print_exposure_compensation,
            print_y_filter_shift=params.enlarger.y_filter_shift,
            print_m_filter_shift=params.enlarger.m_filter_shift,
            print_c_filter_shift=getattr(params.enlarger, 'c_filter_shift', 0.0),
            film_development_time=0.0,   # session choice; not derivable from params
            print_shadow_shape=float(getattr(params.print_render, 'shadow_shape', 0.0)),
            print_highlight_shape=float(getattr(params.print_render, 'highlight_shape', 0.0)),
            negative_bleach_bypass=float(getattr(params.film_render, 'bleach_bypass_amount', 0.0)),
            negative_leuco_cyan_coupling=float(getattr(params.film_render, 'bleach_bypass_leuco_cyan', 1.0)),
            print_bleach_bypass=float(getattr(params.print_render, 'bleach_bypass_amount', 0.0)),
            diffusion_filter_active=bool(params.enlarger.diffusion_filter.active),
            diffusion_filter_family=params.enlarger.diffusion_filter.filter_family,
            diffusion_filter_strength=float(params.enlarger.diffusion_filter.strength),
            diffusion_filter_spatial_scale=float(params.enlarger.diffusion_filter.spatial_scale),
            diffusion_filter_halo_warmth=float(params.enlarger.diffusion_filter.halo_warmth),
            diffusion_filter_core_intensity=float(params.enlarger.diffusion_filter.core_intensity),
            diffusion_filter_core_size=float(params.enlarger.diffusion_filter.core_size),
            diffusion_filter_halo_intensity=float(params.enlarger.diffusion_filter.halo_intensity),
            diffusion_filter_halo_size=float(params.enlarger.diffusion_filter.halo_size),
            diffusion_filter_bloom_intensity=float(params.enlarger.diffusion_filter.bloom_intensity),
            diffusion_filter_bloom_size=float(params.enlarger.diffusion_filter.bloom_size),
            scan_lens_blur=params.scanner.lens_blur,
            scan_white_correction=params.scanner.white_correction,
            scan_white_level=params.scanner.white_level,
            scan_black_correction=params.scanner.black_correction,
            scan_black_level=params.scanner.black_level,
            scan_unsharp_mask=tuple(params.scanner.unsharp_mask),
            output_color_space="sRGB",
            saving_color_space="sRGB",
            saving_cctf_encoding=params.io.output_cctf_encoding,
            export_format=ExportFormats.jpeg_8.value,
            export_full_precision=True,
            auto_preview=False,
            scan_film=params.io.scan_film,
        ),
        display=DisplayState(
            use_display_transform=True,
            output_interpolation='spline36',
            white_padding=0.01,
            preview_max_size=params.settings.preview_max_size,
        ),
    )


def digest_after_selection(params: RuntimePhotoParams) -> RuntimePhotoParams:
    params = digest_params(params)
    params.io.scan_film = bool(params.film.is_positive)
    return params


def build_default_gui_state(*, film_stock: str, print_paper: str) -> GuiState:
    params = digest_after_selection(init_params(film_profile=film_stock, print_profile=print_paper))
    state = gui_state_from_params(params, film_stock=film_stock, print_paper=print_paper)
    # App-level FACTORY defaults on top of the engine params (NAT 2026-07-11).
    # Deliberately here and NOT in gui_state_from_params: profile switches
    # rebuild a synced state from the USER's current params through that
    # function, and must never clobber user-tweaked values with these.
    state.glare.roughness = 0.35
    state.glare.blur = 0.25
    state.simulation.camera_lens_blur_um = 0.2
    # Auto preview is on by default; the controller keeps it dormant until
    # the first explicit render of the session (NAT 2026-07-18). The toggle
    # is session-scoped: apply_gui_state never restores it from saved state.
    state.simulation.auto_preview = True
    return state


DEFAULT_FILM_STOCK = FilmStocks.kodak_gold_200.value
DEFAULT_PRINT_PAPER = PrintPapers.kodak_supra_endura.value
PROJECT_DEFAULT_GUI_STATE = build_default_gui_state(
    film_stock=DEFAULT_FILM_STOCK,
    print_paper=DEFAULT_PRINT_PAPER,
)
