from __future__ import annotations

from dataclasses import dataclass, field

from spektrafilm.profiles.io import Profile
from spektrafilm.utils.gamut_compression import (
    InputGamutCompressSpec,
    OutputGamutCompressSpec,
)
from spektrafilm.model.scan_finishing import ScanFinishingParams
from spektrafilm.utils.expiration import ExpirationParams
from spektrafilm.utils.morph_curves import FilmChemistryParams, PrintCurvesMorphParams



@dataclass
class DiffusionFilterParams:
    active: bool = False
    # filter_family selects PSF shape and absorption regime. Allowed values
    # are the keys of `_DIFFUSION_FILTER_SHAPES` in spektrafilm.model.diffusion.
    filter_family: str = "black_pro_mist"
    # commercial filter stops: 0, 1/8, 1/4, 1/2, 1, 2 (interpolated in between)
    strength: float = 0.5
    # multiplier on image-plane PSF widths (all per-group lambdas)
    spatial_scale: float = 1.0
    # additive bias to the family's halo warmth axis. The halo is energy-
    # conservingly redistributed across its sub-components per channel:
    # warmth > 0 pushes warm light (R + slight G) toward the OUTER halo
    # and cool light (B) toward the inner halo (and vice versa for
    # warmth < 0). 0 = use family default. Effective warmth is soft-
    # clamped to [-1.5, +1.5].
    halo_warmth: float = 0.0
    # Per-group fine-tune multipliers (advanced). Default 1.0 = use the
    # family preset unchanged. `*_intensity` scales the corresponding
    # group weight (w_c / w_h / w_b); the three weights are then
    # renormalized so they still sum to 1, i.e. the kernel stays
    # unit-normalised and the strength → p_s mapping is unchanged. So
    # these knobs reshuffle the relative split of energy between core,
    # halo and bloom, not the total deflected fraction. `*_size` scales
    # each group's lambda_um uniformly (all sub-components in that group
    # stretched by the same factor).
    core_intensity: float = 1.0
    core_size: float = 1.0
    halo_intensity: float = 1.0
    halo_size: float = 1.0
    bloom_intensity: float = 1.0
    bloom_size: float = 1.0


@dataclass
class CameraParams:
    exposure_compensation_ev: float = 0.0
    auto_exposure: bool = True
    auto_exposure_method: str = "center_weighted"
    lens_blur_um: float = 0.0
    film_format_mm: float = 35.0
    filter_uv: tuple[float, float, float] = (0.0, 410.0, 8.0)
    filter_ir: tuple[float, float, float] = (0.0, 675.0, 15.0)
    # color_filter selects a camera taking filter from the color filter library.
    # Allowed values are the members of `CameraColorFilters` in
    # spektrafilm.model.color_filters ("none" = no filter).
    color_filter: str = "none"
    diffusion_filter: DiffusionFilterParams = field(default_factory=DiffusionFilterParams)


@dataclass
class EnlargerParams:
    illuminant: str = "TH-KG3"
    print_exposure: float = 1.0
    print_exposure_compensation: bool = True
    normalize_print_exposure: bool = True
    y_filter_shift: float = 0.0
    m_filter_shift: float = 0.0
    # C trim (NAT 2026-07-20): the engine always modeled a full CMY head
    # (c_filter_neutral below); this exposes the user shift on top, exactly
    # like Y/M. Classic darkroom practice keeps C at 0 (equal C+M+Y is only
    # neutral density), so the default is a bit-identical no-op.
    c_filter_shift: float = 0.0
    y_filter_neutral: float = 55 # kodak cc values
    m_filter_neutral: float = 65 # kodak cc values
    c_filter_neutral: float = 0 # kodak cc values
    lens_blur: float = 0.0
    diffusion_filter: DiffusionFilterParams = field(default_factory=DiffusionFilterParams)
    preflash_exposure: float = 0.0
    preflash_y_filter_shift: float = 0.0
    preflash_m_filter_shift: float = 0.0
    # NAT 2026-07-22: preflash C trim, same construction as c_filter_shift
    # above — default 0 is a bit-identical no-op.
    preflash_c_filter_shift: float = 0.0


@dataclass
class ScannerParams:
    lens_blur: float = 0.0
    white_correction: bool = False
    black_correction: bool = False
    white_level: float = 0.98
    black_level: float = 0.01
    unsharp_mask: tuple[float, float] = (0.7, 0.7)
    # Display-referred scan finishing (SCAN tab): the scanner's own processing
    # on the digitized output. All defaults are strict no-ops (bit-identity).
    finishing: ScanFinishingParams = field(default_factory=ScanFinishingParams)


@dataclass
class GrainParams:
    active: bool = True
    # Phase 11 grain model tiers (OFX-style): 'production' = the layered
    # dye-cloud Poisson-binomial particle model (the only model until 11A;
    # default MUST stay bit-identical), 'preview' = fast NOT-film-accurate
    # filtered-noise impression (11A), 'rms' = the production model with its
    # particle areas DERIVED from the measured RMS granularity target (11B),
    # 'synthesis' = the Monte Carlo Boolean grain-field research model (11C).
    # Unknown values render as 'production'.
    model: str = 'production'
    # NAT 2026-07-22 (macOS-feel): render-time hint for the 'preview' tier
    # ONLY — the ratio full-resolution width / preview width. The preview
    # noise is computed at the FULL-RES pixel pitch (pixel_size_um / scale),
    # matching what the macOS app shows (it renders production grain at
    # native res and lets the display downscale keep the sparkle). 1.0 =
    # neutral (bit-identical); set by the GUI controller on preview renders,
    # never persisted, never used by exports or other models.
    preview_scale: float = 1.0
    sublayers_active: bool = True
    # 11B RMS: ISO-style granularity target — sigma_D x 1000 measured through
    # a 48 um aperture at net density 1.0 (the number printed on datasheets,
    # e.g. 4.5 for Gold 200). rms_strength scales it (1.0 = the preset).
    rms_granularity: tuple[float, float, float] = (5.0, 5.0, 5.0)
    rms_strength: float = 1.0
    # Phase 12A B&W: ONE shared noise field across the three broadcast
    # channels (neutral grain, never colored). Set by the pipeline from
    # profile.is_bw — never user-facing (mirrors spektrafilm.rs params.rs).
    monochrome: bool = False
    # 11C Grain Synthesis (research tier): simplified user controls...
    synthesis_size: float = 1.0            # multiplies the mean grain radius
    synthesis_amount: float = 1.0          # blend: D + amount*(synth - D)
    synthesis_sharpness: float = 1.0       # divides the observation aperture
    synthesis_quality: float = 1.0         # multiplies the MC sample count
    # ...and the advanced Boolean-model group (collapsed in the GUI).
    synthesis_samples: int = 16
    synthesis_mean_radius_um: float = 0.6
    synthesis_radius_stddev_ratio: float = 0.4
    synthesis_aperture_sigma_um: float = 0.0   # 0 = auto: half the pixel size
    synthesis_cell_size_ratio: float = 4.0     # cell size / mean radius
    synthesis_max_radius_quantile: float = 0.999
    synthesis_coverage_epsilon: float = 1e-4
    synthesis_max_grains_per_cell: int = 32
    synthesis_radius_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Polish 2026-07-21: the OFX's 'Layered' depth split — three stacked
    # Boolean grain fields per channel, radii scaled per depth layer, each
    # carrying a third of D_max (densities of stacked layers add).
    synthesis_layered: bool = False
    synthesis_layer_scale: tuple[float, float, float] = (2.0, 1.0, 0.5)
    particle_area_um2: float = 0.2
    particle_scale: tuple[float, float, float] = (1.6, 1.6, 3.2)
    particle_scale_layers: tuple[float, float, float] = (2.0, 1.0, 0.5)
    density_min: tuple[float, float, float] = (0.03, 0.03, 0.03)
    uniformity: tuple[float, float, float] = (0.97, 0.99, 0.97)
    blur: float = 0.65
    blur_dye_clouds_um: float = 1.0
    micro_structure: tuple[float, float] = (0.2, 30)
    n_sub_layers: int = 1


@dataclass
class HalationParams:
    active: bool = True
    # high-level scalars (default 1.0 preserves the physical low-level defaults)
    scatter_amount: float = 1.0
    scatter_spatial_scale: float = 1.0
    halation_amount: float = 1.0
    halation_spatial_scale: float = 1.0
    # in-emulsion scatter — energy-preserving mixture: Gaussian core + exponential
    # tail (scatter_tail_um is the exponential decay constant, internally
    # dispatched to a Gaussian mixture by fast_exponential_filter)
    scatter_core_um: tuple[float, float, float] = (2.2, 2.0, 1.6)
    scatter_tail_um: tuple[float, float, float] = (9.3, 9.7, 9.1)
    scatter_tail_weight: tuple[float, float, float] = (0.78, 0.65, 0.67)
    # highlight boost — reconstructs pre-clip irradiance before propagation
    boost_ev: float = 0.0
    boost_range: float = 0.3
    protect_ev: float = 4.0
    # back-reflection halation — additive sum of N Gaussians with sqrt(k) widths
    halation_strength: tuple[float, float, float] = (0.05, 0.015, 0.0)
    halation_first_sigma_um: tuple[float, float, float] = (65.0, 65.0, 65.0)
    halation_n_bounces: int = 3
    halation_bounce_decay: float = 0.5
    halation_renormalize: bool = True


@dataclass
class DirCouplersParams:
    active: bool = True
    amount: float = 1.0
    inhibition_samelayer: float = 1.0
    inhibition_interlayer: float = 1.0
    gamma_samelayer_rgb: tuple[float, float, float] = (0.341, 0.324, 0.273)
    gamma_interlayer_r_to_gb: tuple[float, float] = (0.355, 0.305)
    gamma_interlayer_g_to_rb: tuple[float, float] = (0.154, 0.358)
    gamma_interlayer_b_to_rg: tuple[float, float] = (0.171, 0.225)
    diffusion_size_um: float = 20.0
    diffusion_tail_um: float = 200.0 # exponential tail for Lévy-like processes or environmental heterogeneity
    diffusion_tail_weight: float = 0.06

@dataclass
class GlareParams:
    active: bool = True
    percent: float = 0.03
    roughness: float = 0.7
    blur: float = 0.5


@dataclass
class FilmRenderingParams:
    density_curve_gamma: float = 1.0
    grain: GrainParams = field(default_factory=GrainParams)
    halation: HalationParams = field(default_factory=HalationParams)
    dir_couplers: DirCouplersParams = field(default_factory=DirCouplersParams)
    glare: GlareParams = field(default_factory=GlareParams)
    # Film-side s023 chemistry morph (experimental push/pull): when active it
    # is baked into the film density curves + grain sublayers at pipeline init
    # (develop itself is untouched). Inactive (the default) leaves the shipped
    # profile arrays byte-identical.
    chemistry: FilmChemistryParams = field(
        default_factory=lambda: FilmChemistryParams(active=False)
    )
    # Film expiration (emulsion aging, Phase 7): fog + dye-density loss on the
    # sampled curves, gamma axes composed into the chemistry morph, grain
    # coarsening and speed loss folded into their existing systems — all baked
    # at pipeline init exactly like chemistry. Inactive (the default) is a
    # strict no-op: the shipped profile arrays pass through byte-identical.
    expiration: ExpirationParams = field(default_factory=ExpirationParams)
    # Phase 13B: NEGATIVE bleach bypass — retained silver before print
    # exposure / negative scan. Modeled extension ('in development' framing,
    # like the OFX): a spectrally NEUTRAL silver density proportional to the
    # mean developed density (exact through every spectral integral as a
    # 10^-S light factor), plus the leuco-cyan component (cyan dye that
    # never forms when bleach is skipped; coupling 1.0 = the intended path).
    bleach_bypass_amount: float = 0.0
    bleach_bypass_leuco_cyan: float = 1.0


@dataclass
class PrintRenderingParams:
    glare: GlareParams = field(default_factory=GlareParams)
    density_curves_morph: PrintCurvesMorphParams = field(
        default_factory=lambda: PrintCurvesMorphParams(active=False)
    )
    # Phase 13A: rendered toe/shoulder shaping THROUGH the print density
    # response (endpoint-preserving density remap after print develop).
    # Positive shadow_shape opens rendered shadows (softens the approach to
    # paper D-max); negative highlight_shape gives gentler highlight rolloff
    # (lifts the paper toe). 0/0 = bit-identical no-op.
    shadow_shape: float = 0.0
    highlight_shape: float = 0.0
    # Phase 13B: PRINT bleach bypass — retained silver on the print material
    # after print development, before scan. Modeled extension (a spectrally
    # neutral silver density proportional to the developed print density).
    bleach_bypass_amount: float = 0.0


@dataclass
class IOParams:
    input_color_space: str = "ProPhoto RGB"
    input_cctf_decoding: bool = False
    output_color_space: str = "sRGB"
    output_cctf_encoding: bool = True
    # Input gamut compression: smoothly pulls input chromaticities that
    # fall outside the visible spectral locus back inside (where Hanatos
    # 2025's spectral upsampling is well-defined). Baked into the
    # per-film tc_lut at build time so the per-pixel hot path is
    # untouched. See spektrafilm-research/studies/a00/a40_lut_system/n100
    # for the design.
    input_gamut_compress: InputGamutCompressSpec = field(default_factory=InputGamutCompressSpec)
    # Output gamut compression: smoothly compresses out-of-output-gamut
    # chromaticities (via the chroma knee) and above-white lightnesses
    # (via lightness_compression, a one-sided soft roll-off that leaves
    # black at 0) into the output primaries cube. With both engaged the
    # simulation output is guaranteed in [0, 1] and no downstream clip
    # is needed. See spektrafilm-research/studies/a00/a40_lut_system/n110
    # for the design and b40 for the smoothness analysis.
    output_gamut_compress: OutputGamutCompressSpec = field(default_factory=OutputGamutCompressSpec)
    # Camera/framing crop: the cropped region becomes the film gate (grain and
    # halation rescale). Scan-side geometry (skew/desqueeze/crop-into-grain)
    # lives on scanner.finishing instead.
    crop: bool = False
    crop_center: tuple[float, float] = (0.5, 0.5)
    crop_size: tuple[float, float] = (0.1, 0.1)
    upscale_factor: float = 1.0
    scan_film: bool = False


@dataclass
class DebugParams:
    deactivate_spatial_effects: bool = False
    deactivate_stochastic_effects: bool = False
    print_timings: bool = False
    # When True, the pipeline behaves as a deterministic per-pixel transform
    # suitable for LUT sampling: spatial effects, stochastic effects,
    # auto-exposure, and scanner white/black/unsharp corrections are all
    # forced off, regardless of the underlying settings.
    lut_mode: bool = False


@dataclass
class TapsParams:
    """Pipeline tap configuration.

    ``inject`` and ``collect`` name the entry and exit points in the
    pipeline topology. Defaults of None mean "normal end-to-end run"
    (inject at rgb_in, collect at rgb_out).
    """
    inject: str | None = None
    collect: str | None = None


@dataclass
class SettingsParams:
    rgb_to_raw_method: str = "hanatos2025"
    apply_hanatos2025_adaptation_window: bool = True
    apply_hanatos2025_adaptation_surface: bool = False
    spectral_gaussian_blur: float = 0.0
    use_enlarger_lut: bool = False
    use_scanner_lut: bool = False
    lut_resolution: int = 17
    use_fast_stats: bool = False
    preview_max_size: int = 1024   # NAT 2026-07-21 (was 640)
    preview_mode: bool = False
    neutral_print_filters_from_database: bool = True
    

@dataclass
class RuntimePhotoParams:
    film: Profile
    print: Profile
    film_render: FilmRenderingParams = field(default_factory=FilmRenderingParams)
    print_render: PrintRenderingParams = field(default_factory=PrintRenderingParams)
    camera: CameraParams = field(default_factory=CameraParams)
    enlarger: EnlargerParams = field(default_factory=EnlargerParams)
    scanner: ScannerParams = field(default_factory=ScannerParams)
    io: IOParams = field(default_factory=IOParams)
    debug: DebugParams = field(default_factory=DebugParams)
    settings: SettingsParams = field(default_factory=SettingsParams)
    taps: TapsParams = field(default_factory=TapsParams)

    def __post_init__(self):
        if not isinstance(self.film, Profile):
            raise TypeError("film must be a Profile instance")
        if not isinstance(self.print, Profile):
            raise TypeError("print must be a Profile instance")
