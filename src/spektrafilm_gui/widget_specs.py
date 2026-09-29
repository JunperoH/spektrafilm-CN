from __future__ import annotations

from dataclasses import dataclass
from spektrafilm_gui.localization_gpu_zh_cn import translate_spec, translate_button_spec
from enum import Enum

from spektrafilm_gui.options import (
    AutoExposureMethods,
    DiffusionFilterFamilies,
    ExportFormats,
    GrainModels,
    InputGamutCompressAlgorithms,
    NapariInterpolationModes,
    OutputGamutCompressAlgorithms,
    PushPullModes,
    StorageConditions,
    RGBColorSpaces,
    RGBtoRAWMethod,
    RGBWhiteBalance,
    RawWhiteBalance,
)
from spektrafilm.model.color_filters import CameraColorFilters
from spektrafilm.model.illuminants import Illuminants
from spektrafilm.model.stocks import FilmStocks, PrintPapers


@dataclass(frozen=True, slots=True)
class WidgetSpec:
    label: str | None = None
    tooltip: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    decimals: int | None = None
    # Phase 15: marks a 3-float tuple as an RGB-per-channel triplet — its
    # row renders as three channel-tinted knobs + boxes (KnobTripletEditor).
    # Non-RGB triplets (e.g. amp/cutoff/sigma, per-layer scales) stay plain.
    rgb: bool = False


@dataclass(frozen=True, slots=True)
class ButtonSpec:
    text: str
    tooltip: str | None = None
    preserve_case: bool = False


GUI_SECTION_ENUMS: dict[str, dict[str, type[Enum]]] = {
    "input_image": {
        "input_color_space": RGBColorSpaces,
        "spectral_upsampling_method": RGBtoRAWMethod,
        "color_filter": CameraColorFilters,
    },
    "push_pull": {
        "mode": PushPullModes,
    },
    "grain": {
        "model": GrainModels,
    },
    "aging": {
        "storage": StorageConditions,
    },
    "load_raw": {
        "white_balance": RawWhiteBalance,
    },
    "load_rgb": {
        "white_balance": RGBWhiteBalance,
    },
    "display": {
        "output_interpolation": NapariInterpolationModes,
    },
    "input_gamut_compress": {
        "algorithm": InputGamutCompressAlgorithms,
    },
    "output_gamut_compress": {
        "algorithm": OutputGamutCompressAlgorithms,
    },
    "simulation": {
        "film_stock": FilmStocks,
        "auto_exposure_method": AutoExposureMethods,
        "camera_diffusion_filter_family": DiffusionFilterFamilies,
        "print_paper": PrintPapers,
        "print_illuminant": Illuminants,
        "output_color_space": RGBColorSpaces,
        "saving_color_space": RGBColorSpaces,
        "export_format": ExportFormats,
        "diffusion_filter_family": DiffusionFilterFamilies,
    },
}


GUI_WIDGET_SPECS = {
    "simulation": {
        "film_stock": WidgetSpec(label="Film profile", tooltip="Film stock to simulate"),
        "exposure_compensation_ev": WidgetSpec(
            label="Camera compensation ev",
            tooltip="Bias the camera auto-exposure, in stops. Real exposure "
                    "compensation is a few stops; past about +-8 the frame is "
                    "fully blown or crushed to flat grey.",
            min_value=-8,
            max_value=8,
            step=0.1,
        ),
        "auto_exposure": WidgetSpec(
            label="Camera auto exposure",
            tooltip="Use the auto-exposure feature of the virtual camera",
        ),
        "film_format_mm": WidgetSpec(
            label="Film format mm",
            tooltip="Long edge of the film frame in millimeters. Pick a preset "
                    "above (Super 8 to 8x10) or type a custom value here. "
                    "Larger formats = finer-looking grain at the same print size.",
            # NAT 2026-07-25: widened from 8-120 so the presets can reach
            # Super 8 (5.8) and large format (8x10 = 254mm long edge).
            min_value=5.0,
            max_value=260.0,
            step=1.0,
            decimals=0,
        ),
        "camera_lens_blur_um": WidgetSpec(
            label="Camera lens blur um",
            tooltip="Sigma of gaussian filter in um for the camera lens blur. About 5 um for typical lenses, down to 2-4 um for high quality lenses, used for sharp input simulations without lens blur.",
            step=0.05,
            min_value=0,
            max_value=20,
        ),
        "camera_diffusion_filter_active": WidgetSpec(
            label="Diffusion active",
            tooltip="Toggle the diffusion filter on the camera stage before film exposure.",
        ),
        "camera_diffusion_filter_family": WidgetSpec(
            label="Diffusion family",
            tooltip="PSF family used on the camera stage before film exposure.",
        ),
        "camera_diffusion_filter_strength": WidgetSpec(
            label="Diffusion strength",
            tooltip="Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2.",
            min_value=0,
            max_value=2,
            step=0.125,
        ),
        "camera_diffusion_filter_spatial_scale": WidgetSpec(
            label="Spatial scale",
            tooltip="Multiplier on the camera-stage image-plane PSF widths.",
            min_value=0,
            step=0.1,
            max_value=5,
        ),
        "camera_diffusion_filter_halo_warmth": WidgetSpec(
            label="Halo warmth",
            tooltip="Additive offset on the camera-stage halo warmth axis. Positive = warm outer halo / cool inner halo. Negative inverts. 0 = use family default.",
            min_value=-1.5,
            max_value=1.5,
            step=0.05,
        ),
        "camera_diffusion_filter_core_intensity": WidgetSpec(
            label="Core intensity",
            tooltip="Advanced. Multiplier on the camera-stage core weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_core_size": WidgetSpec(
            label="Core size",
            tooltip="Advanced. Multiplier on the camera-stage core lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_halo_intensity": WidgetSpec(
            label="Halo intensity",
            tooltip="Advanced. Multiplier on the camera-stage halo weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_halo_size": WidgetSpec(
            label="Halo size",
            tooltip="Advanced. Multiplier on the camera-stage halo lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_bloom_intensity": WidgetSpec(
            label="Bloom intensity",
            tooltip="Advanced. Multiplier on the camera-stage bloom weight. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "camera_diffusion_filter_bloom_size": WidgetSpec(
            label="Bloom size",
            tooltip="Advanced. Multiplier on the camera-stage bloom lambda. 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "print_paper": WidgetSpec(label="Print profile", tooltip="Print paper to simulate"),
        "print_illuminant": WidgetSpec(label="Print illuminant", tooltip="Print illuminant to simulate"),
        "print_exposure": WidgetSpec(
            label="Print exposure",
            tooltip="Enlarger exposure-time multiplier (1.0 = the metered "
                    "print). Past ~4x the print is crushed to paper black.",
            step=0.02,
            min_value=0,
            max_value=4,
        ),
        "print_exposure_compensation": WidgetSpec(
            label="Print auto compensation",
            tooltip="Auto adjust the print exposure for the camera exposure compensation ev",
        ),
        "print_y_filter_shift": WidgetSpec(
            label="Print Y filter shift",
            tooltip="Y filter shift of the color enlarger from a neutral position, in Kodak CC units",
            step=1,
            min_value=-170,
            max_value=170,
        ),
        "print_m_filter_shift": WidgetSpec(
            label="Print M filter shift",
            tooltip="M filter shift of the color enlarger from a neutral position, in Kodak CC units",
            step=1,
            min_value=-170,
            max_value=170,
        ),
        "print_shadow_shape": WidgetSpec(
            label="Shadow shape",
            tooltip="Shapes rendered shadows THROUGH the paper density response, endpoint-preserving (paper D-min/D-max unchanged). Positive opens shadows while keeping the deepest blacks; negative weighs them down. Composes with print push/pull.",
            min_value=-1.0,
            max_value=1.0,
            step=0.05,
            decimals=3,
        ),
        "print_highlight_shape": WidgetSpec(
            label="Highlight shape",
            tooltip="Shapes rendered highlights through the paper density response, endpoint-preserving. Negative gives gentler highlight rolloff (a more pulled-back shoulder); positive makes highlights harder.",
            min_value=-1.0,
            max_value=1.0,
            step=0.05,
            decimals=3,
        ),
        "negative_bleach_bypass": WidgetSpec(
            label="Negative bleach bypass",
            tooltip="Modeled extension (not measured stock data): retained silver on the NEGATIVE before print exposure / negative scan - adds neutral density proportional to development plus the leuco-cyan component. Small amounts add density and severity; rebalance print exposure after adding it.",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            decimals=3,
        ),
        "negative_leuco_cyan_coupling": WidgetSpec(
            label="Leuco-cyan coupling",
            tooltip="Advanced research multiplier for the negative bleach bypass' cyan-loss component (cyan dye that never forms when bleach is skipped). Keep at 1.0 unless intentionally testing the model.",
            min_value=0.0,
            max_value=3.0,
            step=0.1,
            decimals=3,
        ),
        "print_bleach_bypass": WidgetSpec(
            label="Print bleach bypass",
            tooltip="Modeled extension: retained silver on the PRINT material after print development, before scan - the silver-retention feel on the final print rather than the negative.",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            decimals=3,
        ),
        "film_development_time": WidgetSpec(
            label="Development time min",
            tooltip="B&W stocks only: pick a development time in minutes - the nearest member of the stock's measured development family is used (Double-X: 4/5/6.5/9/12; 2302 print: 2/3.5/5/7/9). 0 = the profile default (the family middle). Color stocks ignore this.",
            min_value=0.0,
            step=0.5,
            decimals=1,
            max_value=30,
        ),
        "print_c_filter_shift": WidgetSpec(
            label="Print C filter shift",
            tooltip="C filter shift of the color enlarger from a neutral position, in Kodak CC units. Classic darkroom practice keeps C at 0 (equal C+M+Y only adds neutral density), but the trim is available for looks that want it.",
            step=1,
            min_value=-170,
            max_value=170,
        ),
        "diffusion_filter_active": WidgetSpec(
            label="Diffusion active",
            tooltip="Toggle the diffusion filter (Pro-Mist family) on the print stage.",
        ),
        "diffusion_filter_family": WidgetSpec(
            label="Diffusion family",
            tooltip="PSF family. pro_mist / classic_soft / glimmerglass are transparent (energy-preserving); black_pro_mist absorbs a fraction of the deflected light, lifting shadows by reducing local contrast.",
        ),
        "diffusion_filter_strength": WidgetSpec(
            label="Diffusion strength",
            tooltip="Commercial filter stop: 0, 1/8=0.125, 1/4=0.25, 1/2=0.5, 1, 2. Maps internally to the (p_s, p_a) deflected/absorbed photon fractions.",
            min_value=0,
            max_value=2,
            step=0.125,
        ),
        "diffusion_filter_spatial_scale": WidgetSpec(
            label="Spatial scale",
            tooltip="Multiplier on the image-plane PSF widths (all per-group lambdas). Adjust for image-format / print-size differences.",
            min_value=0,
            step=0.1,
            max_value=5,
        ),
        "diffusion_filter_halo_warmth": WidgetSpec(
            label="Halo warmth",
            tooltip="Additive offset on the family's halo warmth axis. Positive = warm outer halo / cool inner halo (the look reference images show on mist and bloom filters). Negative inverts. Energy-preserving per channel. 0 = use family default.",
            min_value=-1.5,
            max_value=1.5,
            step=0.05,
        ),
        "diffusion_filter_core_intensity": WidgetSpec(
            label="Core intensity",
            tooltip="Advanced. Multiplier on the family's core weight. The three group weights (core, halo, bloom) are renormalized to sum to 1, so this reshuffles the relative split of energy between groups, not the total deflected fraction. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_core_size": WidgetSpec(
            label="Core size",
            tooltip="Advanced. Multiplier on the family's core lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_halo_intensity": WidgetSpec(
            label="Halo intensity",
            tooltip="Advanced. Multiplier on the family's halo weight. The three group weights (core, halo, bloom) are renormalized to sum to 1. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_halo_size": WidgetSpec(
            label="Halo size",
            tooltip="Advanced. Multiplier on the family's halo lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_bloom_intensity": WidgetSpec(
            label="Bloom intensity",
            tooltip="Advanced. Multiplier on the family's bloom weight. The three group weights (core, halo, bloom) are renormalized to sum to 1. 1.0 = use family default.",
            min_value=0,
            max_value=4,
            step=0.05,
        ),
        "diffusion_filter_bloom_size": WidgetSpec(
            label="Bloom size",
            tooltip="Advanced. Multiplier on the family's bloom lambda (all sub-components stretched together). 1.0 = use family default.",
            min_value=0.1,
            max_value=4,
            step=0.05,
        ),
        "scan_lens_blur": WidgetSpec(
            label="Scan lens blur",
            tooltip="Sigma of gaussian filter in pixel for the scanner lens blur",
            step=0.05,
            min_value=0,
            max_value=10,
        ),
        "scan_white_correction": WidgetSpec(
            label="Scan white correction",
            tooltip="Enable white point correction applied to the scanner output",
        ),
        "scan_white_level": WidgetSpec(
            label="Scan white level",
            tooltip="Target white level applied when white correction is enabled",
            min_value=0,
            max_value=1,
            step=0.005,
            decimals=3,
        ),
        "scan_black_correction": WidgetSpec(
            label="Scan black correction",
            tooltip="Enable black point correction applied to the scanner output",
        ),
        "scan_black_level": WidgetSpec(
            label="Scan black level",
            tooltip="Target black level applied when black correction is enabled",
            min_value=0,
            max_value=1,
            step=0.005,
            decimals=3,
        ),
        "scan_unsharp_mask": WidgetSpec(
            label="Scan unsharp mask",
            tooltip="Apply unsharp mask to the scan, [sigma in pixel, amount]",
            step=0.05,
            min_value=0,
        ),
        "output_color_space": WidgetSpec(label="Output color space", tooltip="Output color space of the simulation"),
        "saving_color_space": WidgetSpec(label="Saving color space", tooltip="Color space of the saved image file"),
        "saving_cctf_encoding": WidgetSpec(
            label="Encode tones (normal)",
            tooltip="On = tonally-normal: apply the colour space's transfer curve so the file looks correct on open (Lightroom/Capture One). Off = flat/linear: no curve, maximum grading headroom, looks flat on open (best for Resolve). Embeds the matching ICC variant either way.",
        ),
        "export_format": WidgetSpec(
            label="Export format",
            tooltip="JPEG (8-bit) for web; TIFF (16-bit) is the primary high-precision scan base; PNG (16-bit). 16-bit is written straight from the full-precision float render, never an 8-bit buffer promoted to 16-bit.",
        ),
        "export_full_precision": WidgetSpec(
            label="Full-precision re-render",
            tooltip="On = re-render the full-resolution input with exact per-pixel spectra (LUT-off; about the cost of one Scan) before writing - the highest-fidelity scanner base. Off = export the CURRENT scan as-is, no recompute (instant; uses the LUT result you already see). Both write the chosen bit depth from a float buffer; this only affects spectral fidelity and speed.",
        ),
        "auto_preview": WidgetSpec(label="Auto preview", tooltip="trigger the preview after every change of gui parameters, use mouse scrollwheel on parameters field, read preview tooltip for details"),
        "scan_film": WidgetSpec(label="Scan film", tooltip="Show a scan of the negative instead of the print"),
    },
    "display": {
        "use_display_transform": WidgetSpec(
            label="Use display transform",
            tooltip="Use Pillow.ImageCms to retrive the display transform (only in Windows) and apply it to the napari viewer output, if disabled the output color space is used",
        ),
        "output_interpolation": WidgetSpec(
            label="Output interpolation",
            tooltip="Napari interpolation mode used to display the output layer in the viewer.",
        ),
        "white_padding": WidgetSpec(
            label="White padding",
            tooltip="Expand the white border layer around the normalized preview frame, expressed as a fraction of the image long edge.",
            min_value=0,
            max_value=1,
            step=0.01,
        ),
        "preview_max_size": WidgetSpec(
            label="Preview max size",
            tooltip="max size of the long edge of the preview image in pixels",
            min_value=128,
            max_value=10240,
            step=128,
        ),
    },
    "special": {
        "film_gamma_factor": WidgetSpec(
            label="Film gamma factor",
            tooltip="Gamma factor of the density curves of the negative, < 1 reduce contrast, > 1 increase contrast",
            step=0.05,
            min_value=0,
            max_value=5,
        ),
        "film_channel_swap": WidgetSpec(label="Film channel swap"),
        "print_gamma_factor": WidgetSpec(
            label="Print gamma factor",
            tooltip="Gamma factor of the print paper, < 1 reduce contrast, > 1 increase contrast. Multiplies the Chemistry section's gamma factor (both act on the same print-curve morph).",
            step=0.05,
            min_value=0.05,
            max_value=5,
        ),
        "print_channel_swap": WidgetSpec(label="Print channel swap",
        min_value=0,
        max_value=2,
        step=1
        )
    },
    "glare": {
        "active": WidgetSpec(tooltip="Add glare to the print"),
        "percent": WidgetSpec(
            tooltip="Percentage of the glare light (typically 0.1-0.25)",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "roughness": WidgetSpec(
            tooltip="Roughness of the glare light (0-1)",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "blur": WidgetSpec(
            tooltip="Sigma of gaussian blur in pixels for the glare",
            min_value=0,
            step=0.1,
            max_value=5,
        ),
    },
    "halation": {
        "scatter_amount": WidgetSpec(
            tooltip="High-level scatter strength. 1.0 = full physical scatter, 0.0 = no scatter. Scales the fraction of light that undergoes in-emulsion scattering.",
            min_value=0,
            step=0.05,
            max_value=5,
        ),
        "scatter_spatial_scale": WidgetSpec(
            tooltip="High-level scatter size multiplier (1.0 = physical defaults). Scales both core and tail sigmas.",
            min_value=0,
            step=0.1,
            max_value=5,
        ),
        "halation_amount": WidgetSpec(
            tooltip="High-level halation strength multiplier (1.0 = physical defaults). Scales the per-channel halation amplitudes.",
            min_value=0,
            step=0.05,
            max_value=5,
        ),
        "halation_spatial_scale": WidgetSpec(
            tooltip="High-level halation size multiplier (1.0 = physical defaults). Scales the first-bounce sigma.",
            min_value=0,
            step=0.1,
            max_value=5,
        ),
        "boost_ev": WidgetSpec(
            tooltip="Maximum highlight boost in stops.",
            min_value=0,
            step=0.5,
            max_value=10,
        ),
        "protect_ev": WidgetSpec(
            tooltip="Protected range above midgray for the boost onset in stops.",
            min_value=0,
            step=0.5,
            max_value=12,
        ),
        "boost_range": WidgetSpec(
            tooltip="Controls how quickly the highlight boost ramps in, from 0 to 1.",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "scatter_core_um": WidgetSpec(
            tooltip="Sigma of the scatter core Gaussian per channel [R,G,B], in micrometers. Controls fine-scale sharpness loss in the emulsion.",
            min_value=0,
            max_value=20,
            step=0.5,
            rgb=True,
        ),
        "scatter_tail_um": WidgetSpec(
            tooltip="Decay constant of the scatter exponential tail per channel [R,G,B], in micrometers (internally approximated by a sum of Gaussians). Controls extended low-level spread within the emulsion.",
            min_value=0,
            max_value=50,
            step=1.0,
            rgb=True,
        ),
        "scatter_tail_weight": WidgetSpec(
            tooltip="Weight of the scatter tail Gaussian per channel [R,G,B] (0-100, percentage). Tail weight + core weight = 100. Higher values put more scattered light into the long tail.",
            min_value=0,
            max_value=100,
            step=1,
            rgb=True,
        ),
        "halation_strength": WidgetSpec(
            tooltip="Total back-reflection halation amplitude per channel [R,G,B] (0-100, percentage). Typical red channel: weak AH 2-8, no AH 8-25. The blue channel is usually near zero.",
            min_value=0,
            max_value=100,
            step=0.5,
            rgb=True,
        ),
        "halation_first_sigma_um": WidgetSpec(
            tooltip="Sigma of the first halation bounce per channel [R,G,B], in micrometers. Set by the base thickness (40-80 um for typical cine/still bases).",
            min_value=0,
            max_value=200,
            step=1.0,
            rgb=True,
        ),
        "halation_n_bounces": WidgetSpec(
            tooltip="Number of multi-bounce Gaussians summed in the halation pass. Subsequent bounces use sqrt(k)-spaced widths. Typical: 2-3.",
            min_value=1,
            max_value=5,
            step=1,
        ),
        "halation_bounce_decay": WidgetSpec(
            tooltip="Per-bounce amplitude decay ratio (rho). Physical range 0.3-0.7. Controls how fast the halation energy falls off between bounces.",
            min_value=0,
            max_value=1,
            step=0.05,
        ),
        "halation_renormalize": WidgetSpec(
            tooltip="If enabled, divide by (1 + sum of bounce amplitudes) so mid-grey is preserved. If disabled, halation is purely additive and subtly lifts shadows as well as highlights.",
        ),
    },
    "couplers": {
        "amount": WidgetSpec(
            tooltip="Global multiplier on the DIR coupler inhibition matrix. 1.0 leaves the per-channel gammas as-is.",
            min_value=0,
            step=0.05,
            max_value=4,
        ),
        "inhibition_samelayer": WidgetSpec(
            tooltip="Multiplier on the same-layer (diagonal) inhibition. Controls overall contrast / gamma reduction within each RGB layer.",
            min_value=0,
            step=0.05,
            max_value=4,
        ),
        "inhibition_interlayer": WidgetSpec(
            tooltip="Multiplier on the cross-layer (off-diagonal) inhibition. Controls saturation enhancement from interlayer DIR effects.",
            min_value=0,
            step=0.05,
            max_value=4,
        ),
        "gamma_samelayer_rgb": WidgetSpec(
            tooltip="Per-channel same-layer DIR gamma (R, G, B). Effective gamma reduction of each layer's density curve.",
            min_value=0,
            max_value=8,
            step=0.02,
            rgb=True,
        ),
        "gamma_interlayer_r_to_gb": WidgetSpec(
            tooltip="DIR inhibition from the R layer onto the G and B layers respectively (g_R->G, g_R->B).",
            min_value=0,
            step=0.02,
        ),
        "gamma_interlayer_g_to_rb": WidgetSpec(
            tooltip="DIR inhibition from the G layer onto the R and B layers respectively (g_G->R, g_G->B).",
            min_value=0,
            step=0.02,
        ),
        "gamma_interlayer_b_to_rg": WidgetSpec(
            tooltip="DIR inhibition from the B layer onto the R and G layers respectively (g_B->R, g_B->G).",
            min_value=0,
            step=0.02,
        ),
        "diffusion_size_um": WidgetSpec(
            tooltip="Sigma in um for the diffusion of the couplers, (5-20 um), controls sharpness and affects saturation.",
            min_value=0,
            step=5,
            max_value=200,
        ),
    },
    "grain": {
        "active": WidgetSpec(tooltip="Add grain to the negative"),
        "model": WidgetSpec(
            label="Grain model",
            tooltip="Which grain model renders (preview AND export - a model choice, not a preview/export split). 'production' is the film-accurate layered dye-cloud particle model. 'preview' is a fast impression matched to production's noise amplitude - use it for interactive look-building, then switch back for final texture. All grain settings below drive both models.",
        ),
        "particle_area_um2": WidgetSpec(
            tooltip="Area of the particles in um2, relates to ISO. Approximately 0.1 for ISO 100, 0.1 for ISO 200, 0.4 for ISO 400 and so on.",
            step=0.2,
            min_value=0,
            max_value=5,
        ),
        "particle_scale": WidgetSpec(
            tooltip="Scale of particle area for the RGB layers, multiplies particle_area_um2",
            min_value=0.05,
            max_value=8,
            step=0.1,
            rgb=True,
        ),
        "particle_scale_layers": WidgetSpec(
            tooltip="Scale of particle area for the sublayers in every color layer, multiplies particle_area_um2",
            min_value=0,
            step=0.25,
        ),
        "density_min": WidgetSpec(
            tooltip="Minimum density of the grain, typical values (0.03-0.06)",
            min_value=0,
            max_value=0.5,
            step=0.01,
            rgb=True,
        ),
        "uniformity": WidgetSpec(
            tooltip="Uniformity of the grain, typical values (0.94-0.98)",
            min_value=0,
            max_value=1,
            step=0.01,
            rgb=True,
        ),
        "blur": WidgetSpec(
            tooltip="Sigma of gaussian blur in pixels for the grain, to be increased at high magnifications, (should be 0.8-0.9 at high resolution, reduce down to 0.6 for lower res).",
            min_value=0,
            step=0.05,
            max_value=5,
        ),
        "blur_dye_clouds_um": WidgetSpec(
            tooltip="Scale the sigma of gaussian blur in um for the dye clouds, to be used at high magnifications, (default 1)",
            min_value=0,
            step=0.1,
            max_value=10,
        ),
        "micro_structure": WidgetSpec(
            tooltip="Parameter for micro-structure due to clumps at the molecular level, [sigma blur of micro-structure / ultimate light-resolution (0.10 um default), size of molecular clumps in nm (30 nm default)]. Only for insane magnifications.",
            min_value=0,
            step=0.1,
        ),
        "rms_granularity": WidgetSpec(
            label="RMS granularity",
            tooltip="RMS model only: the ISO-style granularity target per channel - sigma density x1000 measured through a 48 um aperture at net density 1.0, the number printed on datasheets (e.g. 4.5). The RMS model keeps the full production grain texture but derives its particle areas from this target.",
            min_value=0.1,
            max_value=30,
            step=0.5,
            rgb=True,
        ),
        "rms_strength": WidgetSpec(
            label="RMS strength",
            tooltip="RMS model only: scales the granularity target. 1.0 = the dialed values as-is; raise for more visible stock-relative grain, lower for a cleaner look.",
            min_value=0.0,
            step=0.1,
            max_value=4,
        ),
        "synthesis_size": WidgetSpec(
            label="Synthesis size",
            tooltip="Grain Synthesis only: multiplies the mean grain radius.",
            min_value=0.05,
            step=0.1,
            max_value=5,
        ),
        "synthesis_amount": WidgetSpec(
            label="Synthesis amount",
            tooltip="Grain Synthesis only: blends the synthesized grain field over the clean density (1.0 = full synthesis).",
            min_value=0.0,
            step=0.1,
            max_value=2,
        ),
        "synthesis_sharpness": WidgetSpec(
            label="Synthesis sharpness",
            tooltip="Grain Synthesis only: divides the observation aperture - higher = sharper, more resolved grain.",
            min_value=0.05,
            step=0.1,
            max_value=5,
        ),
        "synthesis_quality": WidgetSpec(
            label="Synthesis quality",
            tooltip="Grain Synthesis only: multiplies the Monte Carlo sample count. Higher = smoother estimate, slower render.",
            min_value=0.05,
            step=0.25,
            max_value=8,
        ),
        "synthesis_samples": WidgetSpec(
            label="Samples",
            tooltip="Base Monte Carlo samples per pixel (research control).",
            min_value=1,
            step=1,
            max_value=512,
        ),
        "synthesis_mean_radius_um": WidgetSpec(
            label="Mean radius um",
            tooltip="Mean silver-grain radius on the film plane, in micrometers (research control).",
            min_value=0.001,
            step=0.05,
            max_value=5,
        ),
        "synthesis_radius_stddev_ratio": WidgetSpec(
            label="Radius stddev ratio",
            tooltip="Lognormal radius spread as a fraction of the mean (research control).",
            min_value=0.001,
            step=0.05,
            max_value=3,
        ),
        "synthesis_aperture_sigma_um": WidgetSpec(
            label="Aperture sigma um",
            tooltip="Gaussian observation aperture on the film plane, in micrometers. 0 = automatic (half the pixel footprint).",
            min_value=0.0,
            step=0.25,
            max_value=20,
        ),
        "synthesis_cell_size_ratio": WidgetSpec(
            label="Cell size ratio",
            tooltip="Sampling cell size as a multiple of the mean radius (research control).",
            min_value=1.0,
            step=0.5,
            max_value=10,
        ),
        "synthesis_max_radius_quantile": WidgetSpec(
            label="Max radius quantile",
            tooltip="Lognormal quantile where grain radii are clamped (research control).",
            min_value=0.5,
            max_value=0.999999,
            step=0.001,
            decimals=4,
        ),
        "synthesis_coverage_epsilon": WidgetSpec(
            label="Coverage epsilon",
            tooltip="Floor for (1 - coverage) before the density log - caps the maximum synthesized density (research control).",
            min_value=1e-9,
            step=0.0001,
            decimals=6,
            max_value=0.1,
        ),
        "synthesis_max_grains_per_cell": WidgetSpec(
            label="Max grains per cell",
            tooltip="Cap on instantiated grains per sampling cell (research control).",
            min_value=1,
            step=1,
            max_value=128,
        ),
        "synthesis_radius_scale": WidgetSpec(
            label="Radius scale RGB",
            tooltip="Per-channel multiplier on the mean grain radius (research control).",
            min_value=0.05,
            max_value=4,
            step=0.1,
            rgb=True,
        ),
        "synthesis_layered": WidgetSpec(
            label="Layered",
            tooltip="Split the Boolean grain field into three stacked depth layers (radii scaled below, each carrying a third of D max). Coarser texture layering at ~3x the render cost (research control).",
        ),
        "synthesis_layer_scale": WidgetSpec(
            label="Layer scale",
            tooltip="Per-depth-layer multiplier on the mean grain radius when Layered is on (research control).",
            min_value=0.05,
            step=0.1,
        ),
    },
    "chemistry": {
        "active": WidgetSpec(
            tooltip="Enable print chemistry: morph the print density curves reconstructed from the profile's parametric model. The Tune tab's Print gamma factor multiplies this section's gamma factor and works even while this is off.",
        ),
        "gamma_factor": WidgetSpec(
            tooltip="Global coupled gamma multiplier for the print density curves.",
            min_value=0.5,
            max_value=2.0,
            step=0.05,
        ),
        "gamma_factor_fast": WidgetSpec(
            tooltip="Gamma factor applied to the fast sub-layer.",
            min_value=0.5,
            max_value=2.0,
            step=0.05,
        ),
        "gamma_factor_slow": WidgetSpec(
            tooltip="Gamma factor applied to the mid and slow sub-layers.",
            min_value=0.5,
            max_value=2.0,
            step=0.05,
        ),
        "gamma_factor_red": WidgetSpec(
            tooltip="Per-channel gamma factor applied to the red channel.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
        ),
        "gamma_factor_green": WidgetSpec(
            tooltip="Per-channel gamma factor applied to the green channel.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
        ),
        "gamma_factor_blue": WidgetSpec(
            tooltip="Per-channel gamma factor applied to the blue channel.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
        ),
        "developer_exhaustion": WidgetSpec(
            tooltip="Blend all three print sub-layers toward the matched Gumbel shoulder while preserving midgray via a common horizontal offset.",
            min_value=0.0,
            max_value=1.0,
            step=0.02,
        ),
    },
    "input_gamut_compress": {
        "active": WidgetSpec(
            tooltip="Compress input chromaticities toward the visible spectral locus before spectral upsampling; handles extreme out-of-gamut inputs. Baked into the input LUT at build time, so it costs nothing per pixel. Off passes input through unchanged.",
        ),
        "algorithm": WidgetSpec(
            tooltip="xy: radial compression in CIE 1931 chromaticity (ACES RGC family, hue-preserving). oklch: perceptual chroma reduction at constant Oklch (L, h).",
        ),
        "knee": WidgetSpec(
            tooltip="Reinhard knee (threshold, limit, power). Threshold must stay below 1. Defaults (0.0, 1.0, 6.0) apply a soft full-range roll-off that asymptotes at the spectral locus boundary.",
            min_value=0.0,
            step=0.05,
            decimals=2,
        ),
    },
    "output_gamut_compress": {
        "algorithm": WidgetSpec(
            tooltip="off (default): no output compression, the scan keeps its final [0,1] clip. cam16ucs: CAM16-UCS chroma reduction, smoothest constant-hue behavior. oklch: perceptual chroma reduction in OkLab. aces_rgc: per-channel ACES RGC v1.3, matches Resolve/Nuke/OCIO. oklrab/jzazbz: alternative perceptual spaces. WARNING: runs per pixel on the CPU - previews are fast, but a 24 MP scan/export adds ~30 s (cam16ucs), ~11 s (oklch) or ~3 s (aces_rgc).",
        ),
        "knee": WidgetSpec(
            tooltip="Reinhard knee (threshold, limit, power) on normalized chroma C/C_max. Threshold must stay below 1. Default (0.0, 1.0, 6.0) rolls off smoothly from the center with the asymptote on the output cube edge.",
            min_value=0.0,
            step=0.05,
            decimals=2,
        ),
    },
    "push_pull": {
        "active": WidgetSpec(
            label="Active",
            tooltip="Master bypass for push/pull development (quick A/B). Off ignores the stops and morph controls without losing their values.",
        ),
        "mode": WidgetSpec(
            label="Mode",
            tooltip="Push/pull development model. standard: the stable timing/gamma model - stops map onto the plain density-curve gamma (1.2x per stop, calibrated on Kodak's published Portra 800 push-1/push-2 data). experimental: tone-region and layer-dependent development warping (s023 chemistry morph) - the same stops drive a coupled per-sublayer gamma, plus the fast/slow and exhaustion controls below; grain inherits the process.",
        ),
        "stops": WidgetSpec(
            label="Push / pull stops",
            tooltip="Positive values push the negative (more contrast and effective speed), negative values pull (softer development). Applies before print exposure, grain, and scan-negative output. Rebalance film exposure after large pushes.",
            min_value=-3.0,
            max_value=3.0,
            step=0.25,
            decimals=2,
        ),
        "gamma_factor_fast": WidgetSpec(
            label="Fast-layer gamma",
            tooltip="Experimental mode only: extra development gamma on the FAST emulsion sub-layer (lowest threshold, develops on the least exposure - toe-dominant on negatives). 1.0 = neutral.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
            decimals=2,
        ),
        "gamma_factor_slow": WidgetSpec(
            label="Slow-layer gamma",
            tooltip="Experimental mode only: extra development gamma on the MID and SLOW emulsion sub-layers (higher thresholds - shoulder-dominant on negatives). 1.0 = neutral.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
            decimals=2,
        ),
        "developer_exhaustion": WidgetSpec(
            label="Developer exhaustion",
            tooltip="Experimental mode only: blends each sub-layer toward the matched Gumbel-max CDF (exhausted-developer response). Midgray D(0) and Dmax are preserved by construction. 0 = fresh developer.",
            min_value=0.0,
            max_value=1.0,
            step=0.02,
            decimals=2,
        ),
    },
    "aging": {
        "active": WidgetSpec(
            label="Active",
            tooltip="Master bypass for film expiration (quick A/B). Off ignores the age and manual knobs without losing their values. Aging changes the emulsion (fog, dye loss, crossover, grain, speed) and flows honestly through develop, print, and scan - the print filters can balance the global cast but not the crossover, exactly like a real aged negative.",
        ),
        "age_years": WidgetSpec(
            label="Age (years)",
            tooltip="Years past the expiry date. Drives a GENERIC expired-colour-negative model (base fog with a blue-heavy cast, contrast loss, colour crossover, dye/Dmax loss, coarser grain, about 1 stop speed loss per decade at room storage) - an empirically-tuned plausible look, NOT a datasheet. Bias it per stock with the manual knobs below.",
            min_value=0.0,
            max_value=60.0,
            step=1.0,
            decimals=1,
        ),
        "storage": WidgetSpec(
            label="Storage",
            tooltip="How the film was stored: modulates the aging rate. frozen (~x0.05) / cold fridge (~x0.3) / room (x1) / hot-humid (~x2.5, attic or tropics).",
        ),
        "fog_rgb": WidgetSpec(
            label="Fog R/G/B",
            tooltip="EXTRA per-channel base fog in density units, added on top of the age macro (which is blue-heaviest by default). Fog lifts D(0) and the toe - the colored shadow cast and lifted blacks no gamma change can make - while Dmax stays untouched.",
            min_value=-0.5,
            max_value=1.0,
            step=0.01,
            rgb=True,
        ),
        "gamma": WidgetSpec(
            label="Contrast trim",
            tooltip="EXTRA global development gamma, multiplied onto the macro's contrast loss. 1.0 = neutral; below 1 flattens the aged negative further.",
            min_value=0.3,
            max_value=2.0,
            step=0.02,
            decimals=2,
        ),
        "gamma_rgb": WidgetSpec(
            label="Crossover R/G/B",
            tooltip="EXTRA per-channel gamma (crossover-by-slope), multiplied onto the macro's tilt. Unequal values make shadows and highlights drift toward opposite hues - the crossover a print filter cannot balance away. 1.0 = neutral.",
            min_value=0.5,
            max_value=2.0,
            step=0.02,
            rgb=True,
        ),
        "density_scale": WidgetSpec(
            label="Dye density",
            tooltip="EXTRA dye-density / Dmax scale, multiplied onto the macro's dye loss. Below 1 fades all three dye layers - the muddy, desaturated aged look. 1.0 = neutral.",
            min_value=0.3,
            max_value=1.5,
            step=0.02,
            decimals=2,
        ),
        "grain_gain": WidgetSpec(
            label="Grain gain",
            tooltip="EXTRA grain particle-area gain, multiplied onto the macro's coarsening (aging coarsens grain; high-ISO stocks age faster - raise this for them). 1.0 = neutral.",
            min_value=0.25,
            max_value=4.0,
            step=0.05,
            decimals=2,
        ),
        "speed_loss_ev": WidgetSpec(
            label="Speed loss ev",
            tooltip="EXTRA speed loss in EV, added onto the macro's ~1 stop per decade. The emulsion becomes less sensitive (curves shift toward more exposure), so the negative thins unless you compensate the camera exposure - just like shooting real expired stock at box speed.",
            min_value=-3.0,
            max_value=6.0,
            step=0.25,
            decimals=2,
        ),
    },
    "scan_finishing": {
        "crop": WidgetSpec(
            label="Crop",
            tooltip="Scan-time crop: a cutout of the RENDERED film. Grain, halation and diffusion keep their full-frame scale - zoom into the grain, unlike the camera crop on MAIN which re-frames the film gate.",
        ),
        "crop_center": WidgetSpec(
            label="Crop center",
            tooltip="Center of the scan crop in normalized coordinates x, y (0-1) of the rendered scan.",
            step=0.01, min_value=0, max_value=1,
        ),
        "crop_size": WidgetSpec(
            label="Crop size",
            tooltip="Normalized size of the scan crop in x, y as fraction of the long side.",
            step=0.01, min_value=0, max_value=1,
        ),
        "skew_deg": WidgetSpec(
            label="Skew",
            tooltip="Rotate the rendered film under the scanner (degrees, counter-clockwise positive) before the scan crop - straighten horizons while the crop stays axis-aligned. Edges replicate; crop away the corners.",
            min_value=-45.0, max_value=45.0, step=0.1, decimals=1,
        ),
        "desqueeze": WidgetSpec(
            label="Desqueeze",
            tooltip="Anamorphic desqueeze at scan time, before skew and crop. Positive stretches the WIDTH (2.0 = classic 2x anamorphic; 1.33/1.5/1.8 for other lenses), negative stretches the HEIGHT. 0, 1 and -1 = off. Grain stretches with the image, exactly like a real anamorphic scan.",
            min_value=-2.0, max_value=2.0, step=0.01, decimals=2,
        ),
        "active": WidgetSpec(
            label="Active",
            tooltip="Master bypass for the scan finishing (quick A/B against the raw scan). All controls below default to identity - the render is bit-identical until something moves.",
        ),
        "exposure_ev": WidgetSpec(
            label="Scan exposure",
            tooltip="Scanner exposure trim in EV, applied to the LINEAR scan before encoding (a real scanner's lamp/analog gain).",
            min_value=-4.0, max_value=4.0, step=0.05, decimals=2,
        ),
        "gain_red_ev": WidgetSpec(
            label="Cyan / red",
            tooltip="Per-channel analog gain in EV: negative = toward cyan, positive = toward red. Linear-domain, like a scanner's channel gain (the CMY thinking of a lab, RGB math underneath).",
            min_value=-2.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "gain_green_ev": WidgetSpec(
            label="Magenta / green",
            tooltip="Per-channel analog gain in EV: negative = toward magenta, positive = toward green.",
            min_value=-2.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "gain_blue_ev": WidgetSpec(
            label="Yellow / blue",
            tooltip="Per-channel analog gain in EV: negative = toward yellow, positive = toward blue.",
            min_value=-2.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "color_separation": WidgetSpec(
            label="Color separation",
            tooltip="Density-domain dye unmixing (from NegPy's scanner emulation): subtracts each channel's contamination by the other two. NEUTRALS ARE PRESERVED exactly (the matrix is row-normalized) - purer color axes without a white-balance shift. 0 = off.",
            min_value=0.0, max_value=1.0, step=0.02, decimals=2,
        ),
        "black_point": WidgetSpec(
            label="Black point",
            tooltip="Levels: input value mapped to black. Master; the per-channel offsets below add to it. Unclipped - latitude is preserved.",
            min_value=-0.25, max_value=0.5, step=0.005, decimals=3,
        ),
        "white_point": WidgetSpec(
            label="White point",
            tooltip="Levels: input value mapped to white. Master; per-channel offsets add to it.",
            min_value=0.5, max_value=1.5, step=0.005, decimals=3,
        ),
        "black_point_rgb": WidgetSpec(
            label="Black point R/G/B",
            tooltip="Per-channel black point offsets added to the master (scanner per-channel floor - the Dmin analysis vocabulary the future real-film inversion will reuse).",
            min_value=-0.25, max_value=0.25, step=0.005, decimals=3,
            rgb=True,
        ),
        "white_point_rgb": WidgetSpec(
            label="White point R/G/B",
            tooltip="Per-channel white point offsets added to the master (scanner per-channel ceiling / Dmax).",
            min_value=-0.25, max_value=0.25, step=0.005, decimals=3,
            rgb=True,
        ),
        "gamma": WidgetSpec(
            label="Gamma",
            tooltip="Midtone gamma on the encoded scan (scanner gamma). 1 = neutral; >1 brightens midtones.",
            min_value=0.2, max_value=3.0, step=0.02, decimals=2,
        ),
        "contrast": WidgetSpec(
            label="Contrast",
            tooltip="Linear contrast about the 0.5 pivot of the encoded scan. Combine with toe/shoulder for a filmic S-curve. This is scanner-side contrast - print contrast lives in the enlarger/paper simulation.",
            min_value=-1.0, max_value=1.0, step=0.02, decimals=2,
        ),
        "toe": WidgetSpec(
            label="Toe",
            tooltip="Shadow soft knee: smoothly lifts-and-compresses the darkest values like a film toe (C1-continuous, monotone). 0 = off.",
            min_value=0.0, max_value=1.0, step=0.02, decimals=2,
        ),
        "shoulder": WidgetSpec(
            label="Shoulder",
            tooltip="Highlight soft knee: rolls the brightest values smoothly toward white like a paper shoulder. 0 = off.",
            min_value=0.0, max_value=1.0, step=0.02, decimals=2,
        ),
        "micro_contrast": WidgetSpec(
            label="Micro contrast",
            tooltip="Scanner texture: luma-only local contrast at grain scale (fixed 1.5 px radius). Positive sharpens fine detail and grain bite; negative softens it. Chroma untouched - no color fringing. 0 = off.",
            min_value=-1.0, max_value=1.0, step=0.05, decimals=2,
        ),
        "saturation": WidgetSpec(
            label="Saturation",
            tooltip="Global saturation about Rec.709 luminance. 1 = neutral.",
            min_value=0.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "vibrance": WidgetSpec(
            label="Vibrance",
            tooltip="Saturation weighted toward the least-saturated pixels (protects already-saturated colors and skin). 1 = neutral.",
            min_value=0.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "hue_saturation_ryg": WidgetSpec(
            label="Hue sat R/Y/G",
            tooltip="Per-hue saturation for the red / yellow / green axes (smooth 60-degree windows). 1 = neutral each.",
            min_value=0.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "hue_saturation_cbm": WidgetSpec(
            label="Hue sat C/B/M",
            tooltip="Per-hue saturation for the cyan / blue / magenta axes. 1 = neutral each.",
            min_value=0.0, max_value=2.0, step=0.02, decimals=2,
        ),
        "shadow_tint_hue": WidgetSpec(
            label="Shadow tint hue",
            tooltip="Split toning: hue (degrees, 0=red 120=green 240=blue) tinted into the shadows. Inert while the strength is 0.",
            min_value=0.0, max_value=360.0, step=5.0, decimals=0,
        ),
        "shadow_tint_strength": WidgetSpec(
            label="Shadow tint strength",
            tooltip="Split toning strength for the shadows (zero-sum chroma - luminance is approximately preserved).",
            min_value=0.0, max_value=1.0, step=0.02, decimals=2,
        ),
        "highlight_tint_hue": WidgetSpec(
            label="Highlight tint hue",
            tooltip="Split toning: hue tinted into the highlights. Inert while the strength is 0.",
            min_value=0.0, max_value=360.0, step=5.0, decimals=0,
        ),
        "highlight_tint_strength": WidgetSpec(
            label="Highlight tint strength",
            tooltip="Split toning strength for the highlights.",
            min_value=0.0, max_value=1.0, step=0.02, decimals=2,
        ),
    },
    "preflashing": {
        "active": WidgetSpec(
            label="Active",
            tooltip="Master bypass for the preflash (quick A/B). Off renders with zero preflash exposure while keeping the dialed values.",
        ),
        "exposure": WidgetSpec(
            tooltip="Preflash exposure value in ev for the print",
            step=0.005,
            min_value=0,
            max_value=2,
        ),
        "y_filter_shift": WidgetSpec(
            tooltip="Shift the Y filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
            min_value=-170,
            max_value=170,
        ),
        "m_filter_shift": WidgetSpec(
            tooltip="Shift the M filter of the enlarger from the neutral position for the preflash, typical values (-20-20), in Kodak CC units",
            step=1,
            min_value=-170,
            max_value=170,
        ),
        "c_filter_shift": WidgetSpec(
            tooltip="Shift the C filter of the enlarger from the neutral position for the preflash, in Kodak CC units. Classic darkroom practice keeps C at 0 (equal C+M+Y is only neutral density).",
            step=1,
            min_value=-170,
            max_value=170,
        ),
    },
    "input_image": {
        "crop": WidgetSpec(label="Crop", tooltip="Crop image to a fraction of the original size to preview details at full scale"),
        "crop_center": WidgetSpec(
            label="Crop center",
            tooltip="Center of the crop region in relative coordinates in x, y (0-1)",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "crop_size": WidgetSpec(
            label="Crop size",
            tooltip="Normalized size of the crop region in x, y (0,1), as fraction of the long side.",
            step=0.01,
            min_value=0,
            max_value=1,
        ),
        "input_color_space": WidgetSpec(
            label="Input color space",
            tooltip="Color space of the input image, will be internally converted to sRGB and negative values clipped",
        ),
        "apply_cctf_decoding": WidgetSpec(
            label="Apply CCTF decoding",
            tooltip="Apply the inverse cctf transfer function of the color space",
        ),
        "upscale_factor": WidgetSpec(label="Upscale factor", tooltip="Scale image size up to increase resolution",
                                     min_value=0.0,
                                     max_value=4,
                                     step=0.5,
                                     ),
        "spectral_upsampling_method": WidgetSpec(
            label="Spectral upsampling",
            tooltip="Method to reconstruct spectra from RGB. hanatos2025 (default): irradiance recovery over the full visible locus. jakob2019: bounded reflectance, smooth sigmoid (Jakob & Hanika 2019). otsu2018: bounded reflectance, clustered basis (Otsu 2018) - lowest error on real-surface colors, can show cluster edges on extreme colors. mallett2019: sRGB-basis reflectance via the shared LUT path. arctic2026beta02: memory-color reflectance with measured skin/vegetation priors (arctic, beta; daylight-tuned, local addition). mallett2019_legacy: the old per-pixel mallett (sRGB only, will clip input).",
        ),
        "apply_hanatos2025_adaptation_window": WidgetSpec(
            label="hanatos2025 adaptation window",
            tooltip="Apply the hanatos2025 bandpass adaptation window when reconstructing spectra.",
        ),
        "apply_hanatos2025_adaptation_surface": WidgetSpec(
            label="hanatos2025 adaptation surface",
            tooltip="Apply the hanatos2025 surface adaptation polynomial when reconstructing spectra.",
        ),
        "spectral_gaussian_blur": WidgetSpec(
            label="Spectral gaussian blur",
            tooltip="Sigma in nm for Gaussian blur applied to reconstructed spectra.",
            min_value=0,
            step=0.1,
            max_value=20,
        ),
        "filter_uv": WidgetSpec(
            label="UV filter",
            tooltip="Filter UV light, (amplitude, wavelength cutoff in nm, sigma in nm). It mainly helps for avoiding UV light ruining the reds. Changing this enlarger filters neutral will be affected.",
            min_value=0,
            step=1,
        ),
        "filter_ir": WidgetSpec(
            label="IR filter",
            tooltip="Filter IR light, (amplitude, wavelength cutoff in nm, sigma in nm). Changing this enlarger filters neutral will be affected.",
            min_value=0,
            step=1,
        ),
        "color_filter": WidgetSpec(
            label="Taking filter",
            tooltip="Camera color (taking) filter in front of the lens: its measured spectral transmittance is folded into the film sensitivity. The attenuation and color cast are the intended effect (classic B&W-style contrast filters: X0/X1 green, Y2/YA3 yellow, R1 red). 'none' disables it.",
        ),
    },
    "load_raw": {
        "white_balance": WidgetSpec(
            label="White balance",
            tooltip="Select white balance settings, if custom you can tune temperature and tint",
        ),
        "temperature": WidgetSpec(
            label="Temperature",
            tooltip="Temperature in Kelvin for the custom whitebalance, not used for the other white balance settings",
            step=100,
            min_value=1000,
            max_value=12000,
        ),
        "tint": WidgetSpec(
            label="Tint",
            tooltip="Tint value for the custom white balance, not used for the other white balance settings",
            min_value=0,
            step=0.01,
            max_value=5,
        ),
        "lens_correction": WidgetSpec(label="Lens correction", tooltip="Apply lens corrections"),
    },
    "load_rgb": {
        "white_balance": WidgetSpec(
            label="White balance",
            tooltip="White balance applied when loading the RGB image. 'as_shot' locks the file's default white (untouched), 'daylight' adapts from 5600K, 'tungsten' adapts from 4500K, 'custom' uses temperature and tint. Changes re-import the loaded file automatically.",
        ),
        "temperature": WidgetSpec(
            label="Temperature",
            tooltip="Scene temperature in Kelvin for the custom white balance; the image is adapted from this white to the 6504K daylight reference. Not used for the other white balance settings.",
            step=100,
            min_value=1000,
            max_value=12000,
        ),
        "tint": WidgetSpec(
            label="Tint",
            tooltip="Tint value for the custom white balance, 1.0 is neutral, applied to the green channel in linear RGB. Not used for the other white balance settings.",
            min_value=0,
            step=0.01,
            max_value=5,
        ),
        "recover_16bit": WidgetSpec(
            label="Color adaptation",
            tooltip=(
                "Compensates the flatter, less-saturated look a display-encoded TIFF/PNG/JPEG "
                "has versus a RAW import: a fixed +2% contrast and +3% colour density punch. "
                "Applies as soon as it is ticked (the loaded file is re-imported)."
            ),
        ),
    },
}


GUI_AUXILIARY_SPECS = {
    "scan_for_print": WidgetSpec(
        label="Scan for print",
        tooltip="Scan the image for print, ie white and black correction of the scanner are active, and glare is deactivated.",
    ),
}


GUI_BUTTON_SPECS = {
    "preview": ButtonSpec(
        text="PREVIEW",
        tooltip="run the simulation on a small preview and deactivates grain, halation, blurs, unsharp mask (diffusion filters are active)",
        preserve_case=True,
    ),
    "scan": ButtonSpec(
        text="SCAN",
        tooltip="Run the full simulation on the full-resolution input",
        preserve_case=True,
    ),
    "save": ButtonSpec(
        text="SAVE",
        tooltip="Save the current output layer to an image file",
        preserve_case=True,
    ),
}


EMPTY_WIDGET_SPEC = WidgetSpec()


def get_widget_spec(section_name: str, field_name: str) -> WidgetSpec:
    return translate_spec(section_name, field_name, GUI_WIDGET_SPECS.get(section_name, {}).get(field_name, EMPTY_WIDGET_SPEC))


def get_auxiliary_spec(name: str) -> WidgetSpec:
    return translate_spec('auxiliary', name, GUI_AUXILIARY_SPECS.get(name, EMPTY_WIDGET_SPEC))


def get_button_spec(name: str) -> ButtonSpec:
    return translate_button_spec(GUI_BUTTON_SPECS[name])
