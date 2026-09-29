from __future__ import annotations

from enum import Enum


class RGBColorSpaces(Enum):
    sRGB = "sRGB"
    DCI_P3 = "DCI-P3"
    DisplayP3 = "Display P3"
    AdobeRGB = "Adobe RGB (1998)"
    ITU_R_BT2020 = "ITU-R BT.2020"
    ProPhotoRGB = "ProPhoto RGB"
    ACES2065_1 = "ACES2065-1"


class ExportFormats(Enum):
    """Export format presets. Each maps to a (file extension, bit depth) pair
    via ``EXPORT_FORMAT_SPECS``. 16-bit options come straight from the
    full-precision float render, never an 8-bit buffer promoted to 16-bit."""

    jpeg_8 = "JPEG (8-bit)"
    tiff_16 = "TIFF (16-bit)"
    png_16 = "PNG (16-bit)"


# Maps each ExportFormats value -> (extension passed to save_image_oiio, bit_depth).
EXPORT_FORMAT_SPECS: dict[str, tuple[str, int]] = {
    ExportFormats.jpeg_8.value: ("jpg", 8),
    ExportFormats.tiff_16.value: ("tif", 16),
    ExportFormats.png_16.value: ("png", 16),
}


# Colour spaces whose gamut is too wide to encode safely in 8 bits: pairing any
# of these with an 8-bit format (JPEG) risks visible posterisation/banding. The
# export panel warns when this combination is selected.
WIDE_GAMUT_COLOR_SPACES: frozenset[str] = frozenset({
    "Adobe RGB (1998)",
    "ITU-R BT.2020",
    "ProPhoto RGB",
    "ACES2065-1",
    "DCI-P3",
    "Display P3",
})


def export_format_spec(format_value: str) -> tuple[str, int]:
    """Return (extension, bit_depth) for an ExportFormats value."""
    return EXPORT_FORMAT_SPECS[format_value]


class RGBtoRAWMethod(Enum):
    # Registry-backed samplers (data-driven: one .npy + .toml per method under
    # data/luts/spectral_upsampling; new methods appear here when shipped).
    hanatos2025 = "hanatos2025"          # irradiance recovery (default)
    jakob2019 = "jakob2019"              # bounded reflectance, smooth sigmoid
    otsu2018 = "otsu2018"                # bounded reflectance, clustered basis
    mallett2019 = "mallett2019"          # sRGB-basis reflectance via the registry LUT
    # LOCAL ADDITION: arctic2026 beta02 memory-color reflectance (pixls.us t/57512/59),
    # data-driven via its .npy + .toml pair under data/luts/spectral_upsampling.
    arctic2026beta02 = "arctic2026beta02"
    # Pre-registry per-pixel mallett path kept verbatim (green-channel midgray
    # normalization, unclipped out-of-sRGB reflectance).
    mallett2019_legacy = "mallett2019_legacy"


class GrainModels(Enum):
    # Phase 11 grain model tiers (OFX-style). 'production' = the layered
    # dye-cloud particle model (film-accurate, the render tier and default);
    # 'preview' = fast filtered-noise impression matched to the production
    # model's per-pixel std (look-building tier); 'rms' = production texture
    # with the amplitude CALIBRATED to the ISO-style RMS granularity value
    # (11B); 'synthesis' = the Monte Carlo Boolean grain-field research
    # tier, much heavier (11C).
    production = "production"
    preview = "preview"
    rms = "rms"
    synthesis = "synthesis"


class PushPullModes(Enum):
    # Film push/pull development model: 'standard' = plain density-curve gamma
    # (stable lab timing/gamma model); 'experimental' = the film-side s023
    # chemistry morph (per-sublayer coupled gamma + developer exhaustion).
    standard = "standard"
    experimental = "experimental"


class StorageConditions(Enum):
    # Film expiration (Aging section): storage modulates the age->degradation
    # rate. Values mirror spektrafilm.utils.expiration.STORAGE_CONDITIONS.
    frozen = "frozen"
    cold = "cold"
    room = "room"
    hot_humid = "hot-humid"


class RawWhiteBalance(Enum):
    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class RGBWhiteBalance(Enum):
    """White balance modes for already-demosaiced RGB input (Import RGB).

    Mirrors the RAW dropdown's four options. There is no camera metadata for a
    demosaiced file, so ``as_shot`` here means "lock the file's default white"
    (leave it untouched). ``daylight`` and ``tungsten`` adapt from a fixed scene
    white to the 6504 K daylight reference; ``custom`` uses temperature/tint.

    Because the two import paths start from different base whites, the tungsten
    scene white is tuned per path (RAW 3800 K, RGB 4500 K) so the tungsten look
    matches across RAW and RGB."""

    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class AutoExposureMethods(Enum):
    center_weighted = "center_weighted"
    matrix = "matrix"
    multi_zone = "multi_zone"
    partial = "partial"
    highlight_weighted = "highlight_weighted"
    median = "median"
    average = "average"


class NapariInterpolationModes(Enum):
    nearest = "nearest"
    linear = "linear"
    cubic = "cubic"
    spline16 = "spline16"
    spline36 = "spline36"
    lanczos = "lanczos"
    blackman = "blackman"


class DiffusionFilterFamilies(Enum):
    glimmerglass = "glimmerglass"
    black_pro_mist = "black_pro_mist"
    pro_mist = "pro_mist"
    cinebloom = "cinebloom"


class InputGamutCompressAlgorithms(Enum):
    xy = "xy"
    oklch = "oklch"


class OutputGamutCompressAlgorithms(Enum):
    off = "off"
    oklch = "oklch"
    aces_rgc = "aces_rgc"
    oklrab = "oklrab"
    jzazbz = "jzazbz"
    cam16ucs = "cam16ucs"
