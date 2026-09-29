"""Display-referred scan finishing — the SCAN tab's engine.

Models a scanner's own processing chain on the digitized output of the
simulated (later: real, inverted) scan. Two groups, applied at the end of the
scanning stage:

- LINEAR group (before the output cctf): scan exposure and per-channel analog
  gain (EV; the GUI labels them as bipolar Cyan<->Red / Magenta<->Green /
  Yellow<->Blue pairs), and color separation — NegPy's density-domain channel
  unmixing (row-normalized matrix blend, so NEUTRALS ARE PRESERVED exactly).
- DISPLAY group (on the encoded output; when the file is written linear the
  ops run inside a temporary encode/decode round-trip so the LOOK is identical
  regardless of the write-time encoding): black/white points (master +
  per-channel offsets), midtone gamma, contrast about the 0.5 pivot,
  exponential toe/shoulder soft knees (C1-continuous, monotone; the shoulder
  rolls smoothly to 1 like paper white, the toe lifts-and-compresses like a
  film toe), saturation, vibrance, per-hue saturation (R/Y/G and C/B/M
  triples), and shadow/highlight split toning (zero-sum chroma vectors, so
  luminance is approximately preserved).

Every parameter defaults to identity and every op is guarded by an exact
identity check: at defaults the input array is returned UNTOUCHED (same
object), pinning bit-identity of the historical render.

This stage is deliberately NOT part of the spectral simulation: it is the
digitization/finishing end (parity-safe by construction), and it is the tab
that will later serve real inverted film the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ScanFinishingParams",
    "scan_finishing_is_identity",
    "scan_geometry_is_identity",
    "apply_scan_finishing_linear",
    "apply_scan_finishing_display",
    "apply_scan_geometry",
    "build_display_curve_lut",
    "CURVE_IDENTITY",
    "CURVE_LUT_SIZE",
]


# NegPy's default dye-crosstalk unmixing matrix (lab stage): negative
# off-diagonals subtract each channel's contamination by the other two.
_SEPARATION_MATRIX = np.array([
    [1.00, -0.05, -0.02],
    [-0.04, 1.00, -0.08],
    [-0.01, -0.10, 1.00],
])

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])  # Rec.709

# Hue centers for the per-hue saturation triples, degrees: R Y G / C B M.
_HUE_CENTERS_RYG = (0.0, 60.0, 120.0)
_HUE_CENTERS_CBM = (180.0, 240.0, 300.0)

# Micro-contrast ("texture"): fixed small radius so it bites on grain-scale
# detail without the halos of large-radius local contrast. Luma-only.
MICRO_CONTRAST_SIGMA = 1.5

# Phase 10A finishing point curve (Lightroom-style): control points in [0,1]^2,
# monotone-cubic (PCHIP) in x with y free — non-monotone y is allowed, this is
# display-referred and any shape is safe. Evaluated through a sampled 1D LUT
# (CURVE_LUT_SIZE entries/channel) so the CPU and the GPU read the exact same
# table; inputs outside [0,1] are clamped into it (flat extension), which is
# also Lightroom's behavior.
CURVE_IDENTITY = ((0.0, 0.0), (1.0, 1.0))
CURVE_LUT_SIZE = 1024


@dataclass
class ScanFinishingParams:
    active: bool = True
    # -- geometry group (applied LAST, on the final scan: a cutout of the
    # rendered film — grain/halation keep their full-frame scale, unlike the
    # camera crop in io params which re-frames the film gate) ---------------
    crop: bool = False
    crop_center: tuple = (0.5, 0.5)
    crop_size: tuple = (1.0, 1.0)
    skew_deg: float = 0.0                    # rotate the film under the scanner
    desqueeze: float = 0.0                   # anamorphic desqueeze at scan time
    # -- linear group ------------------------------------------------------
    exposure_ev: float = 0.0                 # scan exposure trim
    gain_red_ev: float = 0.0                 # bipolar Cyan <-> Red
    gain_green_ev: float = 0.0               # bipolar Magenta <-> Green
    gain_blue_ev: float = 0.0                # bipolar Yellow <-> Blue
    color_separation: float = 0.0            # 0..1 toward the unmixing matrix
    # -- display group -----------------------------------------------------
    black_point: float = 0.0
    white_point: float = 1.0
    black_point_rgb: tuple = (0.0, 0.0, 0.0)  # per-channel offsets
    white_point_rgb: tuple = (0.0, 0.0, 0.0)
    gamma: float = 1.0
    contrast: float = 0.0                     # -1..1, pivot 0.5
    toe: float = 0.0                          # 0..1 shadow knee strength
    shoulder: float = 0.0                     # 0..1 highlight knee strength
    saturation: float = 1.0
    vibrance: float = 1.0
    hue_saturation_ryg: tuple = (1.0, 1.0, 1.0)
    hue_saturation_cbm: tuple = (1.0, 1.0, 1.0)
    shadow_tint_hue: float = 210.0            # inert while strength is 0
    shadow_tint_strength: float = 0.0
    highlight_tint_hue: float = 45.0
    highlight_tint_strength: float = 0.0
    # Point curve (after split toning, before micro_contrast): luma curve
    # applies to all three channels, then the per-channel curves.
    curve_luma: tuple = CURVE_IDENTITY
    curve_r: tuple = CURVE_IDENTITY
    curve_g: tuple = CURVE_IDENTITY
    curve_b: tuple = CURVE_IDENTITY
    micro_contrast: float = 0.0               # -1..1, luma texture at MICRO_CONTRAST_SIGMA px


def _linear_is_identity(p: ScanFinishingParams) -> bool:
    return (
        float(p.exposure_ev) == 0.0
        and float(p.gain_red_ev) == 0.0
        and float(p.gain_green_ev) == 0.0
        and float(p.gain_blue_ev) == 0.0
        and float(p.color_separation) == 0.0
    )


def _normalize_curve_points(points) -> tuple:
    """Tolerant normalization for curve control points: JSON round-trips hand
    back lists of lists, old states have no curve fields at all. Sorted by x;
    duplicate x keeps the LAST point (the most recently placed one)."""
    if points is None:
        return CURVE_IDENTITY
    cleaned = [(float(x), float(y)) for x, y in points]
    cleaned.sort(key=lambda pt: pt[0])
    deduped: list = []
    for pt in cleaned:
        if deduped and deduped[-1][0] == pt[0]:
            deduped[-1] = pt
        else:
            deduped.append(pt)
    return tuple(deduped)


def _curve_is_identity(points) -> bool:
    normalized = _normalize_curve_points(points)
    return len(normalized) < 2 or normalized == CURVE_IDENTITY


def _curves_are_identity(p: ScanFinishingParams) -> bool:
    return (
        _curve_is_identity(getattr(p, 'curve_luma', None))
        and _curve_is_identity(getattr(p, 'curve_r', None))
        and _curve_is_identity(getattr(p, 'curve_g', None))
        and _curve_is_identity(getattr(p, 'curve_b', None))
    )


def _curve_evaluator(points):
    """PCHIP through the normalized points, flat outside their x range.
    Returns None for an identity curve (callers pass values through)."""
    if _curve_is_identity(points):
        return None
    normalized = _normalize_curve_points(points)
    xs = np.array([pt[0] for pt in normalized])
    ys = np.array([pt[1] for pt in normalized])
    from scipy.interpolate import PchipInterpolator

    interpolator = PchipInterpolator(xs, ys)

    def evaluate(values: np.ndarray) -> np.ndarray:
        return interpolator(np.clip(values, xs[0], xs[-1]))
    return evaluate


def build_display_curve_lut(p: ScanFinishingParams, size: int = CURVE_LUT_SIZE):
    """The finishing curve as three per-channel 1D LUTs (float32 (size, 3)),
    or None when every curve is identity. The luma-then-per-channel
    composition is folded in at BUILD time (exact function composition, no
    intermediate sampling), so applying is a single linear-interp lookup —
    the same table on the CPU and the GPU."""
    if _curves_are_identity(p):
        return None
    grid = np.linspace(0.0, 1.0, int(size))
    master = _curve_evaluator(getattr(p, 'curve_luma', None))
    base = grid if master is None else master(grid)
    lut = np.empty((int(size), 3), dtype=np.float32)
    for column, name in enumerate(('curve_r', 'curve_g', 'curve_b')):
        channel = _curve_evaluator(getattr(p, name, None))
        lut[:, column] = base if channel is None else channel(base)
    return lut


def _apply_curve_lut(out: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Linear-interp lookup into the sampled curve, mirroring the shader's
    curve_sample exactly (clamp to [0,1], gather, lerp)."""
    n = lut.shape[0]
    table = lut.astype(np.float64)
    t = np.clip(out, 0.0, 1.0) * (n - 1)
    i0 = np.floor(t).astype(np.intp)
    i1 = np.minimum(i0 + 1, n - 1)
    fraction = t - i0
    channels = np.arange(3)
    a = table[i0, channels]
    b = table[i1, channels]
    return a + (b - a) * fraction


def _display_is_identity(p: ScanFinishingParams) -> bool:
    if not _curves_are_identity(p):
        return False
    return (
        float(p.black_point) == 0.0
        and float(p.white_point) == 1.0
        and tuple(float(v) for v in p.black_point_rgb) == (0.0, 0.0, 0.0)
        and tuple(float(v) for v in p.white_point_rgb) == (0.0, 0.0, 0.0)
        and float(p.gamma) == 1.0
        and float(p.contrast) == 0.0
        and float(p.toe) == 0.0
        and float(p.shoulder) == 0.0
        and float(p.saturation) == 1.0
        and float(p.vibrance) == 1.0
        and tuple(float(v) for v in p.hue_saturation_ryg) == (1.0, 1.0, 1.0)
        and tuple(float(v) for v in p.hue_saturation_cbm) == (1.0, 1.0, 1.0)
        and float(p.shadow_tint_strength) == 0.0
        and float(p.highlight_tint_strength) == 0.0
        and float(getattr(p, 'micro_contrast', 0.0)) == 0.0
    )


def scan_geometry_is_identity(p: ScanFinishingParams | None) -> bool:
    if p is None or not p.active:
        return True
    return (
        not bool(getattr(p, 'crop', False))
        and float(getattr(p, 'skew_deg', 0.0)) == 0.0
        and float(getattr(p, 'desqueeze', 0.0)) in (0.0, 1.0, -1.0)
    )


def scan_finishing_is_identity(p: ScanFinishingParams | None) -> bool:
    if p is None or not p.active:
        return True
    return _linear_is_identity(p) and _display_is_identity(p)


def apply_scan_geometry(rgb: np.ndarray, p: ScanFinishingParams) -> np.ndarray:
    """Scan-time geometry, applied to the FINAL scan (after the color
    finishing): desqueeze -> skew -> crop, all as a cutout of the rendered
    film. Grain and halation keep their full-frame scale — this is zooming
    into the finished scan, not re-framing the camera."""
    if scan_geometry_is_identity(p):
        return rgb
    from spektrafilm.utils.crop_resize import apply_desqueeze, apply_skew, crop_image

    out = np.asarray(rgb)
    desqueeze = float(getattr(p, 'desqueeze', 0.0))
    if desqueeze not in (0.0, 1.0, -1.0):
        out = apply_desqueeze(out, desqueeze)
    skew = float(getattr(p, 'skew_deg', 0.0))
    if skew != 0.0:
        out = apply_skew(out, skew)
    if bool(getattr(p, 'crop', False)):
        out = crop_image(out, center=tuple(p.crop_center), size=tuple(p.crop_size))
    return out


# --------------------------------------------------------------------------- #
# linear group
# --------------------------------------------------------------------------- #
def apply_scan_finishing_linear(rgb: np.ndarray, p: ScanFinishingParams) -> np.ndarray:
    """Scanner analog side, on LINEAR output-space RGB."""
    if not p.active or _linear_is_identity(p):
        return rgb
    out = np.asarray(rgb, dtype=float)

    gains = 2.0 ** (float(p.exposure_ev) + np.array(
        [float(p.gain_red_ev), float(p.gain_green_ev), float(p.gain_blue_ev)]))
    if np.any(gains != 1.0):
        out = out * gains

    separation = min(max(float(p.color_separation), 0.0), 1.0)
    if separation > 0.0:
        # Density-domain unmixing (NegPy lab): blend identity toward the
        # unmixing matrix and row-normalize so a neutral (equal densities)
        # maps to itself exactly.
        matrix = np.eye(3) * (1.0 - separation) + _SEPARATION_MATRIX * separation
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-6)
        density = -np.log10(np.fmax(out, 1e-6))
        density = np.einsum('...c,kc->...k', density, matrix)
        out = np.where(out > 1e-6, 10.0 ** -density, out)

    return out


# --------------------------------------------------------------------------- #
# display group
# --------------------------------------------------------------------------- #
def _rgb_hue_chroma(out: np.ndarray):
    mx = out.max(axis=-1)
    mn = out.min(axis=-1)
    chroma = mx - mn
    safe = np.where(chroma > 1e-9, chroma, 1.0)
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    hue = np.zeros_like(mx)
    m_r = (mx == r)
    m_g = (mx == g) & ~m_r
    m_b = ~(m_r | m_g)
    hue[m_r] = (60.0 * ((g - b) / safe))[m_r] % 360.0
    hue[m_g] = (60.0 * ((b - r) / safe) + 120.0)[m_g]
    hue[m_b] = (60.0 * ((r - g) / safe) + 240.0)[m_b]
    return hue, chroma


def _hue_weight(hue: np.ndarray, center: float, width: float = 60.0) -> np.ndarray:
    distance = np.abs((hue - center + 180.0) % 360.0 - 180.0)
    weight = np.clip(1.0 - distance / width, 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)   # smoothstep window


def _tint_vector(hue_deg: float) -> np.ndarray:
    """Unit-chroma RGB direction for a hue, made zero-sum so adding it keeps
    the (flat-weight) luminance approximately unchanged."""
    h = float(hue_deg) % 360.0
    x = 1.0 - abs((h / 60.0) % 2.0 - 1.0)
    idx = int(h // 60.0) % 6
    rgb = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][idx]
    vec = np.array(rgb, dtype=float)
    return vec - vec.mean()


def apply_scan_finishing_display(rgb: np.ndarray, p: ScanFinishingParams) -> np.ndarray:
    """Scanner digital side, on the ENCODED (display-referred) output."""
    if not p.active or _display_is_identity(p):
        return rgb
    out = np.asarray(rgb, dtype=float)

    # levels (master + per-channel offsets); unclipped to preserve latitude
    black = float(p.black_point) + np.asarray(p.black_point_rgb, dtype=float)
    white = float(p.white_point) + np.asarray(p.white_point_rgb, dtype=float)
    if np.any(black != 0.0) or np.any(white != 1.0):
        out = (out - black) / np.maximum(white - black, 1e-3)

    gamma = max(float(p.gamma), 0.05)
    if gamma != 1.0:
        out = np.where(out > 0.0, np.power(np.fmax(out, 0.0), 1.0 / gamma), out)

    contrast = min(max(float(p.contrast), -1.0), 1.0)
    if contrast != 0.0:
        out = 0.5 + (out - 0.5) * (1.0 + contrast)

    toe = min(max(float(p.toe), 0.0), 1.0)
    if toe > 0.0:
        # C1 exponential toe: identity above w, smooth film-like lift/compress
        # below (monotone, defined for negative inputs too).
        w = 0.25 * toe
        out = np.where(out < w, w * np.exp((out - w) / w), out)

    shoulder = min(max(float(p.shoulder), 0.0), 1.0)
    if shoulder > 0.0:
        # C1 exponential shoulder: identity below 1-w, rolls smoothly to 1.
        w = 0.25 * shoulder
        knee = 1.0 - w
        out = np.where(out > knee, 1.0 - w * np.exp(-(out - knee) / w), out)

    luma = np.einsum('...c,c->...', out, _LUMA_WEIGHTS)[..., None]

    saturation = max(float(p.saturation), 0.0)
    if saturation != 1.0:
        out = luma + (out - luma) * saturation
        luma = np.einsum('...c,c->...', out, _LUMA_WEIGHTS)[..., None]

    vibrance = max(float(p.vibrance), 0.0)
    if vibrance != 1.0:
        mx = np.fmax(out.max(axis=-1), 1e-6)
        pixel_sat = np.clip((out.max(axis=-1) - out.min(axis=-1)) / mx, 0.0, 1.0)
        factor = (1.0 + (vibrance - 1.0) * (1.0 - pixel_sat))[..., None]
        out = luma + (out - luma) * factor
        luma = np.einsum('...c,c->...', out, _LUMA_WEIGHTS)[..., None]

    hue_scales = tuple(float(v) for v in p.hue_saturation_ryg) + tuple(
        float(v) for v in p.hue_saturation_cbm)
    if hue_scales != (1.0,) * 6:
        hue, chroma = _rgb_hue_chroma(out)
        weight_sum = np.zeros_like(hue)
        scale_sum = np.zeros_like(hue)
        for center, scale in zip(_HUE_CENTERS_RYG + _HUE_CENTERS_CBM, hue_scales):
            weight = _hue_weight(hue, center)
            weight_sum += weight
            scale_sum += weight * max(scale, 0.0)
        scale = np.where((weight_sum > 1e-9) & (chroma > 1e-9),
                         scale_sum / np.maximum(weight_sum, 1e-9), 1.0)[..., None]
        out = luma + (out - luma) * scale
        luma = np.einsum('...c,c->...', out, _LUMA_WEIGHTS)[..., None]

    shadow_strength = min(max(float(p.shadow_tint_strength), 0.0), 1.0)
    highlight_strength = min(max(float(p.highlight_tint_strength), 0.0), 1.0)
    if shadow_strength > 0.0 or highlight_strength > 0.0:
        luma_flat = np.clip(luma[..., 0], 0.0, 1.0)
        if shadow_strength > 0.0:
            mask = (1.0 - luma_flat) ** 2
            out = out + (0.25 * shadow_strength) * mask[..., None] * _tint_vector(p.shadow_tint_hue)
        if highlight_strength > 0.0:
            mask = luma_flat ** 2
            out = out + (0.25 * highlight_strength) * mask[..., None] * _tint_vector(p.highlight_tint_hue)

    # Point curve (Phase 10A), after split toning so it has the last word on
    # tonality; micro_contrast stays last so texture bites on the final tones.
    curve_lut = build_display_curve_lut(p)
    if curve_lut is not None:
        out = _apply_curve_lut(out, curve_lut)

    micro = min(max(float(getattr(p, 'micro_contrast', 0.0)), -1.0), 1.0)
    if micro != 0.0 and out.ndim == 3:
        # Scanner "texture": LUMA-ONLY unsharp at a fixed small radius —
        # equal-per-channel detail add preserves chroma (no color fringing);
        # negative values soften the grain bite instead.
        from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
        luma_flat = np.einsum('...c,c->...', out, _LUMA_WEIGHTS)
        detail = luma_flat - fast_gaussian_filter(luma_flat, MICRO_CONTRAST_SIGMA)
        out = out + micro * detail[..., None]

    return out
