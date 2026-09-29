"""White balance and tint adjustment for already-demosaiced RGB input.

This mirrors the white balance semantics of
``spektrafilm.utils.raw_file_processor`` so the Import RGB panel behaves like
the Import Raw panel: the scene white derived from the requested colour
temperature is chromatically adapted (Von Kries) to the 6504 K daylight
reference, and ``tint`` scales the green channel in linear RGB.

The implementation is intentionally self-contained (numpy + colour only) so it
does not pull in the heavy RAW stack (rawpy, exiv2, lensfunpy) when loading
plain TIFF/PNG/JPEG input. The two small helpers are duplicated from
``raw_file_processor`` on purpose: importing them from there would defeat that
isolation.
"""

from __future__ import annotations

import colour
import numpy as np

# Scene white for the tungsten preset on the RGB (demosaiced) path. Tuned to
# 4500 K -- higher than the RAW path's 3800 K -- because a rendered TIFF/JPEG
# starts from a more neutral base than LibRaw's daylight output, so it needs a
# warmer assumed scene white to reach the same tungsten look. (See PROGRESS /
# raw_file_processor for the RAW-side 3800 K.)
_TUNGSTEN_TEMPERATURE = 4500.0
# Scene white for the daylight preset (adapted to the 6504 K reference below).
_DAYLIGHT_TEMPERATURE = 5600.0
_DAYLIGHT_REFERENCE_TEMPERATURE = 6504.0


def _whitepoint_xyz_from_temperature(temperature: float) -> np.ndarray:
    """Convert a colour temperature to an XYZ whitepoint.

    Daylight whitepoints above 4000 K are better approximated by the CIE
    daylight locus than by a pure Planckian radiator. Warmer illuminants such
    as tungsten are modelled with the Kang 2002 Planckian approximation.
    """

    method = 'CIE Illuminant D Series' if temperature >= 4000.0 else 'Kang 2002'
    xy = colour.CCT_to_xy(np.float64(temperature), method=method)
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)


def apply_rgb_white_balance(
    rgb: np.ndarray,
    *,
    white_balance: str = 'none',
    temperature: float | None = None,
    tint: float | None = None,
    color_space: str = 'sRGB',
    cctf_encoded: bool = True,
) -> np.ndarray:
    """Apply a raw-style white balance to a demosaiced RGB image.

    Parameters
    ----------
    rgb
        RGB image, float, encoded as described by ``color_space`` and
        ``cctf_encoded``.
    white_balance
        ``'as_shot'`` (or legacy ``'none'``) returns the input untouched,
        ``'daylight'`` adapts from a 5600 K scene white, ``'tungsten'`` adapts
        from a 4500 K scene white, ``'custom'`` adapts from ``temperature`` and
        applies ``tint``.
    temperature
        Scene correlated colour temperature in kelvin for ``'custom'`` mode.
    tint
        Green channel multiplier in linear RGB for ``'custom'`` mode, 1.0 is
        neutral.
    color_space
        Name of the image colour space in ``colour.RGB_COLOURSPACES`` (the
        Import RGB input colour space).
    cctf_encoded
        Whether ``rgb`` is CCTF encoded. When True the image is decoded before
        the linear-light adjustment and re-encoded afterwards, so the output
        keeps the same encoding as the input.
    """

    if white_balance in ('none', 'as_shot'):
        return np.asarray(rgb, dtype=np.float32)
    if white_balance == 'daylight':
        target_temperature: float = _DAYLIGHT_TEMPERATURE
        target_tint: float | None = 1.0
    elif white_balance == 'tungsten':
        target_temperature = _TUNGSTEN_TEMPERATURE
        target_tint = 1.0
    elif white_balance == 'custom':
        if temperature is None:
            raise ValueError('A custom RGB white balance requires a temperature value.')
        target_temperature = float(temperature)
        target_tint = tint
    else:
        raise ValueError(f'Unsupported RGB white balance mode: {white_balance!r}')

    try:
        colourspace = colour.RGB_COLOURSPACES[color_space]
    except KeyError as exc:
        raise ValueError(f'Unknown input colour space: {color_space!r}') from exc

    linear = np.asarray(rgb, dtype=np.float64)
    if cctf_encoded:
        linear = colourspace.cctf_decoding(linear)

    scene_white_xyz = _whitepoint_xyz_from_temperature(target_temperature)
    reference_white_xyz = _whitepoint_xyz_from_temperature(_DAYLIGHT_REFERENCE_TEMPERATURE)
    if not np.allclose(scene_white_xyz, reference_white_xyz):
        scene_white_xyz = scene_white_xyz / scene_white_xyz[1]
        reference_white_xyz = reference_white_xyz / reference_white_xyz[1]
        xyz = colour.RGB_to_XYZ(
            linear,
            colourspace=colourspace,
            chromatic_adaptation_transform=None,
            apply_cctf_decoding=False,
        )
        xyz = colour.chromatic_adaptation(
            xyz,
            scene_white_xyz,
            reference_white_xyz,
            method='Von Kries',
        )
        linear = colour.XYZ_to_RGB(
            xyz,
            colourspace=colourspace,
            chromatic_adaptation_transform=None,
            apply_cctf_encoding=False,
        )

    if target_tint is not None and not np.isclose(float(target_tint), 1.0):
        linear = linear * np.array([1.0, float(target_tint), 1.0], dtype=np.float64)

    # Adaptation can push values slightly out of gamut; clip the low end so a
    # later CCTF encoding never sees negative values.
    linear = np.maximum(linear, 0.0)
    if cctf_encoded:
        linear = colourspace.cctf_encoding(linear)
    return np.asarray(linear, dtype=np.float32)


# 16-bit recovery: a display-encoded TIFF/PNG/JPEG imported into the film
# pipeline reads flatter and, especially, LESS saturated ("colour density") than
# the same scene from a RAW. Rather than invert the file's unknown rendering
# (a full transfer-curve decode overshoots badly on an already-graded file), the
# recovery applies a small, fixed, predictable punch: a modest contrast lift and
# a slightly stronger colour-density lift. Values are the defaults NAT tuned by
# eye (2026-07-06: first cut +15%/+25% read ~3x too strong; +5%/+8% still too
# strong; settled at +2%/+3%).
_RECOVERY_CONTRAST = 1.02        # +2% contrast around mid grey
_RECOVERY_COLOR_DENSITY = 1.03   # +3% colour density (saturation)
# Rec.709 luma weights for the achromatic axis of the colour-density boost.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def apply_16bit_recovery(
    rgb: np.ndarray,
    *,
    contrast: float = _RECOVERY_CONTRAST,
    color_density: float = _RECOVERY_COLOR_DENSITY,
    pivot: float = 0.5,
) -> np.ndarray:
    """Fixed contrast + colour-density lift for the Import RGB "16-bit recovery".

    This is an artistic correction, not a physical inverse of the file's
    rendering. It applies, in the file's own value domain:

    * contrast: ``out = (x - pivot) * contrast + pivot`` around mid grey,
    * colour density: ``out = luma + (x - luma) * color_density`` (saturation),

    then clips to ``[0, 1]``. Defaults are +2% contrast and +3% colour
    density. Returns float32. Idempotent-safe on already-linear/encoded data
    because it only reshapes values around the pivot and the achromatic axis.
    """
    x = np.asarray(rgb, dtype=np.float64)
    if contrast != 1.0:
        x = (x - pivot) * float(contrast) + pivot
    if color_density != 1.0:
        luma = np.tensordot(x[..., :3], _LUMA_WEIGHTS, axes=([-1], [0]))[..., None]
        x = luma + (x - luma) * float(color_density)
    x = np.clip(x, 0.0, 1.0)
    return np.asarray(x, dtype=np.float32)
