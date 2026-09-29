"""Export the current spektrafilm look as a 3D LUT (and OCIO config).

A "look" LUT is the whole film -> print -> scan transform for the settings on
screen, baked into a 3D cube by sampling the end-to-end pipeline over a grid of
input colours. Spatial and stochastic effects (grain, halation, blur, glare)
and auto-exposure are forced off so the transform is per-pixel deterministic,
which is what a LUT must be.

This needs only the standard SimulationPipeline (no pipeline tap system), so it
works in this GPU-accelerated build. It reuses the LUT format writers under
``spektrafilm_lut_creator.formats`` and emits a minimal OCIO config that
references the written .cube.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spektrafilm_lut_creator.formats import Lut, get_format


# Formats we expose in the GUI, mapped to (registry name, file extension).
LOOK_LUT_FORMATS: dict[str, tuple[str, str]] = {
    "cube": ("cube", ".cube"),
    "3dl": ("3dl", ".3dl"),
    "hald_png": ("hald_png", ".png"),
}


@dataclass
class LookLutSpec:
    """Configuration for a look-LUT export."""

    size: int = 33
    input_color_space: str = "sRGB"
    # Input domain sampled through the pipeline, in the input colour space.
    # (0, 1) is the standard unit cube most .cube consumers expect; values are
    # scene-linear when input_cctf_decoding is False (the default).
    domain_min: float = 0.0
    domain_max: float = 1.0
    title: str = "spektrafilm look"

    def __post_init__(self) -> None:
        self.size = int(self.size)
        if not (2 <= self.size <= 129):
            raise ValueError(f"LUT size must be in [2, 129], got {self.size}")
        if not (self.domain_max > self.domain_min):
            raise ValueError("domain_max must be > domain_min")


def _deterministic_params(params, spec: LookLutSpec):
    """A copy of ``params`` forced to a per-pixel deterministic transform in the
    requested input colour space (no spatial/stochastic effects, no crop, no
    auto-exposure)."""
    p = copy.deepcopy(params)
    p.debug.deactivate_spatial_effects = True
    p.debug.deactivate_stochastic_effects = True
    p.debug.debug_mode = "off"
    p.camera.auto_exposure = False
    p.film_render.grain.active = False
    p.print_render.glare.active = False
    p.io.crop = False
    p.io.upscale_factor = 1.0
    p.io.input_color_space = spec.input_color_space
    p.io.input_cctf_decoding = False
    p.settings.preview_mode = False
    # LUT sampling is tiny; keep the LUTs off so the transform is exact.
    p.settings.use_enlarger_lut = False
    p.settings.use_scanner_lut = False
    p.settings.use_fast_stats = True
    return p


def _grid_rgb(size: int, lo: float, hi: float) -> np.ndarray:
    """(size**3, 3) grid with R varying fastest (Adobe .cube order)."""
    coords = np.linspace(lo, hi, size)
    n3 = size ** 3
    ramp = np.arange(n3)
    ir = ramp % size
    ig = (ramp // size) % size
    ib = (ramp // (size * size)) % size
    return np.stack([coords[ir], coords[ig], coords[ib]], axis=-1)


def sample_look_lut(params, spec: LookLutSpec) -> Lut:
    """Sample the end-to-end pipeline over the input cube and return a Lut.

    ``params`` is a RuntimePhotoParams (e.g. from build_params_from_state); this
    forces it deterministic and samples the film->print->scan transform. The
    output is in the pipeline's output colour space / encoding.
    """
    from spektrafilm.runtime.api import Simulator, digest_params

    p = _deterministic_params(params, spec)
    n = spec.size
    grid = _grid_rgb(n, spec.domain_min, spec.domain_max)          # (n^3, 3), R fastest
    # Sample as a squarish 2D image, NOT an (n^3, 1) column: a 1-pixel-wide
    # column of millions of rows blows the GPU's 65535-workgroups-per-dimension
    # limit (Gotcha 3) at high resolution. A C-order reshape preserves the flat
    # R-fastest ordering, so the round-trip through (n^2, n) is exact.
    image = grid.reshape(n * n, n, 3)                              # (n^2, n, 3)
    out = np.asarray(Simulator(digest_params(p, apply_stocks_specifics=True)).process(image))
    out = out[..., :3].reshape(n, n, n, 3).astype(float)           # table[b, g, r]
    out = np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0)
    return Lut(
        table=out,
        domain_min=(spec.domain_min,) * 3,
        domain_max=(spec.domain_max,) * 3,
        title=spec.title,
    )


def hald_size_ok(size: int) -> bool:
    """HaldCLUT requires a perfect-square cube resolution (16, 25, 36, 49, 64...)."""
    root = int(round(size ** 0.5))
    return root * root == size


def write_look_luts(lut: Lut, out_path: str, formats: list[str], *, header_lines=None) -> tuple[list[str], dict]:
    """Write ``lut`` in each requested format next to ``out_path`` (stem reused,
    correct extension per format). Resilient: one format failing (e.g. HaldCLUT
    needs a perfect-square resolution) does not stop the others. Returns
    ``(written_paths, {format: error_message})``."""
    base = Path(out_path)
    stem_dir = base.parent
    stem = base.stem
    written: list[str] = []
    errors: dict[str, str] = {}
    for fmt_key in formats:
        if fmt_key not in LOOK_LUT_FORMATS:
            errors[fmt_key] = f"unknown format {fmt_key!r}"
            continue
        registry_name, ext = LOOK_LUT_FORMATS[fmt_key]
        target = stem_dir / f"{stem}{ext}"
        try:
            get_format(registry_name).write(lut, target, header_lines=header_lines)
            written.append(str(target))
        except Exception as exc:  # noqa: BLE001 - report, keep writing the rest
            errors[fmt_key] = str(exc)
    return written, errors


def write_ocio_config(config_dir: str, cube_filename: str, *, colorspace_name: str = "spektrafilm look") -> str:
    """Write a minimal OCIO config (config.ocio) whose display/look colorspace
    applies ``cube_filename`` (which must sit in the same dir or a ``luts``
    subdir). Returns the config path.

    Text-only (no PyOpenColorIO dependency) -- a valid OCIO v2 config that
    Resolve / Nuke / OCIO-aware apps can load to apply the exported look.
    """
    cube = Path(cube_filename).name
    cs = colorspace_name
    config = f"""ocio_profile_version: 2

description: spektrafilm exported look. Apply the '{cs}' colorspace to a
  scene-linear image in the LUT's input colour space to get the film render.

search_path: "luts:."
strictparsing: false
luma: [0.2126, 0.7152, 0.0722]

roles:
  scene_linear: linear
  reference: linear
  color_timing: {cs}
  compositing_log: linear
  data: raw

displays:
  spektrafilm:
    - !<View> {{name: "Film look", colorspace: {cs}}}
    - !<View> {{name: "Raw", colorspace: raw}}

active_displays: [spektrafilm]
active_views: [Film look, Raw]

colorspaces:
  - !<ColorSpace>
    name: linear
    family: ""
    bitdepth: 32f
    isdata: false
    allocation: lg2
    allocationvars: [-12, 12]

  - !<ColorSpace>
    name: raw
    family: ""
    bitdepth: 32f
    isdata: true
    allocation: uniform

  - !<ColorSpace>
    name: {cs}
    family: spektrafilm
    bitdepth: 32f
    isdata: false
    allocation: uniform
    from_scene_reference: !<FileTransform> {{src: {cube}, interpolation: tetrahedral}}
"""
    path = Path(config_dir) / "config.ocio"
    path.write_text(config, encoding="utf-8")
    return str(path)
