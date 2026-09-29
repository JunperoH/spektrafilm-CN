from __future__ import annotations

from functools import lru_cache

from qtpy import QtGui

from spektrafilm_gui.theme_palette import ACCENT_COLOR_TEXT

try:
    import pyconify
except ImportError:  # pragma: no cover - exercised in runtime fallback only
    pyconify = None


HEADER_ICON_SIZE = 16

_SECTION_HEADER_ICONS = {
    'import rgb': 'tabler:photo-plus',
    'import raw': 'tabler:photo-cog',
    'input': 'tabler:arrow-big-down-lines',
    'camera': 'tabler:camera',
    'profiles': 'tabler:toilet-paper',
    'exposure control': 'tabler:exposure',
    'enlarger': 'tabler:building-lighthouse',
    'scanner': 'tabler:scan',
    'preview and crop': 'tabler:crop',
    'crop and upscale': 'tabler:crop',
    'output': 'tabler:arrow-big-down-lines',
    'export': 'tabler:arrow-big-down-lines',
    'roll': 'tabler:stack-2',
    'presets': 'tabler:adjustments-horizontal',
    'analysis': 'tabler:chart-histogram',
    'metadata': 'tabler:info-square-rounded',
    'grain': 'tabler:grain',
    'halation': 'tabler:time-duration-0',
    'couplers': 'tabler:chart-sankey',
    'glare': 'tabler:background',
    'chemistry': 'tabler:flask',
    'push / pull': 'tabler:arrows-up-down',
    'aging': 'tabler:hourglass-low',
    'frame': 'tabler:crop',
    'exposure': 'tabler:scan',
    'levels': 'tabler:adjustments',
    'response': 'tabler:contrast',
    'curve': 'tabler:vector-bezier-2',
    'color': 'tabler:palette',
    'toning': 'tabler:droplet',
    'input gamut compress': 'tabler:triangle-off',
    'output gamut compress': 'tabler:triangles',
    'preflash': 'tabler:sparkles',
    'diffusion': 'tabler:artboard',
    'spectral upsampling': 'tabler:prism-light',
    'tune': 'tabler:stroke-curved',
    'experimental': 'tabler:flask',
    'gui parameters': 'tabler:adjustments-horizontal',
    'display': 'tabler:skew-x',
    'napari layers': 'tabler:stack-2',
    'development': 'tabler:stopwatch',
    'bleach bypass': 'tabler:droplet-off',
    'paper shaping': 'tabler:wave-sine',
    'grain synthesis': 'tabler:chart-dots',
    'paper': 'tabler:texture',           # NAT 2026-07-25 (paper texture, ADVANCED)
    'defaults': 'tabler:restore',
    'locations': 'tabler:folder-open',   # NAT 2026-07-22
}


def section_header_icon_name(title: str) -> str | None:
    return _SECTION_HEADER_ICONS.get(title.strip().lower())


@lru_cache(maxsize=None)
def icon_by_name(icon_name: str, size: int = HEADER_ICON_SIZE,
                 color: str = ACCENT_COLOR_TEXT) -> QtGui.QIcon:
    """One iconify glyph as a QIcon. An empty QIcon whenever the icon service
    is unavailable or the name is unknown. Callers must treat the icon as
    decoration, never as a requirement."""
    if not icon_name or pyconify is None:
        return QtGui.QIcon()

    try:
        path = pyconify.svg_path(icon_name, color=color, width=size, height=size)
    except (OSError, TypeError, ValueError):
        return QtGui.QIcon()

    return QtGui.QIcon(str(path))


@lru_cache(maxsize=None)
def section_header_icon(title: str, size: int = HEADER_ICON_SIZE) -> QtGui.QIcon:
    icon_name = section_header_icon_name(title)
    if icon_name is None:
        return QtGui.QIcon()
    return icon_by_name(icon_name, size)