"""Phase 6C: napari is gone — this shim is the plain-Python image/layer MODEL.

The controller and ViewerLayerService only ever used a narrow slice of the
napari API (mapped in Phase 6 recon): layers with data / name / visible /
metadata / scale / translate / extent.world, a list with .selection.active and
.remove, add_image / add_shapes, camera attributes, reset_view, window (as the
carrier of our own host main-window reference for status/dialogs), and
mouse_drag_callbacks. HeadlessViewer provides exactly that slice with plain
objects; the wgpu canvas is the display (the controller mirrors buffers to it
at the display funnels).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np


class ShimExtent:
    def __init__(self, layer: 'ShimLayer'):
        self._layer = layer

    @property
    def world(self) -> np.ndarray:
        data = np.asarray(self._layer.data)
        if data.ndim < 2:
            return np.array([[0.0, 0.0], [1.0, 1.0]])
        height, width = data.shape[:2]
        scale = self._layer.scale
        translate = self._layer.translate
        minimum = np.array([float(translate[0]), float(translate[1])])
        maximum = minimum + np.array([height * float(scale[0]), width * float(scale[1])])
        return np.stack([minimum, maximum])


class ShimLayer:
    """Plain layer record. `_type_string` mirrors napari's discriminator so
    the existing duck-typed checks keep working."""

    def __init__(self, data: Any, name: str = '', kind: str = 'image', **attributes: Any):
        self._type_string = kind
        self.data = data
        self.name = name
        self.visible = True
        self.metadata: dict[str, Any] = {}
        self.scale = (1.0, 1.0)
        self.translate = (0.0, 0.0)
        self.opacity = 1.0
        self.editable = True
        for key, value in attributes.items():
            setattr(self, key, value)

    @property
    def extent(self) -> ShimExtent:
        return ShimExtent(self)

    def refresh(self) -> None:  # napari layers repaint here; the shim has no view
        return None


class ShimLayerList(list):
    def __init__(self):
        super().__init__()
        self.selection = SimpleNamespace(active=None)

    def remove(self, layer) -> None:  # type: ignore[override]
        super().remove(layer)
        if self.selection.active is layer:
            self.selection.active = None

    def move(self, src_index: int, dest_index: int) -> None:  # type: ignore[override]
        """napari LayerList.move shim: relocate the layer at src_index to
        dest_index. Matches napari's insert-at-dest semantics; dest_index may
        equal len(self) to mean 'to the top/end' (list.insert clamps)."""
        self.insert(dest_index, self.pop(src_index))


class ShimCamera:
    def __init__(self):
        self.zoom = 1.0
        self.center = (0.0, 0.0)
        self.mouse_pan = True
        self.mouse_zoom = True


class HeadlessViewer:
    """Drop-in for the narrow napari.Viewer surface the app used."""

    def __init__(self):
        self.layers = ShimLayerList()
        self.camera = ShimCamera()
        self.window = SimpleNamespace()          # carries _agx_host_window
        self.mouse_drag_callbacks: list = []
        self.status = ''
        self._reset_view_hook = None             # set to the canvas reset

    # -- layer construction -------------------------------------------------
    def add_image(self, data, *, name: str = '', **attributes) -> ShimLayer:
        layer = ShimLayer(data, name=name, kind='image', **attributes)
        self.layers.append(layer)
        self.layers.selection.active = layer
        return layer

    def add_shapes(self, data, *, name: str = '', **attributes) -> ShimLayer:
        layer = ShimLayer(data, name=name, kind='shapes', **attributes)
        self.layers.append(layer)
        self.layers.selection.active = layer
        return layer

    # -- camera ---------------------------------------------------------------
    def reset_view(self) -> None:
        if callable(self._reset_view_hook):
            self._reset_view_hook()
