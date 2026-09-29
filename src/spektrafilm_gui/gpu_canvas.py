"""Phase 6A: wgpu display canvas (feature-flagged beside napari).

Two layers, split for testability:

- :class:`GPUImageRenderer` — pure wgpu: uploads an RGB(A) numpy buffer as a
  texture and renders it as a bicubic-sampled quad (NegPy's gpu_widget shader,
  verbatim math) with fit-to-viewport + zoom/pan uniforms, into ANY target
  texture view. Headless-testable: ``render_to_array`` draws into an offscreen
  rgba8unorm texture and reads it back — at 1:1 fit the bicubic kernel is an
  exact identity, so display parity is asserted byte-for-byte.
- :class:`GPUCanvasWidget` — the PySide6 shell: a rendercanvas.pyside6
  RenderCanvas inside a QWidget, driving the renderer at the swapchain, with
  wheel zoom (anchored), drag pan, double-click reset, and background color.

The widget swaps in behind the ``SPEKTRAFILM_WGPU_CANVAS`` env flag without
removing napari (Phase 6C does the removal). It reuses the compute
:class:`spektrafilm.gpu.device.GPUDevice` singleton — one device for compute
AND display, which the 6D resident path requires.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

import numpy as np

FLAG_ENV = 'SPEKTRAFILM_WGPU_CANVAS'

# Composition guide overlays (Ctrl+O cycles): drawn by the interaction overlay
# over the displayed image rect, following zoom/pan.
GUIDE_MODES = ('off', 'grid', 'thirds', 'golden', 'diagonals', 'triangle', 'golden_spiral', 'center')
_PHI_LO = 1.0 - 1.0 / 1.618033988749895   # 0.381966...
_PHI_HI = 1.0 / 1.618033988749895         # 0.618033...


def guide_lines(mode: str, rect: tuple[float, float, float, float], orient: int = 0):
    """Line segments ((x0, y0), (x1, y1)) for a composition guide over the
    image rect (x, y, w, h) in widget px. Pure function (testable headless).

    - grid:      uniform 4x4 grid (lines at 1/4, 1/2, 3/4).
    - thirds:    rule-of-thirds grid (lines at 1/3 and 2/3).
    - golden:    golden-ratio grid (lines at 1-1/phi and 1/phi).
    - diagonals: the two main diagonals plus the four 45-degree corner
                 diagonals of the 'diagonal method'.
    - triangle:  main diagonal plus perpendiculars from the other two corners
                 (Lightroom's 'Triangle').
    - golden_spiral: a phi-growth logarithmic spiral (the golden / Fibonacci
                 spiral) anchored at the golden-section 'eye', fit to the rect.
    - center:    center cross.
    """
    x, y, w, h = rect
    if mode == 'thirds':
        fractions = (1.0 / 3.0, 2.0 / 3.0)
    elif mode == 'grid':
        fractions = (0.25, 0.5, 0.75)
    elif mode == 'golden':
        fractions = (_PHI_LO, _PHI_HI)
    elif mode == 'center':
        fractions = (0.5,)
    elif mode == 'diagonals':
        m = min(w, h)
        lines = [
            ((x, y), (x + w, y + h)),
            ((x + w, y), (x, y + h)),
        ]
        if abs(w - h) > 1e-6:  # 45-deg corner diagonals differ from the mains
            lines += [
                ((x, y), (x + m, y + m)),
                ((x + w, y), (x + w - m, y + m)),
                ((x, y + h), (x + m, y + h - m)),
                ((x + w, y + h), (x + w - m, y + h - m)),
            ]
        return lines
    elif mode == 'triangle':
        dd = w * w + h * h
        if dd <= 1e-12:
            return []
        t_a = (w * w) / dd
        t_b = (h * h) / dd
        if orient % 2 == 0:   # main diagonal TL->BR; perpendiculars from TR and BL
            return [
                ((x, y), (x + w, y + h)),
                ((x + w, y), (x + t_a * w, y + t_a * h)),
                ((x, y + h), (x + t_b * w, y + t_b * h)),
            ]
        # anti-diagonal TR->BL; perpendiculars from TL and BR
        return [
            ((x + w, y), (x, y + h)),
            ((x, y), (x + w - t_a * w, y + t_a * h)),
            ((x + w, y + h), (x + w - t_b * w, y + t_b * h)),
        ]
    elif mode == 'golden_spiral':
        return _golden_spiral_lines(x, y, w, h, orient)
    else:
        return []
    lines = []
    for f in fractions:
        lines.append(((x + f * w, y), (x + f * w, y + h)))   # vertical
        lines.append(((x, y + f * h), (x + w, y + f * h)))   # horizontal
    return lines


def _golden_spiral_lines(x, y, w, h, orient=0):
    """Polyline approximation of the golden (Fibonacci) spiral: a logarithmic
    spiral that grows by phi every quarter turn, anchored at the golden-section
    'eye' of the rect. The raw log-spiral only fits a golden-ratio rectangle,
    so the polyline is affinely FIT to the image rect (Lightroom-style stretch)
    — it touches the frame and never leaves it, at any aspect ratio. `orient`
    (0..3, Shift+O) picks the corner by mirroring the fitted spiral: 0 none,
    1 horizontal, 2 both (180 deg), 3 vertical."""
    import math
    if w <= 1e-6 or h <= 1e-6:
        return []
    phi = 1.618033988749895
    inv_phi = 1.0 / phi
    ex, ey = x + w * inv_phi, y + h * inv_phi     # golden-section eye
    r0 = math.hypot(x - ex, y - ey)               # reach toward the (x, y) corner
    if r0 <= 1e-6:
        return []
    theta0 = math.atan2(y - ey, x - ex)
    b = math.log(phi) / (math.pi / 2.0)           # phi per quarter turn
    pts = []
    steps = 256
    span = 4.25 * 2.0 * math.pi                   # ~4 turns inward
    for i in range(steps + 1):
        t = (i / steps) * span
        r = r0 * math.exp(-b * t)
        if r < 0.4:
            break
        ang = theta0 - t                          # wind inward
        pts.append((ex + r * math.cos(ang), ey + r * math.sin(ang)))
    if len(pts) < 2:
        return []
    # Fit the polyline's bounding box exactly onto the rect: in-frame at any
    # aspect (the log-spiral overshoots non-golden rects between tangencies).
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    scale_x = w / max(max_x - min_x, 1e-9)
    scale_y = h / max(max_y - min_y, 1e-9)
    pts = [(x + (px - min_x) * scale_x, y + (py - min_y) * scale_y)
           for px, py in pts]
    o = orient % 4
    flip_h = o in (1, 2)
    flip_v = o in (2, 3)
    if flip_h or flip_v:
        cx, cy = x + w / 2.0, y + h / 2.0
        pts = [((2.0 * cx - px) if flip_h else px,
                (2.0 * cy - py) if flip_v else py) for px, py in pts]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


_SHADER = """
struct RenderUniforms {
    rect: vec4<f32>,
    transform: vec4<f32> // x: zoom, y: pan_x, z: pan_y
};
@group(0) @binding(1) var<uniform> params: RenderUniforms;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0)
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
    );

    let ndc_pos = positions[in_vertex_index];
    let zoom = params.transform.x;
    let pan = params.transform.yz;

    let base_pos = vec2<f32>(
        (ndc_pos.x + 1.0) * 0.5 * params.rect.z + params.rect.x,
        (ndc_pos.y - 1.0) * 0.5 * params.rect.w + params.rect.y
    );
    let final_pos = base_pos * zoom + vec2<f32>(pan.x, -pan.y) * 2.0;

    var out: VertexOutput;
    out.pos = vec4<f32>(final_pos, 0.0, 1.0);
    out.uv = uvs[in_vertex_index];
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;

fn cubic(v: f32) -> f32 {
    let a = 0.5;
    let x = abs(v);
    if (x < 1.0) {
        return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    } else if (x < 2.0) {
        return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    }
    return 0.0;
}

fn textureSampleBicubic(uv: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(tex);
    let fdims = vec2<f32>(f32(dims.x), f32(dims.y));

    let pixel = uv * fdims - 0.5;
    let ipos = floor(pixel);
    let fpos = fract(pixel);

    var col = vec4<f32>(0.0);
    for (var y = -1; y <= 2; y++) {
        for (var x = -1; x <= 2; x++) {
            let offset = vec2<f32>(f32(x), f32(y));
            let coord = vec2<i32>(ipos + offset);
            let c = clamp(coord, vec2<i32>(0), vec2<i32>(dims) - 1);
            let weight = cubic(f32(x) - fpos.x) * cubic(f32(y) - fpos.y);
            col += textureLoad(tex, c, 0) * weight;
        }
    }
    return col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleBicubic(in.uv);
}
"""


def wgpu_canvas_enabled() -> bool:
    import os
    return os.environ.get(FLAG_ENV, '').strip() not in ('', '0', 'false', 'off')


def compute_canvas_layout(viewport: tuple[float, float],
                          image_size: tuple[int, int],
                          compare_size: Optional[tuple[int, int]] = None,
                          gap_world: float = 0.05):
    """Pure layout math (testable without a GPU): fit the image — or the
    side-by-side union image+compare — into the viewport, centered.

    World units: each image is normalized by ITS OWN long edge (napari's
    convention, so the compare layer matches the napari path). Returns a dict
    with per-quad rects in WIDGET PIXELS at zoom=1/pan=0:
    {'image': (x, y, w, h), 'compare': (x, y, w, h) | None, 'scale': px_per_world}.
    """
    view_w, view_h = float(viewport[0]), float(viewport[1])
    img_w, img_h = float(image_size[0]), float(image_size[1])
    img_long = max(img_w, img_h, 1.0)
    out_w, out_h = img_w / img_long, img_h / img_long
    if compare_size is None:
        total_w, total_h = out_w, out_h
    else:
        cmp_w_px, cmp_h_px = float(compare_size[0]), float(compare_size[1])
        cmp_long = max(cmp_w_px, cmp_h_px, 1.0)
        cmp_w, cmp_h = cmp_w_px / cmp_long, cmp_h_px / cmp_long
        total_w = out_w + gap_world + cmp_w
        total_h = max(out_h, cmp_h)
    scale = min(view_w / total_w, view_h / total_h)
    origin_x = (view_w - total_w * scale) / 2.0
    center_y = view_h / 2.0
    image_rect = (origin_x, center_y - out_h * scale / 2.0, out_w * scale, out_h * scale)
    compare_rect = None
    if compare_size is not None:
        compare_rect = (origin_x + (out_w + gap_world) * scale,
                        center_y - cmp_h * scale / 2.0,
                        cmp_w * scale, cmp_h * scale)
    return {'image': image_rect, 'compare': compare_rect, 'scale': scale}


def apply_zoom_pan_to_rect(rect, viewport, zoom: float, pan) -> tuple[float, float, float, float]:
    """Replicate the shader transform in widget pixels: NDC' = NDC*zoom +
    (pan_x, -pan_y)*2. Returns the transformed rect (x, y, w, h)."""
    view_w, view_h = float(viewport[0]), float(viewport[1])
    x, y, w, h = rect
    # rect corners to NDC (y-down widget px -> y-up NDC)
    def to_ndc(px, py):
        return (px / view_w) * 2.0 - 1.0, 1.0 - (py / view_h) * 2.0
    def to_px(nx, ny):
        return (nx + 1.0) * 0.5 * view_w, (1.0 - ny) * 0.5 * view_h
    x0, y0 = to_ndc(x, y)
    x1, y1 = to_ndc(x + w, y + h)
    zx0 = x0 * zoom + pan[0] * 2.0
    zy0 = y0 * zoom - pan[1] * 2.0
    zx1 = x1 * zoom + pan[0] * 2.0
    zy1 = y1 * zoom - pan[1] * 2.0
    px0, py0 = to_px(zx0, zy0)
    px1, py1 = to_px(zx1, zy1)
    return px0, py0, px1 - px0, py1 - py0


def _to_rgba8(image: np.ndarray) -> np.ndarray:
    """Any RGB(A) buffer (float 0-1 or uint8) -> contiguous (H, W, 4) uint8."""
    data = np.asarray(image)
    if data.ndim != 3 or data.shape[2] not in (3, 4):
        raise ValueError(f'expected (H, W, 3|4) image, got {data.shape}')
    if np.issubdtype(data.dtype, np.floating):
        data = (np.clip(data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.shape[2] == 3:
        rgba = np.empty((*data.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = data
        rgba[..., 3] = 255
        data = rgba
    return np.ascontiguousarray(data)


class GPUImageRenderer:
    """Pure-wgpu image quad renderer (no Qt). One per target format."""

    def __init__(self, device: Any, target_format: str):
        import wgpu

        self._wgpu = wgpu
        self.device = device
        self.target_format = target_format
        self._texture = None
        self._compare_texture = None
        self.image_size: tuple[int, int] = (1, 1)
        self.compare_size: Optional[tuple[int, int]] = None

        self.uniform_buffer = device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.compare_uniform_buffer = device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        shader = device.create_shader_module(code=_SHADER)
        self.bind_group_layout = device.create_bind_group_layout(entries=[
            {
                'binding': 0,
                'visibility': wgpu.ShaderStage.FRAGMENT,
                'texture': {
                    'sample_type': wgpu.TextureSampleType.unfilterable_float,
                    'view_dimension': wgpu.TextureViewDimension.d2,
                },
            },
            {
                'binding': 1,
                'visibility': wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                'buffer': {'type': wgpu.BufferBindingType.uniform, 'min_binding_size': 32},
            },
        ])
        self.pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout]),
            vertex={'module': shader, 'entry_point': 'vs_main'},
            primitive={
                'topology': wgpu.PrimitiveTopology.triangle_strip,
                'strip_index_format': wgpu.IndexFormat.uint32,
            },
            fragment={'module': shader, 'entry_point': 'fs_main',
                      'targets': [{'format': target_format}]},
        )

    # -- content -------------------------------------------------------------
    def _upload(self, image: np.ndarray, texture, current_size):
        wgpu = self._wgpu
        rgba = _to_rgba8(image)
        height, width = rgba.shape[:2]
        if texture is None or current_size != (width, height):
            if texture is not None:
                texture.destroy()
            texture = self.device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
        self.device.queue.write_texture(
            {'texture': texture},
            rgba,
            {'bytes_per_row': width * 4, 'rows_per_image': height},
            (width, height, 1),
        )
        return texture, (width, height)

    def set_image(self, image: np.ndarray) -> None:
        self._texture, self.image_size = self._upload(image, self._texture, self.image_size)

    def set_compare_image(self, image: np.ndarray) -> None:
        self._compare_texture, self.compare_size = self._upload(
            image, self._compare_texture, self.compare_size)

    def clear_compare_image(self) -> None:
        if self._compare_texture is not None:
            self._compare_texture.destroy()
        self._compare_texture = None
        self.compare_size = None

    def clear_image(self) -> None:
        if self._texture is not None:
            self._texture.destroy()
        self._texture = None
        self.image_size = (1, 1)

    @property
    def has_image(self) -> bool:
        return self._texture is not None

    def layout(self, viewport: tuple[float, float]):
        """Current zoom-1 layout (widget px) — shared with the interaction
        code so the mouse math and the shader agree by construction."""
        return compute_canvas_layout(
            viewport, self.image_size,
            self.compare_size if self._compare_texture is not None else None)

    # -- drawing -------------------------------------------------------------
    def draw(self, target_view: Any, viewport: tuple[int, int], *,
             zoom: float = 1.0, pan: tuple[float, float] = (0.0, 0.0),
             background: tuple[float, float, float] = (0.02, 0.02, 0.02)) -> None:
        wgpu = self._wgpu
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=[{
            'view': target_view,
            'load_op': wgpu.LoadOp.clear,
            'store_op': wgpu.StoreOp.store,
            'clear_value': (*background, 1),
        }])
        if self._texture is not None:
            view_w, view_h = float(viewport[0]), float(viewport[1])
            layout = self.layout(viewport)

            def _pack_rect(rect):
                x, y, w, h = rect
                return struct.pack(
                    'ffffffff',
                    (x / view_w) * 2.0 - 1.0,
                    1.0 - (y / view_h) * 2.0,
                    (w / view_w) * 2.0,
                    (h / view_h) * 2.0,
                    float(zoom), float(pan[0]), float(pan[1]), 0.0,
                )

            def _quad(texture, uniform_buffer, rect):
                self.device.queue.write_buffer(uniform_buffer, 0, _pack_rect(rect))
                bind_group = self.device.create_bind_group(
                    layout=self.bind_group_layout,
                    entries=[
                        {'binding': 0, 'resource': texture.create_view()},
                        {'binding': 1, 'resource': {'buffer': uniform_buffer,
                                                    'offset': 0, 'size': 32}},
                    ])
                render_pass.set_pipeline(self.pipeline)
                render_pass.set_bind_group(0, bind_group)
                render_pass.draw(4, 1, 0, 0)

            _quad(self._texture, self.uniform_buffer, layout['image'])
            if self._compare_texture is not None and layout['compare'] is not None:
                _quad(self._compare_texture, self.compare_uniform_buffer, layout['compare'])
        render_pass.end()
        self.device.queue.submit([encoder.finish()])

    def render_to_array(self, width: int, height: int, *,
                        zoom: float = 1.0, pan: tuple[float, float] = (0.0, 0.0),
                        background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Headless draw into an offscreen rgba8unorm texture; returns
        (height, width, 4) uint8 — the display-parity instrument."""
        wgpu = self._wgpu
        target = self.device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.draw(target.create_view(), (width, height),
                  zoom=zoom, pan=pan, background=background)
        bytes_per_row = (width * 4 + 255) // 256 * 256  # buffer copies need 256 alignment
        readback = self.device.create_buffer(
            size=bytes_per_row * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {'texture': target},
            {'buffer': readback, 'bytes_per_row': bytes_per_row, 'rows_per_image': height},
            (width, height, 1),
        )
        self.device.queue.submit([encoder.finish()])
        readback.map_sync(mode=wgpu.MapMode.READ)
        try:
            raw = np.frombuffer(readback.read_mapped(), dtype=np.uint8)
        finally:
            readback.unmap()
        rows = raw.reshape(height, bytes_per_row)[:, :width * 4]
        result = rows.reshape(height, width, 4).copy()
        target.destroy()
        return result


class GPUCanvasWidget:
    """PySide6 shell around GPUImageRenderer (constructed lazily so importing
    this module never requires Qt). Call :func:`create_gpu_canvas_widget`."""


def create_cpu_canvas_widget(parent=None):
    """Software display fallback for machines with NO compatible GPU (Phase 6
    non-negotiable: the app must still launch and run). A fit-scaled QLabel
    with the same public API subset the controller drives; interaction is
    reduced (no pan/zoom, crop by numeric entry only)."""
    from qtpy import QtCore, QtGui, QtWidgets

    class _CpuCanvas(QtWidgets.QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet('background-color: #1C1C1C;')
            self.setMinimumSize(64, 64)
            self._pixmap = None
            self.zoom = 1.0
            self.pan = [0.0, 0.0]
            self.crop_mode = 'off'
            self.guide_mode = 'off'
            self.guide_orient = 0
            self.content_inset = (0.0, 0.0)
            self.crop_frame_uv = None
            self.renderer = SimpleNamespaceRenderer()

        def set_image(self, image) -> None:
            rgba = _to_rgba8(image)
            height, width = rgba.shape[:2]
            self.renderer.image_size = (width, height)
            qimage = _qimage_from_rgba(rgba)
            self._pixmap = QtGui.QPixmap.fromImage(qimage)
            self._rescale()

        def clear_image(self) -> None:
            self._pixmap = None
            self.renderer.image_size = (1, 1)
            self.clear()

        def set_compare(self, image) -> None:  # single-image fallback
            return None

        def clear_compare(self) -> None:
            return None

        def set_background_hex(self, color: str) -> None:
            self.setStyleSheet(f'background-color: {color};')

        def reset_view(self) -> None:
            self._rescale()

        def set_zoom(self, zoom: float) -> None:
            return None

        def set_zoom_percent(self, percent: float) -> None:
            return None

        def arm_crop_tool(self, mode: str, **handlers) -> None:
            self.crop_mode = 'off'   # interactive crop needs the GPU canvas

        def set_crop_frame(self, rect_uv) -> None:
            return None              # the LR frame needs the GPU canvas overlay

        def set_guide_mode(self, mode: str) -> None:
            return None              # guides need the GPU canvas overlay

        def cycle_guide(self) -> str:
            return 'off'

        def set_guide_orient(self, orient: int) -> None:
            return None

        def flip_guide(self) -> int:
            return 0

        def set_content_inset(self, fx: float, fy: float) -> None:
            self.content_inset = (max(float(fx), 0.0), max(float(fy), 0.0))

        def request_redraw(self) -> None:
            self._rescale()

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._rescale()

        def _rescale(self) -> None:
            if self._pixmap is None:
                return
            self.setPixmap(self._pixmap.scaled(
                self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    class SimpleNamespaceRenderer:
        image_size = (1, 1)
        compare_size = None

    def _qimage_from_rgba(rgba):
        height, width = rgba.shape[:2]
        return QtGui.QImage(bytes(rgba.data), width, height, width * 4,
                            QtGui.QImage.Format_RGBA8888).copy()

    return _CpuCanvas(parent)


def create_gpu_canvas_widget(parent=None):
    """Build the Qt widget (rendercanvas.pyside6). Returns None if the GPU
    device is unavailable — callers fall back to napari."""
    from qtpy import QtCore, QtWidgets
    from rendercanvas.pyside6 import RenderCanvas

    from spektrafilm.gpu.device import GPUDevice

    gpu = GPUDevice.get()
    if gpu.device is None or gpu.adapter is None:
        return None

    class _InteractionOverlay(QtWidgets.QWidget):
        """Transparent surface on top of the RenderCanvas (which consumes Qt
        mouse events itself — NegPy's CanvasOverlay pattern): handles wheel
        zoom / drag pan, and the crop rubber band when a tool is armed,
        painting the scrim + white rectangle."""

        def __init__(self, host):
            super().__init__(host)
            self._host = host
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            self._pan_origin = None
            self._rubber = None      # (x0, y0, x1, y1) widget px while drawing
            self._move_origin = None
            self._frame_drag = None  # (hit, last_x, last_y) while dragging the LR frame
            self._line = None        # (x0, y0, x1, y1) widget px straighten line

        # -- painting -------------------------------------------------------
        def paintEvent(self, event) -> None:  # noqa: N802
            frame_active = (self._host.crop_mode == 'frame'
                            and self._host.crop_frame_uv is not None)
            if (self._rubber is None and self._line is None and not frame_active
                    and self._host.guide_mode == 'off'):
                return
            from qtpy import QtGui
            painter = QtGui.QPainter(self)
            if self._host.guide_mode != 'off' and not frame_active:
                self._paint_guides(painter)
            if frame_active:
                self._paint_frame(painter)
            if self._line is not None:
                pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1)
                painter.setPen(pen)
                painter.drawLine(QtCore.QLineF(*self._line))
            if self._rubber is None:
                return
            x0, y0, x1, y1 = self._rubber
            rect = QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1)).normalized()
            self._paint_scrim(painter, rect)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 1, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)

        def _paint_scrim(self, painter, rect) -> None:
            from qtpy import QtGui
            scrim = QtGui.QColor(0, 0, 0, 120)
            full = QtCore.QRectF(self.rect())
            for side in (
                QtCore.QRectF(full.left(), full.top(), full.width(), rect.top() - full.top()),
                QtCore.QRectF(full.left(), rect.bottom(), full.width(), full.bottom() - rect.bottom()),
                QtCore.QRectF(full.left(), rect.top(), rect.left() - full.left(), rect.height()),
                QtCore.QRectF(rect.right(), rect.top(), full.right() - rect.right(), rect.height()),
            ):
                painter.fillRect(side, scrim)

        def _paint_frame(self, painter) -> None:
            """The persistent Lightroom-style crop frame: scrim outside, solid
            border, 8 handles, and the composition guides INSIDE the frame
            (thirds by default while the tool is armed)."""
            from qtpy import QtGui
            frame = self._host.frame_rect_px()
            if frame is None:
                return
            x0, y0, x1, y1 = frame
            rect = QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1)).normalized()
            self._paint_scrim(painter, rect)
            # guides inside the crop box, following its aspect
            mode = self._host.guide_mode if self._host.guide_mode != 'off' else 'thirds'
            lines = guide_lines(mode, (rect.left(), rect.top(), rect.width(), rect.height()),
                                getattr(self._host, 'guide_orient', 0))
            if lines:
                segments = [QtCore.QLineF(QtCore.QPointF(*a), QtCore.QPointF(*b)) for a, b in lines]
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 90), 3))
                painter.drawLines(segments)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 150), 1))
                painter.drawLines(segments)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 235), 1))
            painter.drawRect(rect)
            # 8 handles: corners + edge midpoints
            xm, ym = rect.center().x(), rect.center().y()
            painter.setBrush(QtGui.QColor(255, 255, 255, 235))
            for hx, hy in ((rect.left(), rect.top()), (xm, rect.top()),
                           (rect.right(), rect.top()), (rect.right(), ym),
                           (rect.right(), rect.bottom()), (xm, rect.bottom()),
                           (rect.left(), rect.bottom()), (rect.left(), ym)):
                painter.drawRect(QtCore.QRectF(hx - 3.0, hy - 3.0, 6.0, 6.0))

        def _paint_guides(self, painter) -> None:
            """Composition guides over the displayed image rect (follows
            zoom/pan via the shared layout math). Dark underlay + light line
            so the guides read on any image content."""
            from qtpy import QtGui
            if self._host.renderer.image_size == (1, 1):
                return
            # Guides frame the PHOTO, not the baked white border around it.
            rect = self._host.content_rect_px()
            lines = guide_lines(self._host.guide_mode, rect, getattr(self._host, 'guide_orient', 0))
            if not lines:
                return
            painter.setClipRect(self.rect())
            segments = [
                QtCore.QLineF(QtCore.QPointF(*a), QtCore.QPointF(*b))
                for a, b in lines
            ]
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 110), 3))
            painter.drawLines(segments)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 190), 1))
            painter.drawLines(segments)

        # -- interaction ------------------------------------------------------
        def wheelEvent(self, event) -> None:  # noqa: N802
            delta = event.angleDelta().y() / 120.0
            self._host.set_zoom(self._host.zoom * (1.15 ** delta))

        def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
            if self._host.crop_mode == 'frame':
                self._host.emit_frame_commit()
                return
            self._host.reset_view()

        def keyPressEvent(self, event) -> None:  # noqa: N802
            if self._host.crop_mode == 'frame':
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    self._host.emit_frame_commit()
                    return
                if event.key() == QtCore.Qt.Key_Escape:
                    self._host.emit_frame_cancel()
                    return
            event.ignore()

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if event.button() != QtCore.Qt.LeftButton:
                return
            pos = event.position()
            mode = self._host.crop_mode
            if mode == 'frame':
                frame = self._host.frame_rect_px()
                if frame is not None:
                    from spektrafilm_gui.crop_frame import hit_test
                    hit = hit_test(frame, pos.x(), pos.y())
                    if hit != 'outside':
                        self._frame_drag = (hit, pos.x(), pos.y())
                        return
                self._pan_origin = (pos.x(), pos.y(), self._host.pan[0], self._host.pan[1])
            elif mode == 'line':
                self._line = (pos.x(), pos.y(), pos.x(), pos.y())
                self.update()
            elif mode == 'draw':
                self._rubber = (pos.x(), pos.y(), pos.x(), pos.y())
                self.update()
            elif mode == 'move':
                self._move_origin = (pos.x(), pos.y())
            else:
                self._pan_origin = (pos.x(), pos.y(), self._host.pan[0], self._host.pan[1])

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            pos = event.position()
            if self._frame_drag is not None:
                hit, last_x, last_y = self._frame_drag
                fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
                self._host.emit_frame_drag(hit, pos.x() - last_x, pos.y() - last_y, fine)
                self._frame_drag = (hit, pos.x(), pos.y())
                return
            if self._line is not None:
                self._line = (self._line[0], self._line[1], pos.x(), pos.y())
                self.update()
                return
            if self._rubber is not None:
                x0, y0 = self._rubber[:2]
                x1, y1 = self._host.constrain_corner(x0, y0, pos.x(), pos.y())
                self._rubber = (x0, y0, x1, y1)
                self.update()
                return
            if self._move_origin is not None:
                x0, y0 = self._move_origin
                fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
                self._host.emit_crop_move(pos.x() - x0, pos.y() - y0, fine)
                return
            if self._pan_origin is not None:
                x0, y0, px0, py0 = self._pan_origin
                width = max(self.width(), 1)
                height = max(self.height(), 1)
                self._host.pan[0] = px0 + (pos.x() - x0) / width
                self._host.pan[1] = py0 + (pos.y() - y0) / height
                self._host.request_redraw()

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802
            if self._frame_drag is not None:
                self._frame_drag = None
                self._host.emit_frame_release()
                return
            if self._line is not None:
                line, self._line = self._line, None
                self.update()
                self._host.emit_line_drawn(*line)
                return
            if self._rubber is not None:
                rubber, self._rubber = self._rubber, None
                self.update()
                self._host.emit_crop_drawn(*rubber)
                return
            if self._move_origin is not None:
                self._move_origin = None
                self._host.emit_crop_finished()
                return
            self._pan_origin = None

    class _Widget(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self.canvas = RenderCanvas(parent=self)
            layout.addWidget(self.canvas)

            self._context = self.canvas.get_context('wgpu')
            # Strip -srgb like NegPy: the pipeline output is already encoded.
            self._format = self._context.get_preferred_format(gpu.adapter).replace('-srgb', '')
            self._configure_context()
            self.renderer = GPUImageRenderer(gpu.device, self._format)

            self.zoom = 1.0
            self.pan = [0.0, 0.0]
            # Dark grey default surround (#1C1C1C, NAT 2026-07-12; was black)
            self._background = (0x1C / 255.0, 0x1C / 255.0, 0x1C / 255.0)

            # Interaction overlay ON TOP of the canvas (the canvas consumes
            # mouse events itself, so the overlay is the interaction surface).
            self.crop_mode = 'off'
            self.guide_mode = 'off'
            self.guide_orient = 0
            self.content_inset = (0.0, 0.0)
            self.crop_frame_uv = None     # (u0, v0, u1, v1) persistent LR frame
            self._crop_handlers = {}
            self.overlay = _InteractionOverlay(self)
            self.overlay.setGeometry(self.rect())
            self.overlay.raise_()

            self._resize_timer = QtCore.QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(50)
            self._resize_timer.timeout.connect(self._reconfigure)

            self.canvas.request_draw(self._draw_frame)

        # -- public API (mirrors what the controller needs) ----------------
        def set_image(self, image) -> None:
            self.renderer.set_image(image)
            self.request_redraw()

        def clear_image(self) -> None:
            self.renderer.clear_image()
            self.request_redraw()

        def set_compare(self, image) -> None:
            self.renderer.set_compare_image(image)
            self.request_redraw()

        def clear_compare(self) -> None:
            self.renderer.clear_compare_image()
            self.request_redraw()

        def set_background_hex(self, color: str) -> None:
            color = color.lstrip('#')
            r, g, b = (int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            self._background = (r, g, b)
            self.request_redraw()

        def reset_view(self) -> None:
            self.zoom = 1.0
            self.pan = [0.0, 0.0]
            self.request_redraw()

        def set_zoom(self, zoom: float) -> None:
            self.zoom = max(min(float(zoom), 64.0), 0.02)
            self.request_redraw()

        def set_zoom_percent(self, percent: float) -> None:
            """Screen-pixel semantics like the napari buttons: 100% maps one
            image pixel to one (logical) widget pixel."""
            viewport = (max(self.width(), 1), max(self.height(), 1))
            layout = self.renderer.layout(viewport)
            image_w = max(self.renderer.image_size[0], 1)
            fit_px_per_image_px = layout['image'][2] / image_w
            if fit_px_per_image_px > 0:
                self.set_zoom((percent / 100.0) / fit_px_per_image_px)

        def request_redraw(self) -> None:
            self.canvas.request_draw(self._draw_frame)
            # Guides live on the overlay and must track zoom/pan/image changes.
            if self.guide_mode != 'off':
                self.overlay.update()

        # -- composition guides -------------------------------------------------
        def set_guide_mode(self, mode: str) -> None:
            self.guide_mode = mode if mode in GUIDE_MODES else 'off'
            self.overlay.update()

        def cycle_guide(self) -> str:
            index = GUIDE_MODES.index(self.guide_mode) if self.guide_mode in GUIDE_MODES else 0
            self.set_guide_mode(GUIDE_MODES[(index + 1) % len(GUIDE_MODES)])
            return self.guide_mode

        def set_guide_orient(self, orient: int) -> None:
            self.guide_orient = int(orient) % 4
            self.overlay.update()

        def flip_guide(self) -> int:
            self.guide_orient = (int(self.guide_orient) + 1) % 4
            self.overlay.update()
            return self.guide_orient

        # -- crop tool bridge -------------------------------------------------
        def arm_crop_tool(self, mode: str, **handlers) -> None:
            """mode 'frame' | 'line' | 'draw' | 'move' | 'off'.

            'frame' (Phase 9, the LR persistent crop): handlers
              on_frame_drag(hit, du, dv, fine) with uv deltas ('inside' = move
              the image under the frame; else a handle name), on_frame_release(),
              on_frame_commit(), on_frame_cancel(). The frame itself is pushed
              by the controller via set_crop_frame.
            'line' (straighten): on_line(x0, y0, x1, y1) in widget px.
            'draw'/'move' (legacy rubber band): on_draw(u0, v0, u1, v1),
              on_move(du, dv, fine), on_finish(), aspect_ratio() -> float|None.
            """
            self.crop_mode = mode if mode in ('frame', 'line', 'draw', 'move') else 'off'
            self._crop_handlers = handlers if self.crop_mode != 'off' else {}
            if self.crop_mode == 'frame':
                self.overlay.setFocusPolicy(QtCore.Qt.StrongFocus)
                self.overlay.setFocus()
            else:
                self.overlay.setFocusPolicy(QtCore.Qt.NoFocus)
            if self.crop_mode == 'off':
                self.crop_frame_uv = None
            self.overlay.update()

        def set_crop_frame(self, rect_uv) -> None:
            """The persistent frame in normalized image uv (None hides it)."""
            self.crop_frame_uv = tuple(float(v) for v in rect_uv) if rect_uv is not None else None
            self.overlay.update()

        def frame_rect_px(self):
            """The persistent frame mapped to widget px via the content rect."""
            if self.crop_frame_uv is None:
                return None
            rx, ry, rw, rh = self.content_rect_px()
            u0, v0, u1, v1 = self.crop_frame_uv
            return (rx + u0 * rw, ry + v0 * rh, rx + u1 * rw, ry + v1 * rh)

        def emit_frame_drag(self, hit, dx_px, dy_px, fine) -> None:
            handler = self._crop_handlers.get('on_frame_drag')
            if callable(handler):
                _, _, rw, rh = self.content_rect_px()
                handler(hit, dx_px / max(rw, 1e-9), dy_px / max(rh, 1e-9), fine)

        def emit_frame_release(self) -> None:
            handler = self._crop_handlers.get('on_frame_release')
            if callable(handler):
                handler()

        def emit_frame_commit(self) -> None:
            handler = self._crop_handlers.get('on_frame_commit')
            if callable(handler):
                handler()

        def emit_frame_cancel(self) -> None:
            handler = self._crop_handlers.get('on_frame_cancel')
            if callable(handler):
                handler()

        def emit_line_drawn(self, x0, y0, x1, y1) -> None:
            handler = self._crop_handlers.get('on_line')
            if callable(handler):
                handler(x0, y0, x1, y1)

        def constrain_corner(self, x0, y0, x1, y1):
            aspect = self._crop_handlers.get('aspect_ratio')
            ratio = aspect() if callable(aspect) else None
            if ratio is None:
                return x1, y1
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) > abs(dy) * ratio:
                dx = abs(dy) * ratio * (1 if dx >= 0 else -1)
            else:
                dy = abs(dx) / ratio * (1 if dy >= 0 else -1)
            return x0 + dx, y0 + dy

        def _image_rect_px(self):
            viewport = (max(self.width(), 1), max(self.height(), 1))
            layout = self.renderer.layout(viewport)
            return apply_zoom_pan_to_rect(layout['image'], viewport, self.zoom, self.pan)

        def set_content_inset(self, fx: float, fy: float) -> None:
            """Fraction of the uploaded texture taken by a baked-in border on
            EACH side (the controller's white padding). Crop mapping and the
            composition guides work on the CONTENT rect inside it."""
            self.content_inset = (max(float(fx), 0.0), max(float(fy), 0.0))
            self.overlay.update()

        def content_rect_px(self):
            rx, ry, rw, rh = self._image_rect_px()
            fx, fy = self.content_inset
            return (rx + fx * rw, ry + fy * rh,
                    rw * max(1.0 - 2.0 * fx, 1e-9), rh * max(1.0 - 2.0 * fy, 1e-9))

        def widget_to_image_norm(self, x: float, y: float) -> tuple[float, float]:
            rx, ry, rw, rh = self.content_rect_px()
            return (x - rx) / max(rw, 1e-9), (y - ry) / max(rh, 1e-9)

        def emit_crop_drawn(self, x0, y0, x1, y1) -> None:
            handler = self._crop_handlers.get('on_draw')
            if callable(handler):
                u0, v0 = self.widget_to_image_norm(x0, y0)
                u1, v1 = self.widget_to_image_norm(x1, y1)
                handler(u0, v0, u1, v1)

        def emit_crop_move(self, dx_px, dy_px, fine) -> None:
            handler = self._crop_handlers.get('on_move')
            if callable(handler):
                _, _, rw, rh = self.content_rect_px()
                handler(dx_px / max(rw, 1e-9), dy_px / max(rh, 1e-9), fine)

        def emit_crop_finished(self) -> None:
            handler = self._crop_handlers.get('on_finish')
            if callable(handler):
                handler()

        # -- internals --------------------------------------------------------
        def _configure_context(self) -> None:
            last_error = None
            for alpha_mode in ('premultiplied', 'opaque'):
                try:
                    self._context.configure(device=gpu.device, format=self._format,
                                            alpha_mode=alpha_mode)
                    return
                except Exception as exc:  # try the next platform alpha mode
                    last_error = exc
            if last_error is not None:
                raise last_error

        def resizeEvent(self, event) -> None:  # noqa: N802
            super().resizeEvent(event)
            self.overlay.setGeometry(self.rect())
            self.overlay.raise_()
            self._resize_timer.start()

        def _reconfigure(self) -> None:
            try:
                self._configure_context()
                self.request_redraw()
            except Exception:
                pass

        def _draw_frame(self) -> None:
            try:
                current = self._context.get_current_texture()
            except Exception:
                return  # swapchain unavailable mid-resize; skip the frame
            if current is None:
                return
            self.renderer.draw(
                current.create_view(), (current.width, current.height),
                zoom=self.zoom, pan=tuple(self.pan), background=self._background)

    return _Widget(parent)
