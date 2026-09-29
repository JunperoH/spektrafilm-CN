"""GPU acceleration backend for spektrafilm (Phase 2).

This package adds an optional wgpu compute backend alongside the existing
numba/numpy CPU pipeline. The CPU path remains the reference implementation
and the automatic fallback. Nothing here changes the simulation math; it ports
existing per-pixel computation to the GPU and validates parity against CPU.

Public entry points:
    - device.GPUDevice: singleton wgpu adapter/device manager with is_available.
    - backend: CPU/GPU selection + automatic CPU fallback.
    - spectral: the fused 3->81->3 spectral reduction kernel (printing + scanning).
"""
