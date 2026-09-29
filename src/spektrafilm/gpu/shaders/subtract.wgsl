// Per-pixel out = a - b  (used for log_raw_0 = log_raw - couplers_correction in
// the resident develop chain). 2D dispatch (Gotcha 3).

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn subtract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let base = (y * params.width + x) * 3u;
    out[base] = a[base] - b[base];
    out[base + 1u] = a[base + 1u] - b[base + 1u];
    out[base + 2u] = a[base + 2u] - b[base + 2u];
}
