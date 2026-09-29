// Per-pixel weighted add: out = a*x + b*y. Used to mix the couplers diffusion
//   correction_blurred = (1 - tail_weight)*gaussian(corr) + tail_weight*exponential(corr)
// 2D dispatch (Gotcha 3).

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
    a: f32,
    b: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn axpby(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let base = (py * params.width + px) * 3u;
    out[base] = params.a * x[base] + params.b * y[base];
    out[base + 1u] = params.a * x[base + 1u] + params.b * y[base + 1u];
    out[base + 2u] = params.a * x[base + 2u] + params.b * y[base + 2u];
}
