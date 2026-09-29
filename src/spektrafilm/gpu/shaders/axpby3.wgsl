// Per-pixel weighted add with PER-CHANNEL coefficients:
//   out[..., c] = a[c] * x[..., c] + b[c] * y[..., c]
// Used by the resident halation chain, whose mixes carry per-channel weights
// (scatter_tail_weight, halation_strength are RGB triples). 2D dispatch.

struct Params {
    a: vec4<f32>,        // per-channel A in xyz, w unused
    b: vec4<f32>,        // per-channel B in xyz, w unused
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn axpby3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let base = (py * params.width + px) * 3u;
    out[base] = params.a.x * x[base] + params.b.x * y[base];
    out[base + 1u] = params.a.y * x[base + 1u] + params.b.y * y[base + 1u];
    out[base + 2u] = params.a.z * x[base + 2u] + params.b.z * y[base + 2u];
}
