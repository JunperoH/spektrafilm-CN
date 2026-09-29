// Phase 6D tail: GPU glare (model/glare.py). Two passes around a BlurGPU
// gaussian on the single-channel field:
//   glare_field: per-pixel lognormal variate exp(mu + sigma * N(0,1)) — the
//     linear-space (m, s) -> (mu, sigma) inversion is done HOST-side exactly
//     as fast_stats.fast_lognormal_from_mean_std (both are scalars: glare's
//     mean and std are constant over the frame).
//   glare_add: xyz += field * illum, with illum = illuminant_xyz / 100
//     pre-scaled host-side (the /100 commutes with the blur).
//
// RNG: the SAME counter-based pcg4d + Box-Muller as grain.wgsl, with a FIXED
// seed — statistical parity vs the CPU's unseeded np stream (like grain), and
// deliberately DETERMINISTIC run-to-run, which the CPU glare is not.

struct FieldParams {
    width: u32,
    height: u32,
    seed: u32,
    origin_x: u32,   // 6E tiling: tile origin in GLOBAL image coords so the
    mu: f32,         // RNG stream is tile-invariant (0 = historical behavior)
    sigma: f32,
    origin_y: u32,
    _pad: u32,
};

struct AddParams {
    width: u32,
    height: u32,
    _pad: u32,
    _pad2: u32,
    illum: vec4<f32>,   // illuminant_xyz / 100 in xyz, w unused
};

@group(0) @binding(0) var<uniform> fparams: FieldParams;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;

// --- counter-based RNG (verbatim from grain.wgsl) ---
struct Rng {
    px: u32,
    py: u32,
    seed: u32,
    counter: u32,
    cache: array<u32, 4>,
    idx: u32,
};

fn pcg4d(v_in: vec4<u32>) -> vec4<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x = v.x + v.y * v.w;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v.w = v.w + v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x = v.x + v.y * v.w;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v.w = v.w + v.y * v.z;
    return v;
}

fn rng_init(px: u32, py: u32, seed: u32) -> Rng {
    var r: Rng;
    r.px = px;
    r.py = py;
    r.seed = seed;
    r.counter = 0u;
    r.cache = array<u32, 4>(0u, 0u, 0u, 0u);
    r.idx = 4u;
    return r;
}

fn next_u32(r: ptr<function, Rng>) -> u32 {
    if ((*r).idx >= 4u) {
        let h = pcg4d(vec4<u32>((*r).px, (*r).py, (*r).seed, (*r).counter));
        (*r).cache = array<u32, 4>(h.x, h.y, h.z, h.w);
        (*r).counter = (*r).counter + 1u;
        (*r).idx = 0u;
    }
    let out = (*r).cache[(*r).idx];
    (*r).idx = (*r).idx + 1u;
    return out;
}

fn next_f32(r: ptr<function, Rng>) -> f32 {
    return (f32(next_u32(r)) + 0.5) * 2.3283064365386963e-10;
}

fn next_normal(r: ptr<function, Rng>) -> f32 {
    let u1 = next_f32(r);
    let u2 = next_f32(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(8, 8)
fn glare_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= fparams.width || y >= fparams.height) {
        return;
    }
    var r = rng_init(x + fparams.origin_x, y + fparams.origin_y, fparams.seed);
    let z = next_normal(&r);
    field[y * fparams.width + x] = exp(fparams.mu + fparams.sigma * z);
}

@group(0) @binding(2) var<uniform> aparams: AddParams;
@group(0) @binding(3) var<storage, read> gfield: array<f32>;
@group(0) @binding(4) var<storage, read_write> xyz: array<f32>;

@compute @workgroup_size(8, 8)
fn glare_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= aparams.width || y >= aparams.height) {
        return;
    }
    let pix = y * aparams.width + x;
    let g = gfield[pix];
    let i = pix * 3u;
    xyz[i] = xyz[i] + g * aparams.illum.x;
    xyz[i + 1u] = xyz[i + 1u] + g * aparams.illum.y;
    xyz[i + 2u] = xyz[i + 2u] + g * aparams.illum.z;
}
