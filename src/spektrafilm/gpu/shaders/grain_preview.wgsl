// Phase 11A preview grain, GPU (2026-07-21 polish). Mirrors
// model/grain.py:apply_grain_preview (pre final-blur): per channel, Gaussian
// noise whose per-pixel std is the production model's closed form
//     std = od_particle * sqrt(n * p * (1 - p * u))
// One thread per pixel, all three channels in one dispatch.
//
// PARITY MODEL: statistical (house grain standard). The CPU uses seeded
// numpy normals; the GPU uses its own pcg4d counter stream (same RNG family
// as grain.wgsl), keyed per pixel/channel/seed — deterministic run-to-run.

struct Params {
    width: u32,
    height: u32,
    seed: u32,
    monochrome: u32,            // 1 = one shared noise draw across channels
    density_max: vec4<f32>,     // per channel (w unused)
    n_particles: vec4<f32>,
    uniformity: vec4<f32>,
    density_min: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       density: array<f32>;   // HxWx3
@group(0) @binding(1) var<storage, read_write> out: array<f32>;       // HxWx3
@group(0) @binding(2) var<uniform>             params: Params;

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

fn normal_for(px: u32, py: u32, channel: u32, seed: u32) -> f32 {
    let h = pcg4d(vec4<u32>(px, py, seed + channel * 7919u, 0u));
    let u1 = (f32(h.x) + 0.5) * 2.3283064365386963e-10;
    let u2 = (f32(h.y) + 0.5) * 2.3283064365386963e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(8, 8, 1)
fn preview_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let base = (y * params.width + x) * 3u;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let d_max = params.density_max[ch];
        let n = params.n_particles[ch];
        let u = params.uniformity[ch];
        let d_min = params.density_min[ch];
        let d = density[base + ch] + d_min;
        let p = clamp(d / d_max, 1e-6, 1.0 - 1e-6);
        let od = d_max / n;
        let sigma = od * sqrt(n * p * (1.0 - p * u));
        var draw_ch = ch;
        if (params.monochrome == 1u) {
            draw_ch = 0u;       // one shared field: identical draw per channel
        }
        let z = normal_for(x, y, draw_ch, params.seed);
        out[base + ch] = d + sigma * z - d_min;
    }
}
