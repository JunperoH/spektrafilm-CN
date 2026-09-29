// Phase 11C GPU port (NAT 2026-07-22): the 'synthesis' inhomogeneous
// Boolean grain field, one (channel, depth-layer) field per dispatch.
// Mirrors model/grain_synthesis.py:_synthesis_channel loop-for-loop.
//
// PARITY MODEL: statistical (the house grain standard) — WGSL has no u64,
// so the counter hash is a 32-bit mix instead of the CPU's murmur64; the
// field's MEAN reproduces the input density by the same calibration and
// the texture statistics match the CPU within the documented tolerance.
// Deterministic run-to-run: everything derives from (cell/pixel/sample
// counters + seed), never from thread order.

struct Params {
    width: u32,
    height: u32,
    samples: u32,
    max_grains: u32,
    seed: u32,
    rings: i32,
    _pad0: u32,
    _pad1: u32,
    pixel_um: f32,
    mu_r: f32,
    sigma_r: f32,
    r_max: f32,
    lam_cell: f32,
    inv_cell: f32,
    cell_um: f32,
    aperture_sigma: f32,
    epsilon: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read> p_map: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_field: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const TWO_PI: f32 = 6.28318530717958647692;
const INV_LN10: f32 = 0.43429448190325176;

fn mix32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Deterministic uniform in (0, 1) from three counters + seed (the CPU's
// _hash01 contract; 32-bit mix here).
fn hash01(a: i32, b: i32, c: i32, seed: u32) -> f32 {
    let h = mix32(bitcast<u32>(a) * 0x9E3779B9u
                  ^ mix32(bitcast<u32>(b) * 0x85EBCA6Bu)
                  ^ mix32(bitcast<u32>(c) * 0xC2B2AE35u)
                  ^ mix32(seed));
    return (f32(h >> 8u) + 0.5) * (1.0 / 16777216.0);
}

// Knuth Poisson from the cell's hash stream, capped at max_count.
fn cell_poisson(ci: i32, cj: i32, lam: f32, max_count: u32, seed: u32) -> u32 {
    let limit = exp(-lam);
    var product = 1.0;
    var count = 0u;
    loop {
        if (count > max_count) { break; }
        product = product * hash01(ci, cj, i32(900000u + count), seed);
        if (product <= limit) { break; }
        count = count + 1u;
    }
    return min(count, max_count);
}

@compute @workgroup_size(8, 8, 1)
fn synthesis_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) { return; }

    let center_x = (f32(px) + 0.5) * params.pixel_um;
    let center_y = (f32(py) + 0.5) * params.pixel_um;
    var covered = 0u;

    for (var s = 0u; s < params.samples; s = s + 1u) {
        let u1 = hash01(i32(px), i32(py), i32(s * 2u), params.seed + 7u);
        let u2 = hash01(i32(px), i32(py), i32(s * 2u + 1u), params.seed + 7u);
        let radius_n = sqrt(-2.0 * log(u1));
        let sample_x = center_x + radius_n * cos(TWO_PI * u2) * params.aperture_sigma;
        let sample_y = center_y + radius_n * sin(TWO_PI * u2) * params.aperture_sigma;
        let cell_i = i32(floor(sample_x * params.inv_cell));
        let cell_j = i32(floor(sample_y * params.inv_cell));
        var hit = false;
        for (var di = -params.rings; di <= params.rings; di = di + 1) {
            if (hit) { break; }
            for (var dj = -params.rings; dj <= params.rings; dj = dj + 1) {
                if (hit) { break; }
                let ci = cell_i + di;
                let cj = cell_j + dj;
                let count = cell_poisson(ci, cj, params.lam_cell,
                                         params.max_grains, params.seed);
                for (var g = 0u; g < count; g = g + 1u) {
                    let gk = i32(g * 8u);
                    let gx = (f32(ci) + hash01(ci, cj, gk, params.seed)) * params.cell_um;
                    let gy = (f32(cj) + hash01(ci, cj, gk + 1, params.seed)) * params.cell_um;
                    // lognormal radius, quantile-clamped
                    let v1 = hash01(ci, cj, gk + 2, params.seed);
                    let v2 = hash01(ci, cj, gk + 3, params.seed);
                    let z = sqrt(-2.0 * log(v1)) * cos(TWO_PI * v2);
                    var radius = exp(params.mu_r + params.sigma_r * z);
                    radius = min(radius, params.r_max);
                    let dx = sample_x - gx;
                    let dy = sample_y - gy;
                    if (dx * dx + dy * dy > radius * radius) { continue; }
                    // develops with the LOCAL probability (thinning)
                    let gpx = clamp(i32(gx / params.pixel_um), 0, i32(params.width) - 1);
                    let gpy = clamp(i32(gy / params.pixel_um), 0, i32(params.height) - 1);
                    let p = p_map[u32(gpy) * params.width + u32(gpx)];
                    if (hash01(ci, cj, gk + 4, params.seed) < p) {
                        hit = true;
                        break;
                    }
                }
            }
        }
        if (hit) { covered = covered + 1u; }
    }

    let remainder = max(1.0 - f32(covered) / f32(params.samples), params.epsilon);
    out_field[py * params.width + px] = -log(remainder) * INV_LN10;
}
