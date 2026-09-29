// B2 grain primitive. Mirrors the poisson_binomial + use_fast_stats branch of
// model/grain.py:layer_particle_model and the samplers in utils/fast_stats.py
// (fast_poisson, fast_binomial). One thread per pixel.
//
// PARITY MODEL: statistical, not per-pixel. The ALGORITHM is mirrored exactly
// (same Knuth/Normal-approx/inversion branches and thresholds as fast_stats, so
// the output DISTRIBUTION incl. its approximation artifacts matches), but the
// random stream is the GPU's own counter-based RNG. scipy/numba streams cannot
// be reproduced on a GPU; the parity test judges moments + distribution, and
// the mean field (a deterministic function of density) matches tightly.
//
// RNG: pcg4d counter-based hash (Jarzynski & Olano 2020, "Hash Functions for
// GPU Rendering"), keyed per pixel + per-layer seed. Deterministic and
// reproducible run-to-run, matching the CPU's np.random.seed(seed) determinism.

struct Params {
    width: u32,
    height: u32,
    seed: u32,                  // per-layer seed (CPU uses seed[ch] + sl*10)
    _pad: u32,
    density_max: f32,
    n_particles_per_pixel: f32,
    grain_uniformity: f32,
    od_particle: f32,           // density_max / n_particles_per_pixel (host-precomputed)
    origin_x: u32,              // 6E tiling: tile origin in GLOBAL image coords,
    origin_y: u32,              // so the per-pixel RNG stream is tile-invariant
    _pad2: u32,                 // (origin 0 = historical behavior, bit-identical)
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read>       density: array<f32>;
@group(0) @binding(1) var<storage, read_write> grain: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

// --- counter-based RNG: 4 fresh u32 per pcg4d call, cached ---
struct Rng {
    px: u32,
    py: u32,
    seed: u32,
    counter: u32,
    cache: array<u32, 4>,
    idx: u32,                   // 0..4; 4 means the cache is spent
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
    r.idx = 4u;                 // force a refill on first draw
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

// uniform in (0,1): (u + 0.5) / 2^32, never exactly 0 (safe for log()).
fn next_f32(r: ptr<function, Rng>) -> f32 {
    return (f32(next_u32(r)) + 0.5) * 2.3283064365386963e-10;
}

// standard normal via Box-Muller; one value, two uniforms. Distribution matches
// numba's np.random.randn (the values do not, as required for statistical parity).
fn next_normal(r: ptr<function, Rng>) -> f32 {
    let u1 = next_f32(r);
    let u2 = next_f32(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// Mirrors fast_stats.fast_poisson: Knuth for lambda < 30, Normal approx otherwise.
fn sample_poisson(lam: f32, r: ptr<function, Rng>) -> f32 {
    if (lam <= 0.0) {
        return 0.0;
    }
    if (lam < 30.0) {
        let big_l = exp(-lam);
        var p = 1.0;
        var k = 0;
        // Knuth: draw until the running product <= exp(-lambda); result = draws-1.
        loop {
            p = p * next_f32(r);
            if (p <= big_l) {
                break;
            }
            k = k + 1;
            if (k > 100000) {       // safety net; statistically unreachable
                break;
            }
        }
        return f32(k);
    }
    let z = next_normal(r);
    var s = round(lam + sqrt(lam) * z);
    if (s < 0.0) {
        s = 0.0;
    }
    return s;
}

// Mirrors fast_stats.fast_binomial: direct for n<25, Normal approx for var>10,
// inversion otherwise.
fn sample_binomial(n: i32, p: f32, r: ptr<function, Rng>) -> f32 {
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return f32(n);
    }
    if (n < 25) {
        var count = 0;
        for (var i = 0; i < n; i = i + 1) {
            if (next_f32(r) < p) {
                count = count + 1;
            }
        }
        return f32(count);
    }
    let mean = f32(n) * p;
    let varv = f32(n) * p * (1.0 - p);
    if (varv > 10.0) {
        let z = next_normal(r);
        var approx = round(mean + sqrt(varv) * z);
        if (approx < 0.0) {
            approx = 0.0;
        }
        if (approx > f32(n)) {
            approx = f32(n);
        }
        return approx;
    }
    // inversion (same CDF accumulation as the CPU). Sampled on
    // q = min(p, 1-p) with the complement identity
    // Binom(n, p) == n - Binom(n, 1-p): with p near 1 the start term
    // pow(1-p, n) underflows f32 (e.g. 0.05^128 = 1e-166) and the hardware
    // pow's relative error is amplified ~n-fold in the exponent, both of
    // which bias the walk toward full saturation. On q <= 0.5 the var<=10
    // gate bounds n so pow(1-q, n) >= ~1e-13: no underflow, small pow error.
    // Distribution-identical to the CPU's f64 inversion (which never
    // underflows), so the statistical parity budgets are unchanged.
    let u = next_f32(r);
    let flip = p > 0.5;
    var q = p;
    var uu = u;
    if (flip) {
        q = 1.0 - p;
        // complement of the SAME uniform keeps the walk's quantile geometry
        uu = 1.0 - u;
    }
    var cdf = 0.0;
    var prob = pow(1.0 - q, f32(n));
    var k = 0;
    loop {
        if (!(cdf < uu) || k > n) {
            break;
        }
        cdf = cdf + prob;
        if (k < n) {
            prob = prob * (f32(n - k) / f32(k + 1)) * (q / (1.0 - q));
        }
        k = k + 1;
    }
    if (flip) {
        return f32(n - (k - 1));
    }
    return f32(k - 1);
}

@compute @workgroup_size(8, 8)
fn grain_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let i = y * params.width + x;

    // layer_particle_model arithmetic (poisson_binomial branch):
    var p = density[i] / params.density_max;
    p = clamp(p, 1e-6, 1.0 - 1e-6);
    let saturation = 1.0 - p * params.grain_uniformity * (1.0 - 1e-6);
    let lam = params.n_particles_per_pixel / saturation;

    var r = rng_init(x + params.origin_x, y + params.origin_y, params.seed);
    let seeds = sample_poisson(lam, &r);
    let cnt = sample_binomial(i32(seeds), p, &r);
    grain[i] = cnt * params.od_particle * saturation;
}
