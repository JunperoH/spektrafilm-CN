// Task 7 elementwise transcendental kernels, built on the precision-refined
// software exp2/log2 from color.wgsl (Gotcha 5 fix: hardware pow/exp2/log2 is
// ~4.4e-4 on the target GPU; these are ~3e-7 relative).
//
// Entries (all on interleaved (H, W, 3) f32, per-channel scale c):
//   scale_log10:       out = log10(fmax(x * c[ch], 0) + 1e-10)   filming tail
//   exp10_scale_log10: out = log10(fmax(exp10(x) * c[ch], 0) + 1e-10) printing tail
//   exp10_scale:       out = exp10(x) * c[ch]                    scanning 10**
//
// exp10(x) = exp2(x * log2(10)) with the product carried as a two-float;
// log10(y) = log2(y) * log10(2) with log2 returned as a two-float pair.

struct Params {
    scale: vec4<f32>,    // per-channel c in xyz, w unused
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// ---- precise helpers (same construction as color.wgsl) ---------------------
fn precise_log2(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    var e = i32((bits >> 23u) & 0xFFu) - 127;
    var m = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421356) {
        m = m * 0.5;
        e = e + 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let t = s * (1.0 + s2 * (0.33333333 + s2 * (0.2 + s2 * (0.14285714
                + s2 * (0.11111111 + s2 * 0.09090909)))));
    let C_HI = 2.88539;        // f32(2/ln2)
    let C_LO = 3.851926e-08;   // (2/ln2) - f32(2/ln2)
    let p_hi = C_HI * t;
    let p_lo = fma(C_HI, t, -p_hi) + C_LO * t;
    let ef = f32(e);
    let hi = ef + p_hi;
    let lo = (p_hi - (hi - ef)) + p_lo;
    return vec2<f32>(hi, lo);
}

fn precise_exp2(hi: f32, lo: f32) -> f32 {
    let k = round(hi);
    let f = (hi - k) + lo;
    let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.05550411
            + f * (0.009618129 + f * (0.0013333558 + f * (0.0001540353
            + f * 1.5252734e-05))))));
    return ldexp(p, i32(k));
}

fn exp10(x: f32) -> f32 {
    let L2_10_HI = 3.321928;      // f32(log2(10))
    let L2_10_LO = 7.059537e-08;  // log2(10) - f32(log2(10))
    let y_hi = x * L2_10_HI;
    let y_lo = fma(x, L2_10_HI, -y_hi) + x * L2_10_LO;
    return precise_exp2(y_hi, y_lo);
}

fn log10_pos(y: f32) -> f32 {   // y > 0 (guaranteed by the +1e-10 floor)
    let l2 = precise_log2(y);
    let L10_2_HI = 0.30103;        // f32(log10(2))
    let L10_2_LO = -1.4320989e-08; // log10(2) - f32(log10(2))
    return fma(l2.x, L10_2_HI, fma(l2.y, L10_2_HI, l2.x * L10_2_LO));
}

// ---- entry points -----------------------------------------------------------
fn idx_ok(gid: vec3<u32>) -> bool {
    return gid.x < params.width && gid.y < params.height;
}

@compute @workgroup_size(8, 8)
fn scale_log10(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!idx_ok(gid)) { return; }
    let base = (gid.y * params.width + gid.x) * 3u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let v = max(src[base + c] * params.scale[c], 0.0) + 1e-10;
        dst[base + c] = log10_pos(v);
    }
}

@compute @workgroup_size(8, 8)
fn exp10_scale_log10(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!idx_ok(gid)) { return; }
    let base = (gid.y * params.width + gid.x) * 3u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let v = max(exp10(src[base + c]) * params.scale[c], 0.0) + 1e-10;
        dst[base + c] = log10_pos(v);
    }
}

@compute @workgroup_size(8, 8)
fn exp10_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!idx_ok(gid)) { return; }
    let base = (gid.y * params.width + gid.x) * 3u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        dst[base + c] = exp10(src[base + c]) * params.scale[c];
    }
}
