// B3 colour-conversion primitive. One per-pixel kernel:
//   out = clip?( cctf( M @ in ) )
// It serves both scan-stage colour calls:
//   - colour.XYZ_to_RGB (cctf off): M = the combined chromatic-adaptation +
//     XYZ->RGB matrix that `colour` itself computes (pulled host-side by
//     transforming the basis through the same call, so the matrix is exact),
//     cctf = none, clip = off.
//   - colour.RGB_to_RGB encode + clip: M = identity, cctf = sRGB OETF, clip = on.
//
// Per-pixel 3x3 matrix and transfer function -> strict max_abs parity (f32
// matvec + spow ~1e-6), NOT statistical. Mirrors colour's sRGB cctf exactly:
//   V = 12.92*L                       (L <= 0.0031308; sign preserved)
//   V = 1.055*L^(1/2.4) - 0.055       (L >  0.0031308)

struct Params {
    row0: vec4<f32>,     // M[0, 0:3] in xyz; .w = decode flag (0 none, 1 ProPhoto)
    row1: vec4<f32>,     // M[1, 0:3]; .w unused
    row2: vec4<f32>,     // M[2, 0:3]; .w unused
    width: u32,
    height: u32,
    cctf: u32,           // 0 = none/linear, 1 = sRGB encode
    clip: u32,           // 0 = no, 1 = clamp to [0,1]
};

@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

// ---------------------------------------------------------------------------
// Precision-refined software pow (Gotcha 5 fix, 2026-07-06).
// Hardware pow()/exp2(log2()) via naga/Vulkan is only ~4.4e-4 accurate on the
// target GPU. This implementation avoids the SFU entirely:
//   log2(x): mantissa/exponent split by bit twiddling, atanh-series polynomial
//            on m in [1/sqrt(2), sqrt(2)), 2/ln2 applied as a double-single
//            (hi+lo) constant -> log2 as a two-float (hi, lo) pair.
//   y = p * log2(x): two-float product via fma (Dekker), preserving the bits a
//            single f32 product loses when |y| is a few tens.
//   exp2(y): round-and-reduce to f in [-0.5, 0.5], degree-7 Taylor polynomial
//            (tail < 1e-9 rel), scaled by ldexp.
// Measured accuracy is a few f32 ULP (~3e-7 relative), vs 4.4e-4 for hw pow.
// ---------------------------------------------------------------------------

// log2(x) for x > 0 (normal floats), returned as a two-float (hi, lo) pair.
fn precise_log2(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    var e = i32((bits >> 23u) & 0xFFu) - 127;
    var m = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F800000u); // m in [1, 2)
    if (m > 1.41421356) {
        m = m * 0.5;
        e = e + 1;
    }
    // atanh series: log2(m) = (2/ln2) * (s + s^3/3 + s^5/5 + ...), s in [-0.172, 0.172]
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let t = s * (1.0 + s2 * (0.33333333 + s2 * (0.2 + s2 * (0.14285714
                + s2 * (0.11111111 + s2 * 0.09090909)))));
    let C_HI = 2.88539;        // f32(2/ln2)
    let C_LO = 3.851926e-08;   // (2/ln2) - f32(2/ln2)
    let p_hi = C_HI * t;
    let p_lo = fma(C_HI, t, -p_hi) + C_LO * t;
    let ef = f32(e);
    let hi = ef + p_hi;                      // Fast2Sum: |ef| >= |p_hi| or ef == 0
    let lo = (p_hi - (hi - ef)) + p_lo;
    return vec2<f32>(hi, lo);
}

// exp2 of a two-float exponent (hi + lo).
fn precise_exp2(hi: f32, lo: f32) -> f32 {
    let k = round(hi);
    let f = (hi - k) + lo;                   // f in [-0.5, 0.5] + tiny
    let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.05550411
            + f * (0.009618129 + f * (0.0013333558 + f * (0.0001540353
            + f * 1.5252734e-05))))));
    return ldexp(p, i32(k));
}

// x^p for x > 0 to a few f32 ULP.
fn precise_pow(x: f32, p: f32) -> f32 {
    let l = precise_log2(x);
    let y_hi = p * l.x;
    let y_lo = fma(p, l.x, -y_hi) + p * l.y;
    return precise_exp2(y_hi, y_lo);
}

fn srgb_encode(l: f32) -> f32 {
    if (l <= 0.0031308) {
        return l * 12.92;            // linear toe; sign preserved for l<0
    }
    return 1.055 * precise_pow(l, 1.0 / 2.4) - 0.055;
}

// ProPhoto RGB (ROMM) cctf DECODE, matching colour.cctf_decoding('ProPhoto RGB'):
//   L = V / 16                  (V <  16*E_t = 0.03125; covers negatives)
//   L = V^1.8                   (V >= 0.03125)
fn prophoto_decode(v: f32) -> f32 {
    if (v < 0.03125) {
        return v / 16.0;             // linear toe (and all negatives)
    }
    return precise_pow(v, 1.8);
}

@compute @workgroup_size(8, 8)
fn color_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let i = (y * params.width + x) * 3u;
    var v = vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
    if (params.row0.w > 0.5) {       // decode flag packed in the unused row0.w
        v = vec3<f32>(prophoto_decode(v.x), prophoto_decode(v.y), prophoto_decode(v.z));
    }
    var o = vec3<f32>(
        dot(params.row0.xyz, v),
        dot(params.row1.xyz, v),
        dot(params.row2.xyz, v),
    );
    if (params.cctf == 1u) {
        o = vec3<f32>(srgb_encode(o.x), srgb_encode(o.y), srgb_encode(o.z));
    }
    if (params.clip == 1u) {
        o = clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    dst[i] = o.x;
    dst[i + 1u] = o.y;
    dst[i + 2u] = o.z;
}
