// Phase 6D: SCAN-tab finishing on the GPU. One per-pixel kernel with two
// modes, mirroring model/scan_finishing.py op-for-op IN ORDER:
//   mode 0 = LINEAR group: scan exposure / analog gains, then density-domain
//            color separation (row-normalized matrix, neutrals preserved).
//   mode 1 = DISPLAY group: levels -> gamma -> contrast -> toe -> shoulder ->
//            saturation -> vibrance -> per-hue saturation -> split tints.
// Every op keeps the CPU's own skip-condition so an identity parameter is a
// true no-op, and all pow/exp/log go through the precision-refined software
// path (Gotcha 5: hardware pow is only ~4.4e-4 on the target GPU).

struct Params {
    gains: vec4<f32>,          // xyz = 2^(exposure+gain_ev); w = separation on (>0)
    sep0: vec4<f32>,           // row-normalized blended separation matrix rows
    sep1: vec4<f32>,
    sep2: vec4<f32>,
    black: vec4<f32>,          // xyz = black point per channel; w = gamma
    white: vec4<f32>,          // xyz = white point per channel; w = contrast
    knees: vec4<f32>,          // x = toe, y = shoulder, z = saturation, w = vibrance
    hue_a: vec4<f32>,          // per-hue scales R, Y, G, C (clamped >= 0 host-side)
    hue_b: vec4<f32>,          // x = B, y = M scales; z = shadow amt (0.25*strength), w = highlight amt
    shadow_tint: vec4<f32>,    // zero-sum tint vector, w unused
    highlight_tint: vec4<f32>,
    width: u32,
    height: u32,
    mode: u32,                 // 0 = linear group, 1 = display group
    curve_n: u32,              // LUT entries/channel in curve_lut; 0 = no curve
};

@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
// Phase 10A finishing point curve: three per-channel 1D LUTs built host-side
// (luma-then-per-channel composed at build time), interleaved [i*3 + c].
@group(0) @binding(9) var<storage, read>       curve_lut: array<f32>;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);   // Rec.709

// --------------------------------------------------------------------------
// Precision-refined software log2/exp2 (verbatim from color.wgsl, Gotcha 5).
// --------------------------------------------------------------------------
fn precise_log2(x: f32) -> vec2<f32> {
    let bits = bitcast<u32>(x);
    var e = i32((bits >> 23u) & 0xFFu) - 127;
    var m = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F800000u); // m in [1, 2)
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

fn precise_pow(x: f32, p: f32) -> f32 {
    let l = precise_log2(x);
    let y_hi = p * l.x;
    let y_lo = fma(p, l.x, -y_hi) + p * l.y;
    return precise_exp2(y_hi, y_lo);
}

// e^x via the software exp2 (double-single log2(e) so nothing is lost when
// |x| is a few tens, as in the toe/shoulder knees).
fn precise_exp(x: f32) -> f32 {
    let K_HI = 1.442695;       // f32(log2 e)
    let K_LO = 1.9259630e-08;  // log2(e) - f32(log2 e)
    let y_hi = x * K_HI;
    let y_lo = fma(x, K_HI, -y_hi) + x * K_LO;
    return precise_exp2(y_hi, y_lo);
}

// log10(x) for x > 0.
fn precise_log10(x: f32) -> f32 {
    let l = precise_log2(x);
    return (l.x + l.y) * 0.30103001;   // log10(2)
}

// 10^x.
fn precise_exp10(x: f32) -> f32 {
    let K_HI = 3.3219281;      // f32(log2 10)
    let K_LO = 7.0595370e-08;  // log2(10) - f32(log2 10)
    let y_hi = x * K_HI;
    let y_lo = fma(x, K_HI, -y_hi) + x * K_LO;
    return precise_exp2(y_hi, y_lo);
}

// Positive modulo (numpy's %), for the hue wrap.
fn pmod360(x: f32) -> f32 {
    return x - 360.0 * floor(x / 360.0);
}

// smoothstep window around a hue center, width 60 deg (CPU _hue_weight).
fn hue_weight(hue: f32, center: f32) -> f32 {
    let distance = abs(pmod360(hue - center + 180.0) - 180.0);
    let w = clamp(1.0 - distance / 60.0, 0.0, 1.0);
    return w * w * (3.0 - 2.0 * w);
}

// -------------------------------------------------------------- linear group
fn apply_linear(v_in: vec3<f32>) -> vec3<f32> {
    var v = v_in;
    if (any(params.gains.xyz != vec3<f32>(1.0))) {
        v = v * params.gains.xyz;
    }
    if (params.gains.w > 0.0) {
        // Density-domain unmixing; the row-normalized matrix comes from the
        // host so neutrals are preserved to the same digits as the CPU.
        let clamped = max(v, vec3<f32>(1e-6));
        let d = vec3<f32>(
            -precise_log10(clamped.x),
            -precise_log10(clamped.y),
            -precise_log10(clamped.z),
        );
        let dm = vec3<f32>(
            dot(params.sep0.xyz, d),
            dot(params.sep1.xyz, d),
            dot(params.sep2.xyz, d),
        );
        let unmixed = vec3<f32>(
            precise_exp10(-dm.x),
            precise_exp10(-dm.y),
            precise_exp10(-dm.z),
        );
        v = select(v, unmixed, v > vec3<f32>(1e-6));
    }
    return v;
}

// ------------------------------------------------------------- display group
fn rgb_hue_chroma(v: vec3<f32>) -> vec2<f32> {
    let mx = max(v.x, max(v.y, v.z));
    let mn = min(v.x, min(v.y, v.z));
    let chroma = mx - mn;
    let safe = select(1.0, chroma, chroma > 1e-9);
    var hue = 0.0;
    if (mx == v.x) {
        hue = pmod360(60.0 * ((v.y - v.z) / safe));
    } else if (mx == v.y) {
        hue = 60.0 * ((v.z - v.x) / safe) + 120.0;
    } else {
        hue = 60.0 * ((v.x - v.y) / safe) + 240.0;
    }
    return vec2<f32>(hue, chroma);
}

fn apply_display(v_in: vec3<f32>) -> vec3<f32> {
    var v = v_in;

    let black = params.black.xyz;
    let white = params.white.xyz;
    if (any(black != vec3<f32>(0.0)) || any(white != vec3<f32>(1.0))) {
        v = (v - black) / max(white - black, vec3<f32>(1e-3));
    }

    let gamma = params.black.w;
    if (gamma != 1.0) {
        let inv_g = 1.0 / gamma;
        if (v.x > 0.0) { v.x = precise_pow(v.x, inv_g); }
        if (v.y > 0.0) { v.y = precise_pow(v.y, inv_g); }
        if (v.z > 0.0) { v.z = precise_pow(v.z, inv_g); }
    }

    let contrast = params.white.w;
    if (contrast != 0.0) {
        v = 0.5 + (v - 0.5) * (1.0 + contrast);
    }

    let toe = params.knees.x;
    if (toe > 0.0) {
        let w = 0.25 * toe;
        let lifted = w * vec3<f32>(
            precise_exp((v.x - w) / w),
            precise_exp((v.y - w) / w),
            precise_exp((v.z - w) / w),
        );
        v = select(v, lifted, v < vec3<f32>(w));
    }

    let shoulder = params.knees.y;
    if (shoulder > 0.0) {
        let w = 0.25 * shoulder;
        let knee = 1.0 - w;
        let rolled = 1.0 - w * vec3<f32>(
            precise_exp(-(v.x - knee) / w),
            precise_exp(-(v.y - knee) / w),
            precise_exp(-(v.z - knee) / w),
        );
        v = select(v, rolled, v > vec3<f32>(knee));
    }

    var luma = dot(v, LUMA);

    let saturation = params.knees.z;
    if (saturation != 1.0) {
        v = luma + (v - luma) * saturation;
        luma = dot(v, LUMA);
    }

    let vibrance = params.knees.w;
    if (vibrance != 1.0) {
        let mxc = max(v.x, max(v.y, v.z));
        let mnc = min(v.x, min(v.y, v.z));
        let mx = max(mxc, 1e-6);
        let pixel_sat = clamp((mxc - mnc) / mx, 0.0, 1.0);
        let factor = 1.0 + (vibrance - 1.0) * (1.0 - pixel_sat);
        v = luma + (v - luma) * factor;
        luma = dot(v, LUMA);
    }

    var scales = array<f32, 6>(params.hue_a.x, params.hue_a.y, params.hue_a.z,
                               params.hue_a.w, params.hue_b.x, params.hue_b.y);
    if (params.hue_a.x != 1.0 || params.hue_a.y != 1.0 || params.hue_a.z != 1.0
        || params.hue_a.w != 1.0 || params.hue_b.x != 1.0 || params.hue_b.y != 1.0) {
        let hc = rgb_hue_chroma(v);
        var weight_sum = 0.0;
        var scale_sum = 0.0;
        for (var k = 0u; k < 6u; k = k + 1u) {
            let weight = hue_weight(hc.x, f32(k) * 60.0);
            weight_sum = weight_sum + weight;
            scale_sum = scale_sum + weight * scales[k];
        }
        var scale = 1.0;
        if (weight_sum > 1e-9 && hc.y > 1e-9) {
            scale = scale_sum / max(weight_sum, 1e-9);
        }
        v = luma + (v - luma) * scale;
        luma = dot(v, LUMA);
    }

    let shadow_amt = params.hue_b.z;
    let highlight_amt = params.hue_b.w;
    if (shadow_amt > 0.0 || highlight_amt > 0.0) {
        let luma_flat = clamp(luma, 0.0, 1.0);
        if (shadow_amt > 0.0) {
            let mask = (1.0 - luma_flat) * (1.0 - luma_flat);
            v = v + shadow_amt * mask * params.shadow_tint.xyz;
        }
        if (highlight_amt > 0.0) {
            let mask = luma_flat * luma_flat;
            v = v + highlight_amt * mask * params.highlight_tint.xyz;
        }
    }

    // Point curve (Phase 10A), after the tints, before the micro passes:
    // linear-interp lookup into the host-built table, input clamped to [0,1]
    // (flat extension) — mirrors the CPU's _apply_curve_lut exactly.
    let n = params.curve_n;
    if (n > 1u) {
        v = vec3<f32>(
            curve_sample(v.x, 0u, n),
            curve_sample(v.y, 1u, n),
            curve_sample(v.z, 2u, n),
        );
    }

    return v;
}

fn curve_sample(x: f32, c: u32, n: u32) -> f32 {
    let t = clamp(x, 0.0, 1.0) * f32(n - 1u);
    let i0 = u32(floor(t));
    let i1 = min(i0 + 1u, n - 1u);
    let f = t - f32(i0);
    let a = curve_lut[i0 * 3u + c];
    let b = curve_lut[i1 * 3u + c];
    return a + (b - a) * f;
}

// ---------------------------------------------------------------------------
// Micro-contrast ("texture"), applied LAST in the display group as its own
// passes around a BlurGPU gaussian: luma_extract -> blur -> micro_add.
// Luma-only detail added equally per channel (chroma preserved), mirroring
// the CPU tail of apply_scan_finishing_display.
// ---------------------------------------------------------------------------
struct MicroParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
    amount: f32,      // clamped -1..1 host-side
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(3) var<uniform> mparams: MicroParams;
@group(0) @binding(4) var<storage, read> mc_src: array<f32>;
@group(0) @binding(5) var<storage, read_write> mc_luma: array<f32>;

@compute @workgroup_size(8, 8)
fn luma_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= mparams.width || y >= mparams.height) {
        return;
    }
    let i = y * mparams.width + x;
    let j = i * 3u;
    mc_luma[i] = dot(vec3<f32>(mc_src[j], mc_src[j + 1u], mc_src[j + 2u]), LUMA);
}

@group(0) @binding(6) var<storage, read_write> mc_rgb: array<f32>;
@group(0) @binding(7) var<storage, read> mc_luma_sharp: array<f32>;
@group(0) @binding(8) var<storage, read> mc_luma_blur: array<f32>;

@compute @workgroup_size(8, 8)
fn micro_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= mparams.width || y >= mparams.height) {
        return;
    }
    let i = y * mparams.width + x;
    let d = mparams.amount * (mc_luma_sharp[i] - mc_luma_blur[i]);
    let j = i * 3u;
    mc_rgb[j] = mc_rgb[j] + d;
    mc_rgb[j + 1u] = mc_rgb[j + 1u] + d;
    mc_rgb[j + 2u] = mc_rgb[j + 2u] + d;
}

@compute @workgroup_size(8, 8)
fn scan_finishing(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let i = (y * params.width + x) * 3u;
    let v = vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
    var o: vec3<f32>;
    if (params.mode == 0u) {
        o = apply_linear(v);
    } else {
        o = apply_display(v);
    }
    dst[i] = o.x;
    dst[i + 1u] = o.y;
    dst[i + 2u] = o.z;
}
