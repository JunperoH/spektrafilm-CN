// Phase 6D tail: fused spectral-upsampling apply (rgb_to_raw_hanatos2025's
// per-pixel runtime). One kernel, mirroring the CPU op-for-op:
//   xyz = M @ rgb                      (M = colour's RGB->XYZ + CAT16, host)
//   b   = sum(xyz);  xy = xyz.xy / max(b, 1e-10)
//   tc  = tri2quad(xy)                 (y = ty/max(1-tx,1e-10); x = (1-tx)^2;
//                                       both clamped to [0,1])
//   raw = mitchell_bicubic_2d(lut, tc * (S-1)) * nan_to_num(b)
// The bicubic replicates fast_interp_lut exactly: Mitchell-Netravali
// (B=C=1/3), clamped cell (coord >= S-1 -> base=S-2, frac=1), symmetric
// reflect indexing, weight-sum normalization.

struct Params {
    row0: vec4<f32>,     // M[0, 0:3]
    row1: vec4<f32>,
    row2: vec4<f32>,
    width: u32,
    height: u32,
    lut_size: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;   // (S, S, 3) row-major
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

fn mitchell_weight(t: f32) -> f32 {
    // B = C = 1/3
    let x = abs(t);
    if (x < 1.0) {
        return (1.0 / 6.0) * ((12.0 - 9.0 * (1.0 / 3.0) - 6.0 * (1.0 / 3.0)) * x * x * x
            + (-18.0 + 12.0 * (1.0 / 3.0) + 6.0 * (1.0 / 3.0)) * x * x
            + (6.0 - 2.0 * (1.0 / 3.0)));
    } else if (x < 2.0) {
        return (1.0 / 6.0) * ((-(1.0 / 3.0) - 6.0 * (1.0 / 3.0)) * x * x * x
            + (6.0 * (1.0 / 3.0) + 30.0 * (1.0 / 3.0)) * x * x
            + (-12.0 * (1.0 / 3.0) - 48.0 * (1.0 / 3.0)) * x
            + (8.0 * (1.0 / 3.0) + 24.0 * (1.0 / 3.0)));
    }
    return 0.0;
}

fn safe_index(idx: i32, size: i32) -> i32 {
    if (idx < 0) {
        return -idx;
    } else if (idx >= size) {
        return 2 * (size - 1) - idx;
    }
    return idx;
}

// clamp coord to [0, S-1]; cell base in [0, S-2] with frac = 1 at the top end.
fn base_fraction(coord_in: f32, size: i32) -> vec2<f32> {
    let upper = f32(size - 1);
    var coord = clamp(coord_in, 0.0, upper);
    if (coord >= upper) {
        return vec2<f32>(f32(size - 2), 1.0);
    }
    let base = floor(coord);
    return vec2<f32>(base, coord - base);
}

@compute @workgroup_size(8, 8)
fn rgb_to_raw_tc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let i = (y * params.width + x) * 3u;
    let rgb = vec3<f32>(src[i], src[i + 1u], src[i + 2u]);
    let xyz = vec3<f32>(
        dot(params.row0.xyz, rgb),
        dot(params.row1.xyz, rgb),
        dot(params.row2.xyz, rgb),
    );
    var b = xyz.x + xyz.y + xyz.z;
    let tx = xyz.x / max(b, 1e-10);
    let ty = xyz.y / max(b, 1e-10);
    // tri2quad
    let qy = clamp(ty / max(1.0 - tx, 1e-10), 0.0, 1.0);
    let qx = clamp((1.0 - tx) * (1.0 - tx), 0.0, 1.0);
    if (b != b) {   // nan_to_num
        b = 0.0;
    }

    let size = i32(params.lut_size);
    let scale = f32(size - 1);
    let bfx = base_fraction(qx * scale, size);
    let bfy = base_fraction(qy * scale, size);
    let xb = i32(bfx.x);
    let yb = i32(bfy.x);
    let fx = bfx.y;
    let fy = bfy.y;

    var wx: array<f32, 4>;
    var wy: array<f32, 4>;
    wx[0] = mitchell_weight(fx + 1.0);
    wx[1] = mitchell_weight(fx);
    wx[2] = mitchell_weight(fx - 1.0);
    wx[3] = mitchell_weight(fx - 2.0);
    wy[0] = mitchell_weight(fy + 1.0);
    wy[1] = mitchell_weight(fy);
    wy[2] = mitchell_weight(fy - 1.0);
    wy[3] = mitchell_weight(fy - 2.0);

    var acc = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var a = 0; a < 4; a = a + 1) {
        let xi = safe_index(xb - 1 + a, size);
        for (var j = 0; j < 4; j = j + 1) {
            let yj = safe_index(yb - 1 + j, size);
            let weight = wx[a] * wy[j];
            weight_sum = weight_sum + weight;
            let li = u32((xi * size + yj) * 3);
            acc = acc + weight * vec3<f32>(lut[li], lut[li + 1u], lut[li + 2u]);
        }
    }
    if (weight_sum != 0.0) {
        acc = acc / weight_sum;
    }
    acc = acc * b;
    dst[i] = acc.x;
    dst[i + 1u] = acc.y;
    dst[i + 2u] = acc.z;
}
