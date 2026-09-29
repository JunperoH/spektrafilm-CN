// Phase 6D tail: 3D PCHIP LUT apply (fast_interp_lut._pchip_interp_lut_at_3d_prepared,
// op-for-op). The monotone per-axis slopes and per-cell min/max bounds are
// precomputed HOST-side by the same prepare_lut_pchip_3d the CPU path uses,
// so the GPU differs only by f32 arithmetic. Input coordinates are the
// [0,1]-normalized image channels scaled by (S-1), with the CPU's clamped
// cell semantics (coord >= S-1 -> base = S-2, frac = 1).

struct Params {
    width: u32,
    height: u32,
    size: u32,     // S: lut is (S,S,S,3); bounds are (S-1,S-1,S-1,3)
    _pad: u32,
    xmin: vec4<f32>,    // per-channel domain lower bound (normalization
    inv_range: vec4<f32>, // folded into the kernel: coord = (v-xmin)*inv_range)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read> slope_x: array<f32>;
@group(0) @binding(4) var<storage, read> slope_y: array<f32>;
@group(0) @binding(5) var<storage, read> slope_z: array<f32>;
@group(0) @binding(6) var<storage, read> cell_min: array<f32>;
@group(0) @binding(7) var<storage, read> cell_max: array<f32>;
@group(0) @binding(8) var<storage, read_write> dst: array<f32>;

fn base_fraction(coord_in: f32, size: i32) -> vec2<f32> {
    let upper = f32(size - 1);
    var coord = clamp(coord_in, 0.0, upper);
    if (coord >= upper) {
        return vec2<f32>(f32(size - 2), 1.0);
    }
    let base = floor(coord);
    return vec2<f32>(base, coord - base);
}

fn hermite(y0: f32, y1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
}

fn mix1(v0: f32, v1: f32, t: f32) -> f32 {
    return v0 + t * (v1 - v0);
}

fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    return mix1(mix1(v00, v10, tx), mix1(v01, v11, tx), ty);
}

@compute @workgroup_size(8, 8)
fn lut3d_pchip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let s = i32(params.size);
    let scale = f32(s - 1);
    let pix = (y * params.width + x) * 3u;

    let nr = (src[pix] - params.xmin.x) * params.inv_range.x;
    let ng = (src[pix + 1u] - params.xmin.y) * params.inv_range.y;
    let nb = (src[pix + 2u] - params.xmin.z) * params.inv_range.z;
    let bfr = base_fraction(nr * scale, s);
    let bfg = base_fraction(ng * scale, s);
    let bfb = base_fraction(nb * scale, s);
    let i = i32(bfr.x);
    let j = i32(bfg.x);
    let k = i32(bfb.x);
    let tr = bfr.y;
    let tg = bfg.y;
    let tb = bfb.y;

    let sm1 = s - 1;

    for (var c = 0; c < 3; c = c + 1) {
        // flat index helper: lut[(i*S + j)*S + k)*3 + c]
        let i000 = u32(((i * s + j) * s + k) * 3 + c);
        let i100 = u32((((i + 1) * s + j) * s + k) * 3 + c);
        let i010 = u32(((i * s + (j + 1)) * s + k) * 3 + c);
        let i110 = u32((((i + 1) * s + (j + 1)) * s + k) * 3 + c);
        let i001 = u32(((i * s + j) * s + (k + 1)) * 3 + c);
        let i101 = u32((((i + 1) * s + j) * s + (k + 1)) * 3 + c);
        let i011 = u32(((i * s + (j + 1)) * s + (k + 1)) * 3 + c);
        let i111 = u32((((i + 1) * s + (j + 1)) * s + (k + 1)) * 3 + c);

        let v000 = hermite(lut[i000], lut[i100], slope_x[i000], slope_x[i100], tr);
        let v010 = hermite(lut[i010], lut[i110], slope_x[i010], slope_x[i110], tr);
        let v001 = hermite(lut[i001], lut[i101], slope_x[i001], slope_x[i101], tr);
        let v011 = hermite(lut[i011], lut[i111], slope_x[i011], slope_x[i111], tr);

        let sy00 = mix1(slope_y[i000], slope_y[i100], tr);
        let sy10 = mix1(slope_y[i010], slope_y[i110], tr);
        let sy01 = mix1(slope_y[i001], slope_y[i101], tr);
        let sy11 = mix1(slope_y[i011], slope_y[i111], tr);

        let vz0 = hermite(v000, v010, sy00, sy10, tg);
        let vz1 = hermite(v001, v011, sy01, sy11, tg);

        let sz0 = bilinear(slope_z[i000], slope_z[i100], slope_z[i010], slope_z[i110], tr, tg);
        let sz1 = bilinear(slope_z[i001], slope_z[i101], slope_z[i011], slope_z[i111], tr, tg);

        var v = hermite(vz0, vz1, sz0, sz1, tb);
        let bidx = u32(((i * sm1 + j) * sm1 + k) * 3 + c);
        v = clamp(v, cell_min[bidx], cell_max[bidx]);
        dst[pix + u32(c)] = v;
    }
}
