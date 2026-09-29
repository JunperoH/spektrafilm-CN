// G2d-4 resident-grain helpers.
//
// interp_plane: one (sublayer, channel) plane of
//   model/density_curves.interp_density_cmy_layers + the density_min_layers add:
//     plane[i] = interp(sign * density[i*3 + ch], grid_x, grid_y) + dmin
//   with np.interp clamp semantics, identical bisection to density_curve.wgsl.
//   grid_x is uploaded ALREADY sign-folded and ascending (host asserts); the
//   kernel folds the query with the same sign (positive film: sign = -1).
//
// accum_plane: accumulate a plane into channel `ch` of the interleaved output:
//     accum[i*3 + ch] += plane[i]
//   1:1 thread-to-element mapping, so read_write on accum is hazard-free.

struct InterpParams {
    width: u32,
    height: u32,
    k: u32,
    ch: u32,
    sign: f32,
    dmin: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform>             ip: InterpParams;
@group(0) @binding(1) var<storage, read>       density: array<f32>;  // [H*W*3]
@group(0) @binding(2) var<storage, read>       grid_x: array<f32>;   // [K] ascending
@group(0) @binding(3) var<storage, read>       grid_y: array<f32>;   // [K]
@group(0) @binding(4) var<storage, read_write> plane: array<f32>;    // [H*W]

@compute @workgroup_size(8, 8)
fn interp_plane(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= ip.width || y >= ip.height) {
        return;
    }
    let i = y * ip.width + x;
    let xq = ip.sign * density[i * 3u + ip.ch];
    let k = ip.k;
    var v: f32;
    if (k == 0u) {
        v = 0.0;
    } else if (xq <= grid_x[0]) {
        v = grid_y[0];
    } else if (xq >= grid_x[k - 1u]) {
        v = grid_y[k - 1u];
    } else {
        var lo: u32 = 0u;
        var hi: u32 = k;
        loop {
            if (lo + 1u >= hi) { break; }
            let mid = (lo + hi) / 2u;
            if (grid_x[mid] <= xq) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let xa_lo = grid_x[lo];
        let dx = grid_x[lo + 1u] - xa_lo;
        let t = select(0.0, (xq - xa_lo) / dx, dx != 0.0);
        v = grid_y[lo] + t * (grid_y[lo + 1u] - grid_y[lo]);
    }
    plane[i] = v + ip.dmin;
}

struct AccumParams {
    width: u32,
    height: u32,
    ch: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             ap: AccumParams;
@group(0) @binding(1) var<storage, read>       src_plane: array<f32>;  // [H*W]
@group(0) @binding(2) var<storage, read_write> accum: array<f32>;      // [H*W*3]

@compute @workgroup_size(8, 8)
fn accum_plane(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= ap.width || y >= ap.height) {
        return;
    }
    let i = y * ap.width + x;
    accum[i * 3u + ap.ch] = accum[i * 3u + ap.ch] + src_plane[i];
}
