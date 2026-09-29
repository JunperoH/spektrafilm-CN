// Density-curve interpolation: log-exposure -> density, per pixel per channel.
// GPU port of utils/fast_interp.fast_interp with a channel-specific x-axis,
// which is how model/density_curves.interpolate_exposure_to_density calls it:
//   x_axis = log_exposure[:, None] / gamma[None, :]   (shape K x 3)
//   y_vals = density_curves                            (shape K x 3)
//
// Per pixel per channel, matching fast_interp EXACTLY:
//   if xq <= xa[0]      -> y[0,c]                       (left clamp)
//   if xq >= xa[K-1]    -> y[K-1,c]                     (right clamp)
//   else low = searchsorted(xa, xq, 'right') - 1
//        t   = (xq - xa[low]) / (xa[low+1]-xa[low])     (0 if dx == 0)
//        y   = y[low,c] + t*(y[low+1,c]-y[low,c])
//
// The x-grid is precomputed on the host in f64 then cast to f32, so there is no
// in-shader divide/reciprocal that could diverge from the CPU. 2D dispatch
// (per-pixel x,y) avoids the 1D workgroup-count limit (Gotcha 3) at high res.

struct Params {
    width: u32,
    height: u32,
    k: u32,       // number of curve samples
    _pad: u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       log_raw: array<f32>;        // [H*W*3]
@group(0) @binding(2) var<storage, read>       x_grid: array<f32>;         // [K*3] = log_exposure/gamma per channel
@group(0) @binding(3) var<storage, read>       density_curves: array<f32>; // [K*3]
@group(0) @binding(4) var<storage, read_write> output: array<f32>;         // [H*W*3]

fn interp_channel(xq: f32, channel: u32) -> f32 {
    let k = params.k;
    if (k == 0u) {
        return 0.0;
    }
    let xa0 = x_grid[channel];                       // x_grid[0*3 + channel]
    let xa_last = x_grid[(k - 1u) * 3u + channel];

    if (xq <= xa0) {
        return density_curves[channel];              // y_vals[0, channel]
    }
    if (xq >= xa_last) {
        return density_curves[(k - 1u) * 3u + channel];
    }

    // searchsorted(side='right') - 1 via bisection: largest lo with x_grid[lo] <= xq.
    var lo: u32 = 0u;
    var hi: u32 = k;
    loop {
        if (lo + 1u >= hi) { break; }
        let mid = (lo + hi) / 2u;
        let xa_mid = x_grid[mid * 3u + channel];
        if (xa_mid <= xq) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let xa_lo = x_grid[lo * 3u + channel];
    let xa_hi = x_grid[(lo + 1u) * 3u + channel];
    let dx = xa_hi - xa_lo;
    let t = select(0.0, (xq - xa_lo) / dx, dx != 0.0);
    let y0 = density_curves[lo * 3u + channel];
    let y1 = density_curves[(lo + 1u) * 3u + channel];
    return y0 + t * (y1 - y0);
}

@compute @workgroup_size(8, 8)
fn density_curve_interp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let base = (y * params.width + x) * 3u;
    output[base] = interp_channel(log_raw[base], 0u);
    output[base + 1u] = interp_channel(log_raw[base + 1u], 1u);
    output[base + 2u] = interp_channel(log_raw[base + 2u], 2u);
}
