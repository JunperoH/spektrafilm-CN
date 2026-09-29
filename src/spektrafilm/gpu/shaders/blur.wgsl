// B1 blur primitive: separable Gaussian (FIR, scipy mode='reflect') and
// Young-van Vliet 3rd-order recursive IIR, mirroring
// spektrafilm/utils/fast_gaussian_filter.py exactly (same dispatch rule,
// same taps, same coefficients, same boundary handling). Data layout is
// channel-interleaved HWC float32, matching np.ascontiguousarray(image).
//
// Entry points:
//   fir_pass   : one separable FIR pass along params.axis for one channel
//   iir_h      : YvV forward+backward sweep along rows for one channel
//   iir_v      : YvV forward+backward sweep along columns for one channel
//   copy_chan  : passthrough for sigma <= 0 channels
//   accumulate : out = w*in (params.first==1) or out += w*in (mixture sums)
//
// Recurrences are sequential by nature: iir_h parallelizes over rows,
// iir_v over columns (coalesced). FIR parallelizes per pixel.

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    channel: u32,
    radius: u32,
    axis: u32,      // 0 = vertical (rows move), 1 = horizontal (cols move)
    first: u32,     // accumulate: 1 -> overwrite, 0 -> add
    grid_x: u32,    // accumulate: X-extent (in elements) of the 2D dispatch grid (gx*256)
    b: f32,         // YvV B
    b1: f32,        // YvV b1/b0
    b2: f32,        // YvV b2/b0
    b3: f32,        // YvV b3/b0
    weight: f32,    // accumulate scale
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read>       taps: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

fn idx(x: u32, y: u32) -> u32 {
    return (y * params.width + x) * params.channels + params.channel;
}

// scipy.ndimage mode='reflect': (d c b a | a b c d | d c b a)
// Mirrors _reflect() in fast_gaussian_filter.py.
fn reflect(i_in: i32, n: i32) -> i32 {
    var i = i_in;
    if (i >= 0 && i < n) { return i; }
    if (i >= -n && i < 0) { return -i - 1; }
    if (i >= n && i < 2 * n) { return 2 * n - 1 - i; }
    let period = 2 * n;
    i = i % period;
    if (i < 0) { i = i + period; }
    if (i >= n) { i = period - 1 - i; }
    return i;
}

@compute @workgroup_size(8, 8)
fn fir_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) { return; }
    let r = i32(params.radius);
    var acc: f32 = 0.0;
    if (params.axis == 0u) {
        let n = i32(params.height);
        for (var k: i32 = -r; k <= r; k = k + 1) {
            let yy = reflect(i32(y) + k, n);
            acc = acc + src[idx(x, u32(yy))] * taps[u32(k + r)];
        }
    } else {
        let n = i32(params.width);
        for (var k: i32 = -r; k <= r; k = k + 1) {
            let xx = reflect(i32(x) + k, n);
            acc = acc + src[idx(u32(xx), y)] * taps[u32(k + r)];
        }
    }
    dst[idx(x, y)] = acc;
}

// One thread per row. Forward sweep writes dst, backward sweep re-reads dst,
// exactly like _iir_horizontal (state init = edge sample replication).
@compute @workgroup_size(64)
fn iir_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.x;
    if (y >= params.height) { return; }
    let m = params.width;
    let B = params.b; let B1 = params.b1; let B2 = params.b2; let B3 = params.b3;
    let x0 = src[idx(0u, y)];
    var w1 = x0; var w2 = x0; var w3 = x0;
    for (var j: u32 = 0u; j < m; j = j + 1u) {
        let w = B * src[idx(j, y)] + B1 * w1 + B2 * w2 + B3 * w3;
        dst[idx(j, y)] = w;
        w3 = w2; w2 = w1; w1 = w;
    }
    let xn = dst[idx(m - 1u, y)];
    var y1 = xn; var y2 = xn; var y3 = xn;
    for (var jj: i32 = i32(m) - 1; jj >= 0; jj = jj - 1) {
        let j = u32(jj);
        let v = B * dst[idx(j, y)] + B1 * y1 + B2 * y2 + B3 * y3;
        dst[idx(j, y)] = v;
        y3 = y2; y2 = y1; y1 = v;
    }
}

// One thread per column (coalesced across the warp), like _iir_vertical.
@compute @workgroup_size(64)
fn iir_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= params.width) { return; }
    let n = params.height;
    let B = params.b; let B1 = params.b1; let B2 = params.b2; let B3 = params.b3;
    let x0 = src[idx(x, 0u)];
    var s1 = x0; var s2 = x0; var s3 = x0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let w = B * src[idx(x, i)] + B1 * s1 + B2 * s2 + B3 * s3;
        dst[idx(x, i)] = w;
        s3 = s2; s2 = s1; s1 = w;
    }
    let xn = dst[idx(x, n - 1u)];
    var t1 = xn; var t2 = xn; var t3 = xn;
    for (var ii: i32 = i32(n) - 1; ii >= 0; ii = ii - 1) {
        let i = u32(ii);
        let v = B * dst[idx(x, i)] + B1 * t1 + B2 * t2 + B3 * t3;
        dst[idx(x, i)] = v;
        t3 = t2; t2 = t1; t1 = v;
    }
}

@compute @workgroup_size(8, 8)
fn copy_chan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) { return; }
    dst[idx(x, y)] = src[idx(x, y)];
}

// Whole-buffer (all channels) weighted accumulate for Gaussian mixtures.
// Dispatched over a 2D grid (gx, gy) so no dimension exceeds 65535 at full
// resolution; the flat element index is row * grid_x + col, where grid_x is
// the X-extent in elements (gx*256). See blur.py exponential().
@compute @workgroup_size(256)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.grid_x + gid.x;
    let total = params.width * params.height * params.channels;
    if (i >= total) { return; }
    if (params.first == 1u) {
        dst[i] = params.weight * src[i];
    } else {
        dst[i] = dst[i] + params.weight * src[i];
    }
}
