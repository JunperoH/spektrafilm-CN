// Fused spectral reduction kernel for spektrafilm (Phase 2).
//
// Ports the per-pixel chain that dominates the full-precision (LUT-off) render:
//
//     density_spectral = A * cmy + base            // compute_density_spectral, 3 -> 81
//     light            = 10^(-density_spectral) * illum   // density_to_light, with NaN -> 0
//     out3             = light . B                  // contract 81 -> 3
//
// The (H,W,81) spectral cube is NEVER materialized: each pixel computes its 81
// bands in registers and reduces to 3 channels on the fly. This is the whole
// point of fusing - it removes the ~15.7 GB f64 (7.9 GB f32) intermediate that
// makes the CPU path memory-bound, and parallelizes the 10^(-x).
//
// The same kernel serves printing.expose (B = paper sensitivity, illum =
// enlarger illuminant) and scanning.scan (B = CMFs, illum = scan illuminant).
// Per-stage scalar post-ops (/normalization, *exposure_factor, +preflash,
// log10) are cheap and stay on the CPU/host; they are not ported here.
//
// Math is unchanged from the CPU reference. Only the dtype differs (f32 here vs
// f64 on CPU), which is what the documented parity tolerance accounts for.

struct Params {
    width  : u32,
    height : u32,
    nbands : u32,   // 81 for the default 380-780 nm @ 5 nm grid
    _pad   : u32,
};

// cmy input, flat (H*W*3), index (y*W + x)*3 + c
@group(0) @binding(0) var<storage, read>        cmy   : array<f32>;
// channel_density A, flat (nbands*3), index w*3 + k   (A[w,k])
@group(0) @binding(1) var<storage, read>        amat  : array<f32>;
// base_density, flat (nbands)
@group(0) @binding(2) var<storage, read>        base  : array<f32>;
// illuminant, flat (nbands)
@group(0) @binding(3) var<storage, read>        illum : array<f32>;
// output_matrix B, flat (nbands*3), index w*3 + c   (B[w,c])
@group(0) @binding(4) var<storage, read>        bmat  : array<f32>;
// out, flat (H*W*3)
@group(0) @binding(5) var<storage, read_write>  outp  : array<f32>;
@group(0) @binding(6) var<uniform>              params: Params;
// Per-pixel NaN mask (1 = at least one cmy channel is NaN at this pixel),
// computed on the host. PRIMARY, compiler-proof NaN guard: branching on an
// integer storage value cannot be folded by the driver's no-NaN float
// assumption, which silently defeated both `t != t` and a float-bitcast check
// on NVIDIA/Vulkan. Faithful because amat/base/illum are finite in the
// pipeline, so a NaN density can only come from cmy, and a NaN in any cmy
// channel makes every band's density NaN -> density_to_light zeroes the pixel.
@group(0) @binding(7) var<storage, read>        nanmask : array<u32>;

const LN10 : f32 = 2.302585092994046;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let lin = y * params.width + x;
    let pix = lin * 3u;

    // Primary NaN guard (see binding 7): integer branch the compiler can't fold.
    if (nanmask[lin] != 0u) {
        outp[pix + 0u] = 0.0;
        outp[pix + 1u] = 0.0;
        outp[pix + 2u] = 0.0;
        return;
    }

    let c0 = cmy[pix + 0u];
    let c1 = cmy[pix + 1u];
    let c2 = cmy[pix + 2u];

    var acc0 : f32 = 0.0;
    var acc1 : f32 = 0.0;
    var acc2 : f32 = 0.0;

    let n = params.nbands;
    for (var w : u32 = 0u; w < n; w = w + 1u) {
        let a = w * 3u;
        // density_spectral[w] = base[w] + sum_k A[w,k] * cmy[k]
        let d = base[w] + amat[a + 0u] * c0 + amat[a + 1u] * c1 + amat[a + 2u] * c2;
        // light[w] = 10^(-d) * illum[w]   (10^x == exp(x * ln10))
        var t = exp(-d * LN10) * illum[w];
        // Secondary, per-band NaN->0 for any NaN NOT sourced from cmy (the
        // host nanmask already handles cmy-sourced NaN, the only source in the
        // pipeline). Branchless integer form: zero the BIT PATTERN, never
        // 0.0 * t (which is 0 * NaN = NaN). keep = 0 iff t is NaN. This may
        // still be optimized out under relaxed float; the nanmask is the
        // guaranteed guard.
        let bits = bitcast<u32>(t);
        let keep = u32((bits & 0x7fffffffu) <= 0x7f800000u);
        t = bitcast<f32>(bits * keep);
        acc0 = acc0 + t * bmat[a + 0u];
        acc1 = acc1 + t * bmat[a + 1u];
        acc2 = acc2 + t * bmat[a + 2u];
    }

    outp[pix + 0u] = acc0;
    outp[pix + 1u] = acc1;
    outp[pix + 2u] = acc2;
}
