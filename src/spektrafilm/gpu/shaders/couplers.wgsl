// DIR-coupler exposure correction (per-pixel) -- GPU port of the matvec inside
// model/couplers.compute_exposure_correction_dir_couplers:
//
//   density_silver = positive ? (density_max - density_cmy) : density_cmy
//   density_silver += shift * density_silver^2        (shift = high_exposure_couplers_shift, default 0)
//   correction[m]  = sum_c density_silver[c] * couplers_matrix[c][m]
//
// couplers_matrix is pre-scaled by `amount` on the host (compute_dir_couplers_matrix
// * dir_couplers.amount). The spatial diffusion (gaussian + exponential mix) and
// the final `log_raw - correction` subtract are applied AFTER this kernel
// (diffusion = the existing blur primitive; subtract = a pointwise op), so this
// kernel emits the pre-diffusion correction only. 2D dispatch avoids the 1D
// workgroup-count limit (Gotcha 3).

struct Params {
    width: u32,
    height: u32,
    positive: u32,
    _pad: u32,
    density_max: vec4<f32>,  // .xyz = density_max per channel, .w = high_exposure shift
    m_row0: vec4<f32>,       // couplers_matrix[0, 0:3] (donor layer 0 -> receivers)
    m_row1: vec4<f32>,       // couplers_matrix[1, 0:3]
    m_row2: vec4<f32>,       // couplers_matrix[2, 0:3]
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       density_cmy: array<f32>;  // [H*W*3]
@group(0) @binding(2) var<storage, read_write> correction: array<f32>;   // [H*W*3]

@compute @workgroup_size(8, 8)
fn couplers_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    let base = (y * params.width + x) * 3u;

    var silver = vec3<f32>(density_cmy[base], density_cmy[base + 1u], density_cmy[base + 2u]);
    if (params.positive != 0u) {
        silver = params.density_max.xyz - silver;
    }
    let shift = params.density_max.w;
    if (shift != 0.0) {
        silver = silver + shift * silver * silver;
    }

    // correction[m] = sum_c silver[c] * couplers_matrix[c][m]
    let c0 = silver.x * params.m_row0.x + silver.y * params.m_row1.x + silver.z * params.m_row2.x;
    let c1 = silver.x * params.m_row0.y + silver.y * params.m_row1.y + silver.z * params.m_row2.y;
    let c2 = silver.x * params.m_row0.z + silver.y * params.m_row1.z + silver.z * params.m_row2.z;

    correction[base] = c0;
    correction[base + 1u] = c1;
    correction[base + 2u] = c2;
}
