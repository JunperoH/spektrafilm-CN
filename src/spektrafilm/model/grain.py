import numpy as np
import scipy
import scipy.ndimage
from spektrafilm.model.density_curves import interp_density_cmy_layers
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.utils.fast_stats import fast_binomial, fast_poisson, fast_lognormal_from_mean_std
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter

################################################################################
# Grain (very simple model)
################################################################################

def layer_particle_model(density,
                         density_max=2.2,
                         n_particles_per_pixel=10,
                         grain_uniformity=0.98,
                         seed=None,
                         blur_particle=0.0,
                         method='poisson_binomial',
                         use_fast_stats=False,
                         ):
    # B2 GPU fast path: mirrors this poisson_binomial + use_fast_stats branch on
    # the GPU (statistical parity; see spektrafilm/gpu/grain.py). Engages only for
    # the shipped use_fast_stats path with a concrete seed. Everything else (the
    # scipy full-precision reference, the gamma_beta method, the no-seed/random
    # case) runs the unchanged CPU body below, with automatic fallback on any miss.
    if method == 'poisson_binomial' and use_fast_stats and seed is not None:
        try:
            from spektrafilm.gpu.grain import layer_particle_model_dispatch
            _gpu_grain = layer_particle_model_dispatch(
                density, density_max, n_particles_per_pixel,
                grain_uniformity, seed, blur_particle,
            )
            if _gpu_grain is not None:
                return _gpu_grain
        except ImportError:
            pass

    if seed is not None:
        np.random.seed(seed) # scipy uses np.random
    
    probability_of_development = density/density_max
    probability_of_development = np.clip(probability_of_development, 1e-6, 1-1e-6) # for safe calc
    od_particle = density_max/n_particles_per_pixel
    
    grain = np.zeros_like(density)
    if method=='gamma_beta':
        gamma_rvs = scipy.stats.gamma.rvs
        beta_rvs = scipy.stats.beta.rvs
        seeds = gamma_rvs(n_particles_per_pixel/(1-grain_uniformity+1e-6), size=density.shape) * (1-grain_uniformity+1e-6)
        grain = beta_rvs(probability_of_development*n_particles_per_pixel,
                        (1-probability_of_development)*n_particles_per_pixel)*seeds*od_particle
    elif method=='poisson_binomial':
        if use_fast_stats:
            binom_rvs = fast_binomial
            poisson_rvs = fast_poisson
        else:
            binom_rvs = scipy.stats.binom.rvs
            poisson_rvs = scipy.stats.poisson.rvs
        saturation = 1 - probability_of_development*grain_uniformity*(1-1e-6)
        seeds = poisson_rvs(n_particles_per_pixel/saturation)
        grain = binom_rvs(seeds, probability_of_development)
        grain = np.double(grain)*od_particle*saturation
    
    if blur_particle>0:
        # grain = scipy.ndimage.gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
        grain = fast_gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
    return grain

def add_micro_structure(density_cmy_out, micro_structure, pixel_size_um):
    grain_micro_structure_blur_pixel = micro_structure[0]/pixel_size_um
    grain_micro_structure_sigma = micro_structure[1]*0.001/pixel_size_um  # grain microstructure[1] is in nm
    if grain_micro_structure_sigma > 0.05:
        clumping = fast_lognormal_from_mean_std(np.ones_like(density_cmy_out),
                                                np.ones_like(density_cmy_out)*grain_micro_structure_sigma)
        if grain_micro_structure_blur_pixel>0.4:
            # clumping = scipy.ndimage.gaussian_filter(clumping, (grain_micro_structure_blur_pixel,
            #                                                     grain_micro_structure_blur_pixel, 0))
            clumping = fast_gaussian_filter(clumping, grain_micro_structure_blur_pixel)
        density_cmy_out *= clumping
    return density_cmy_out

def apply_grain_to_density(density_cmy,
                           pixel_size_um=10,
                           particle_area_um2=0.2,
                           particle_scale=[1,0.8,3],
                           density_min=[0.03,0.06,0.04],
                           density_max_curves=[2.2,2.2,2.2],
                           grain_uniformity=[0.98,0.98,0.98],
                           grain_blur=1.0,
                           n_sub_layers=1,
                           fixed_seed=None,
                           ):
    density_min = np.array(density_min)
    density_max = density_max_curves + density_min
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2 = particle_area_um2*np.array(particle_scale)
    n_particles_per_pixel = pixel_area_um2/particle_area_um2
    sigma_blur_pixel = grain_blur
    
    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    if n_sub_layers>1:
        n_particles_per_pixel /= n_sub_layers
    
    density_cmy += density_min
    density_cmy_out = np.zeros_like(density_cmy)
    for ch in np.arange(3):
        for sl in np.arange(n_sub_layers):
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy[:,:,ch],
                                                            density_max=density_max[ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10)
    density_cmy_out /= n_sub_layers
    density_cmy_out -= density_min
    
    if sigma_blur_pixel>0.4:
        # density_cmy_out = scipy.ndimage.gaussian_filter(density_cmy_out, (sigma_blur_pixel, sigma_blur_pixel, 0))
        density_cmy_out = fast_gaussian_filter(density_cmy_out, sigma_blur_pixel)
        
    return density_cmy_out


# experimental
def apply_grain_to_density_layers(density_cmy_layers, # x,y,sublayers,rgb
                                  density_max_layers, # 3x3 [sublayers,rgb]
                                  pixel_size_um=10,
                                  particle_area_um2=0.2,
                                  particle_scale=[1,0.8,3], # rgb
                                  particle_scale_layers=[3,1,0.3], # sublayers
                                  density_min=[0.03,0.06,0.04],
                                  grain_uniformity=[0.98,0.98,0.98],
                                  grain_blur=1.0,
                                  grain_blur_dye_clouds_um=1.0,
                                  grain_micro_structure=(0.1, 30),
                                  fixed_seed=None,
                                  use_fast_stats=False,
                                  ):
    density_max_total = np.sum(density_max_layers, axis=0) # [sublayers,rgb]
    density_max_fractions = density_max_layers/density_max_total[None,:]
    density_min_layers = density_max_fractions*np.array(density_min)[None,:]
    density_max_layers = density_max_layers + density_min_layers
    
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2_layers = (particle_area_um2 * 
                                    np.array(particle_scale)[None,:] * 
                                    np.array(particle_scale_layers)[:,None]) # layers, rgb
    n_particles_per_pixel = pixel_area_um2*density_max_fractions/particle_area_um2_layers

    
    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    density_cmy_layers += density_min_layers
    density_cmy_out = np.zeros(density_cmy_layers.shape[0:3])
    for ch in np.arange(3): # rgb channels
        for sl in np.arange(3): # sublayers
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy_layers[:,:,sl,ch],
                                                            density_max=density_max_layers[sl,ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[sl,ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10,
                                                            blur_particle=grain_blur_dye_clouds_um,
                                                            use_fast_stats=use_fast_stats)
    
    # micro-structure
    density_cmy_out = add_micro_structure(density_cmy_out, grain_micro_structure, pixel_size_um)

    # final
    density_cmy_out -= density_min
    if grain_blur>0:
        # density_cmy_out = scipy.ndimage.gaussian_filter(density_cmy_out, (grain_blur, grain_blur, 0))
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain_blur)
    return density_cmy_out


RMS_APERTURE_UM = 48.0     # ISO-style granularity aperture (diameter, um)
RMS_NET_DENSITY = 1.0      # ...measured at net density 1.0


def _replace_params(grain, **changes):
    """dataclasses.replace when possible; attribute copy for legacy
    SimpleNamespace-style params objects in tests."""
    import dataclasses
    try:
        return dataclasses.replace(grain, **changes)
    except TypeError:
        import copy
        clone = copy.copy(grain)
        for key, value in changes.items():
            setattr(clone, key, value)
        return clone


def resolve_grain_model(grain: GrainParams, density_curves,
                        density_curves_layers=None) -> GrainParams:
    """Phase 11B: turn the 'rms' model into PRODUCTION-equivalent params.

    The OFX approach (manual 7.15): the granularity target is mapped into
    density-domain particle areas, then the normal particle model runs — so
    RMS keeps the full production stochastic texture (sublayers, dye
    clouds, micro structure) and rides the SAME validated CPU/GPU kernels.

    Math: the production per-pixel variance is od^2*n*p*s (see
    apply_grain_preview); averaged over the ISO aperture (area A um^2) the
    pixel size cancels — granularity is scale-invariant:
        sigma_ap^2 = d_max^2 * p * s * area / A          (single layer)
        sigma_ap^2 = sum_l d_max_l^2 * scale_layers_l * p_l * s_l
                     / fraction_l * area / A              (sublayers)
    Inverting at net density 1.0 for the target sigma_ap =
    granularity/1000 * strength gives the per-channel particle area, which
    rides in particle_scale (particle_area_um2 pinned to 1.0). Final blur,
    dye-cloud blur and micro structure redistribute noise well below the
    48 um aperture scale, so the calibration survives them to first order
    (tolerance documented in the 11B test). Any other model string returns
    the params unchanged."""
    if getattr(grain, 'model', 'production') != 'rms':
        return grain
    import dataclasses

    aperture_area = np.pi * (RMS_APERTURE_UM / 2.0) ** 2
    density_min = np.array(grain.density_min, dtype=float)
    uniformity = np.array(grain.uniformity, dtype=float)
    target = (np.maximum(np.array(grain.rms_granularity, dtype=float), 0.1)
              / 1000.0 * max(float(grain.rms_strength), 1e-3))
    curves = np.asarray(density_curves, dtype=float)

    use_layers = bool(grain.sublayers_active) and density_curves_layers is not None
    areas = np.empty(3)
    for ch in range(3):
        total = curves[:, ch]
        finite = np.isfinite(total)
        if use_layers:
            layers = np.asarray(density_curves_layers, dtype=float)[:, :, ch]
            d_max_layers = np.nanmax(layers, axis=0)
            fractions = d_max_layers / max(float(np.sum(d_max_layers)), 1e-9)
            d_min_layers = fractions * density_min[ch]
            d_max_adj = d_max_layers + d_min_layers
            # layer densities where the TOTAL curve crosses net density 1.0
            order = np.argsort(total[finite])
            t_sorted = total[finite][order]
            denom = 0.0
            for layer in range(layers.shape[1]):
                l_sorted = layers[finite, layer][order]
                d_layer = float(np.interp(RMS_NET_DENSITY, t_sorted, l_sorted))
                p_l = np.clip((d_layer + d_min_layers[layer]) / d_max_adj[layer],
                              1e-6, 1 - 1e-6)
                s_l = 1.0 - p_l * uniformity[ch]
                denom += (d_max_adj[layer] ** 2
                          * float(grain.particle_scale_layers[layer])
                          * p_l * s_l / max(float(fractions[layer]), 1e-9))
        else:
            d_max = float(np.nanmax(total)) + density_min[ch]
            p1 = np.clip((RMS_NET_DENSITY + density_min[ch]) / d_max,
                         1e-6, 1 - 1e-6)
            denom = d_max ** 2 * p1 * (1.0 - p1 * uniformity[ch])
        areas[ch] = (target[ch] ** 2) * aperture_area / max(denom, 1e-12)

    return dataclasses.replace(
        grain, model='production',
        particle_area_um2=1.0, particle_scale=tuple(areas))


def apply_grain_preview(density_cmy, pixel_size_um, grain: GrainParams,
                        density_curves):
    """Phase 11A 'preview' grain model: a fast, NOT-film-accurate impression.

    Gaussian noise whose per-pixel std is the CLOSED FORM of the production
    Poisson-binomial model's variance: for value = B*od*s with
    B ~ Binom(N, p), N ~ Poisson(n/s), s = 1 - p*u,
        Var = od^2 * n * p * s   ->   std = od * sqrt(n * p * (1 - p*u))
    so size/amount judgements transfer to the production model, while the
    cost is one seeded normal field per channel plus the final blur. The
    stochastic structure (dye-cloud shapes, sublayer mixing, micro
    clumping) is deliberately NOT reproduced - this is the look-building
    tier, not the render tier. Deterministic (fixed per-channel seeds,
    like the production path)."""
    density_min = np.array(grain.density_min)
    density_max = np.nanmax(density_curves, axis=0) + density_min
    particle_area_um2 = grain.particle_area_um2 * np.array(grain.particle_scale)
    # NAT 2026-07-22 (macOS-feel): the preview noise is computed at the
    # FULL-RESOLUTION pixel pitch. At preview sizes a 'pixel' is tens of um
    # of film — thousands of particles average out and the grain goes soft;
    # the macOS app renders production grain at native res and the display
    # downscale keeps the sparkle. preview_scale = full/preview width ratio
    # (1.0 = neutral, exports and other models untouched).
    scale = max(float(getattr(grain, 'preview_scale', 1.0)), 1.0)
    effective_pitch_um = pixel_size_um / scale
    n_particles_per_pixel = (effective_pitch_um ** 2) / particle_area_um2

    # 2026-07-21 polish: GPU fast path (one dispatch, all three channels;
    # statistical parity like every grain kernel). None -> the CPU numpy
    # path below runs unchanged.
    out = None
    try:
        from spektrafilm.gpu.grain_preview import preview_grain_dispatch
        out = preview_grain_dispatch(
            density_cmy, density_max, n_particles_per_pixel,
            np.array(grain.uniformity, dtype=float), density_min,
            monochrome=getattr(grain, 'monochrome', False),
        )
    except ImportError:
        out = None

    if out is None:
        out = np.empty_like(density_cmy)
        for ch in range(3):
            d = density_cmy[:, :, ch] + density_min[ch]
            p = np.clip(d / density_max[ch], 1e-6, 1 - 1e-6)
            od_particle = density_max[ch] / n_particles_per_pixel[ch]
            std = od_particle * np.sqrt(
                n_particles_per_pixel[ch] * p * (1.0 - p * grain.uniformity[ch]))
            noise = np.random.default_rng(1000 + ch).standard_normal(d.shape)
            out[:, :, ch] = d + std * noise - density_min[ch]

    # The final blur is a full-res-pixel sigma: at preview scale it shrinks
    # accordingly (and is skipped below the production 0.4px threshold —
    # exactly the crunch the full-res render shows). scale == 1.0 keeps the
    # historical behavior bit-identical.
    blur_sigma = grain.blur / scale
    if (blur_sigma > 0 and scale == 1.0) or blur_sigma > 0.4:
        out = fast_gaussian_filter(out, blur_sigma)
    return out


def apply_grain(
    density_cmy,
    pixel_size_um,
    grain: GrainParams,
    density_curves,
    density_curves_layers,
    profile_type,
    bypass_grain=False,
    use_fast_stats=False,
):
    if not grain.active or bypass_grain:
        return density_cmy

    # Phase 12A B&W: one SHARED noise field. The broadcast channels carry
    # identical densities and the bw specifics forced channel-0 params, so
    # channel 0's grain is the single-emulsion result — rendering normally
    # and replicating it is exactly upstream's n_channels==1 behavior, for
    # EVERY model and through the per-layer GPU dispatches unchanged.
    if getattr(grain, 'monochrome', False):
        out = apply_grain(
            density_cmy, pixel_size_um,
            _replace_params(grain, monochrome=False),
            density_curves, density_curves_layers, profile_type,
            bypass_grain=bypass_grain, use_fast_stats=use_fast_stats,
        )
        if out is not density_cmy:
            out = out.copy() if np.shares_memory(out, density_cmy) else out
            out[:, :, 1] = out[:, :, 0]
            out[:, :, 2] = out[:, :, 0]
        return out

    # Phase 11 model routing: 'preview' takes the fast impression path,
    # 'rms' resolves to production-equivalent params (derived particle
    # areas) and falls through to the production code, 'synthesis' runs the
    # Boolean Monte Carlo field; everything else (incl. pre-11A params
    # objects without the field) renders production.
    model = getattr(grain, 'model', 'production')
    if model == 'preview':
        return apply_grain_preview(density_cmy, pixel_size_um, grain,
                                   density_curves)
    if model == 'synthesis':
        from spektrafilm.model.grain_synthesis import apply_grain_synthesis
        return apply_grain_synthesis(density_cmy, pixel_size_um, grain,
                                     density_curves)
    if model == 'rms':
        grain = resolve_grain_model(grain, density_curves, density_curves_layers)

    if not grain.sublayers_active:
        density_max = np.nanmax(density_curves, axis=0)
        return apply_grain_to_density(
            density_cmy,
            pixel_size_um=pixel_size_um,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=density_max,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
        )

    # G2d-4 GPU fast path: interp-to-layers + 9x grain + dye/final blurs as ONE
    # resident submit (single upload/readback). Returns None -> the unchanged
    # path below runs (whose per-layer grain and blurs dispatch individually).
    # Statistical parity + exact wiring gate in test_grain_resident_parity.py.
    try:
        from spektrafilm.gpu.grain_resident import grain_layers_dispatch
        _gpu_out = grain_layers_dispatch(
            density_cmy, pixel_size_um, grain,
            density_curves, density_curves_layers,
            profile_type, use_fast_stats,
        )
        if _gpu_out is not None:
            return _gpu_out
    except ImportError:
        pass

    density_cmy_layers = interp_density_cmy_layers(
        density_cmy,
        density_curves,
        density_curves_layers,
        positive_film=profile_type == 'positive',
    )
    density_max_layers = np.nanmax(density_curves_layers, axis=0)
    return apply_grain_to_density_layers(
        density_cmy_layers,
        density_max_layers=density_max_layers,
        pixel_size_um=pixel_size_um,
        particle_area_um2=grain.particle_area_um2,
        particle_scale=grain.particle_scale,
        particle_scale_layers=grain.particle_scale_layers,
        density_min=grain.density_min,
        grain_uniformity=grain.uniformity,
        grain_blur=grain.blur,
        grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
        grain_micro_structure=grain.micro_structure,
        use_fast_stats=use_fast_stats,
    )

# TODO: make grain parameter with RMS granularity

if __name__=='__main__':
    density = np.ones((128,128))*2
    g1 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.99, sigma_blur=0.)
    g2 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.96, sigma_blur=0.)
    print('g1 ------------------')
    print('Density Test')
    print('Mean', np.mean(g1))
    print('RMS', np.std(g1)*1000)
    print('Skewness', scipy.stats.skew(g1.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g1.flatten()))
    print('g2 ------------------')
    print('Mean', np.mean(g2))
    print('RMS', np.std(g2)*1000)
    print('Skewness', scipy.stats.skew(g2.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g2.flatten()))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(g1, vmin=0, vmax=2.2)
    axs[0].set_title('Uniformity=0.99')
    axs[1].imshow(g2, vmin=0, vmax=2.2)
    axs[1].set_title('Uniformity=0.96')
    fig.suptitle('Fully saturated density with different uniformity')
    plt.show()
