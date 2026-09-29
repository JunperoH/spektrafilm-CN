from __future__ import annotations

import numpy as np
from opt_einsum import contract

from spektrafilm.gpu.spectral import spectral_reduce
from spektrafilm.model.bleach_bypass import bypass_effective_cmy, retained_silver_density
from spektrafilm.model.diffusion import apply_diffusion_filter_um
from spektrafilm.model.develop import apply_print_shape, compute_density_spectral, develop_print_morph
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light


class PrintingStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        enlarger_params,
        settings_params,
        lut_service,
        enlarger_service,
        resize_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._enlarger = enlarger_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._enlarger_service = enlarger_service
        self._resize_service = resize_service
        self._color_reference_service = color_reference_service

    # public methods

    def expose(self, cmy_film_density: np.ndarray) -> np.ndarray:
        
        cmy_film_black = np.zeros((1,1,3)) - np.array(self._film_render.grain.density_min)
        cmy_film_white = np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
        self._color_reference_service.log_raw_print_black = self._film_cmy_to_print_log_raw(cmy_film_black)
        self._color_reference_service.log_raw_print_white = self._film_cmy_to_print_log_raw(cmy_film_white)
        
        log_raw_print = self._lut_service.spectral_compute_enlarger(
            cmy_film_density,
            spectral_calculation=self._film_cmy_to_print_log_raw,
            data_min=-np.array(self._film_render.grain.density_min),
            data_max=np.nanmax(self._film.data.density_curves, axis=0),
            use_lut=self._settings.use_enlarger_lut,
        )    
        correction = self._color_reference_service.black_white_printing_exposure_correction()
        diffusion = self._enlarger.diffusion_filter
        diffusion_noop = (
            not diffusion.active
            or diffusion.strength <= 0
            or diffusion.spatial_scale <= 0
        )
        if diffusion_noop:
            # GPU drop-in fusing 10**x -> exposure scale -> log10 in one pass
            # (precise software exp2/log2, Gotcha 5 fixed); falls through to the
            # identical numpy expressions on any miss.
            try:
                from spektrafilm.gpu.elementwise import exp10_scale_log10_dispatch
                _scale = np.asarray(self._enlarger.print_exposure, dtype=np.float64) * np.asarray(
                    correction, dtype=np.float64
                )
                _gpu = exp10_scale_log10_dispatch(log_raw_print, _scale)
                if _gpu is not None:
                    return _gpu
            except ImportError:
                pass
        raw = 10**log_raw_print
        raw *= self._enlarger.print_exposure
        raw *= correction
        raw = apply_diffusion_filter_um(
            raw,
            diffusion,
            pixel_size_um=self._resize_service.pixel_size_um,
        )
        return np.log10(np.fmax(raw, 0.0) + 1e-10)

    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        density = develop_print_morph(
            log_raw,
            self._print.data.log_exposure,
            self._print.data.density_curves_model,
            density_curves_morph=self._print_render.density_curves_morph,
            profile_type=self._print.info.type,
        )
        # Phase 13A: Shadow/Highlight Shape — endpoint-preserving remap of
        # the developed paper density (== warping the paper response curve;
        # composes with the push/pull morph above). 0/0 returns `density`
        # unchanged (same object).
        return apply_print_shape(
            density,
            self._print.data.density_curves,
            shadow_shape=getattr(self._print_render, 'shadow_shape', 0.0),
            highlight_shape=getattr(self._print_render, 'highlight_shape', 0.0),
        )

    # private methods

    def _film_cmy_to_print_log_raw(self, cmy_film_density: np.ndarray) -> np.ndarray:
        sensitivity = 10 ** self._print.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(self._enlarger.illuminant)

        # Phase 13B: NEGATIVE bleach bypass — the retained neutral silver is
        # spectrally flat, so it factors out of the integral exactly as a
        # 10^-S light factor after the fused reduce; the leuco-cyan component
        # thins the cyan dye before it. A pure per-CMY-value transform, so
        # the enlarger LUT bakes it correctly. The midgray balance reference
        # stays UNBYPASSED: bypass reads as added density/severity to retime
        # by hand, exactly like the real process (manual 6.7).
        bypass_amount = getattr(self._film_render, 'bleach_bypass_amount', 0.0)
        silver = None
        if bypass_amount > 0.0:
            silver = retained_silver_density(cmy_film_density, bypass_amount)
            cmy_film_density = bypass_effective_cmy(
                cmy_film_density, bypass_amount,
                getattr(self._film_render, 'bleach_bypass_leuco_cyan', 1.0))

        print_illuminant = self._enlarger_service.enlarger_filtered_illuminant(enlarger_light_source)
        # Fused spectral block (density -> light -> sensitivity contraction) via
        # the GPU kernel with automatic CPU fallback; the math is unchanged
        # (spectral_reduce's CPU path delegates to compute_density_spectral and
        # density_to_light). Exposure factor, preflash and log10 stay on host.
        raw = spectral_reduce(
            cmy_film_density,
            self._film.data.channel_density,
            self._film.data.base_density,
            print_illuminant,
            sensitivity,
        )
        if silver is not None:
            raw = raw * 10.0 ** (-silver)
        raw = raw * self._compute_exposure_factor_midgray(sensitivity, print_illuminant)
        raw += self._compute_raw_preflash(enlarger_light_source, sensitivity)
        return np.log10(np.fmax(raw, 0.0) + 1e-10)

    def _compute_raw_preflash(self, light_source, sensitivity):
        if self._enlarger.preflash_exposure > 0:
            preflash_illuminant = self._enlarger_service.preflash_filtered_illuminant(light_source)
            density_base = np.asarray(self._film.data.base_density)[None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
            return raw_preflash * self._enlarger.preflash_exposure
        return np.zeros((3,))

    def _compute_exposure_factor_midgray(self, sensitivity, print_illuminant):
        factor_midgray = _exposure_factor(sensitivity, print_illuminant, self._enlarger_service.density_spectral_midgray)
        if self._enlarger_service.density_spectral_midgray_comp is not None:
            factor_midgray_comp = _exposure_factor(sensitivity, print_illuminant,
                                                    self._enlarger_service.density_spectral_midgray_comp)
        else:
            factor_midgray_comp = 1.0
        if self._enlarger.print_exposure_compensation and not self._enlarger.normalize_print_exposure:
            return factor_midgray_comp / factor_midgray
        elif self._enlarger.normalize_print_exposure and self._enlarger.print_exposure_compensation:
            return factor_midgray_comp
        elif self._enlarger.normalize_print_exposure and not self._enlarger.print_exposure_compensation:
            return factor_midgray
        else:
            return 1.0

def _exposure_factor(sensitivity, print_illuminant, density_spectral_midgray):
    light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
    raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
    raw_midgray = np.fmax(raw_midgray, 1e-10)
    # use the geometric mean to normalize the exposure
    raw_midgray_geomean = np.exp(np.mean(np.log(raw_midgray), axis=2, keepdims=True))
    return 1 / raw_midgray_geomean