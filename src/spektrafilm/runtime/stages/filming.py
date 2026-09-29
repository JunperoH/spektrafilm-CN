from __future__ import annotations

import numpy as np

from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.model.color_filters import color_filter_transmittance, compute_band_pass_filter
from spektrafilm.model.diffusion import apply_diffusion_filter_um, apply_gaussian_blur_um, apply_halation_um, boost_highlights
from spektrafilm.model.develop import compute_density_spectral, develop, develop_simple
from spektrafilm.utils.autoexposure import measure_autoexposure_ev
from spektrafilm.utils.spectral_upsampling import rgb_to_raw, rgb_to_raw_mallett2019


class FilmingStage:
    def __init__(self, film, film_render_params, camera_params, io_params, settings_params,
                 lut_service, resize_service, enlarger_service, color_reference_service):
        self._film = film
        self._film_render = film_render_params
        self._camera = camera_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._resize_service = resize_service
        self._enlarger_service = enlarger_service
        
        # send info for hanatos2025 sensitivity adaptation to LUT service.
        # Phase 12A: the adaptation params are FITTED per stock; profiles
        # that ship without them (the B&W pre-port) cannot apply it — gate on
        # the params actually existing, not only on the settings flags.
        hanatos2025_adaptation = self._film.hanatos2025_adaptation()
        hanatos2025_adaptation.apply_window = (
            self._settings.apply_hanatos2025_adaptation_window
            and np.asarray(hanatos2025_adaptation.window_params).size > 0)
        hanatos2025_adaptation.apply_surface = (
            self._settings.apply_hanatos2025_adaptation_surface
            and np.asarray(hanatos2025_adaptation.surface_params).size > 0)
        hanatos2025_adaptation.spectral_gaussian_blur = self._settings.spectral_gaussian_blur
        self._lut_service.set_hanatos2025_adaptation(hanatos2025_adaptation)
        # Input gamut compression is per-bundle config (lives on params.io
        # so the GUI and bundle bakes share the same code path). The
        # service caches the LUT and invalidates it when this spec changes.
        self._lut_service.set_input_gamut_compress(self._io.input_gamut_compress)
        self._enlarger_service.density_spectral_midgray, self._enlarger_service.density_spectral_midgray_comp = self._compute_density_spectral_midgray_to_balance_print()
        self._color_reference_service = color_reference_service

    # public methods

    def auto_exposure(self, image: np.ndarray) -> float:
        if self._camera.auto_exposure:
            small_preview = self._resize_service.small_preview(image)
            autoexposure_ev = measure_autoexposure_ev(
                small_preview,
                self._io.input_color_space,
                self._io.input_cctf_decoding,
                method=self._camera.auto_exposure_method,
            )
            return image * 2 ** autoexposure_ev
        return image

    def expose(self, image: np.ndarray) -> np.ndarray:
        raw = self._rgb_to_film_raw(
            image,
            color_space=self._io.input_color_space,
            apply_cctf_decoding=self._io.input_cctf_decoding,
        )
        raw *= 2 ** self._camera.exposure_compensation_ev
        boost_highlights(raw, self._film_render.halation.boost_ev,
                         self._film_render.halation.boost_range,
                         self._film_render.halation.protect_ev, out=raw)
        raw = apply_diffusion_filter_um(
            raw,
            self._camera.diffusion_filter,
            pixel_size_um=self._resize_service.pixel_size_um,
        )
        raw = apply_gaussian_blur_um(raw, self._camera.lens_blur_um, self._resize_service.pixel_size_um)
        raw = apply_halation_um(raw, self._film_render.halation, self._resize_service.pixel_size_um)
        correction = self._color_reference_service.black_white_filming_exposure_correction()
        # GPU drop-in for the scale + log10 tail (precise software log2, Gotcha 5
        # fixed); falls through to the identical numpy expression on any miss.
        try:
            from spektrafilm.gpu.elementwise import scale_log10_dispatch
            _gpu = scale_log10_dispatch(raw, correction)
            if _gpu is not None:
                return _gpu
        except ImportError:
            pass
        raw *= correction
        log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
        return log_raw

    def develop(self, log_raw: np.ndarray) -> np.ndarray:
        return develop(
            log_raw,
            self._resize_service.pixel_size_um,
            self._film.data.log_exposure,
            self._film.data.density_curves,
            self._film.data.density_curves_layers,
            self._film_render.dir_couplers,
            self._film_render.grain,
            self._film.info.type,
            gamma_factor=self._film_render.density_curve_gamma,
            use_fast_stats=self._settings.use_fast_stats,
        )

    # private methods

    def _rgb_to_film_raw(
        self,
        rgb: np.ndarray,
        *,
        color_space: str = "sRGB",
        apply_cctf_decoding: bool = False,
    ) -> np.ndarray:
        sensitivity = 10 ** self._film.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)

        if self._camera.filter_uv[0] > 0 or self._camera.filter_ir[0] > 0:
            illuminant = standard_illuminant(self._film.info.reference_illuminant)
            band_pass_filter = compute_band_pass_filter(self._camera.filter_uv, self._camera.filter_ir)
            band_pass_filter = np.tile(band_pass_filter[:, None], (1, 3))
            normalization = np.sum(sensitivity * band_pass_filter * illuminant[:, None], axis=0) / np.sum(sensitivity * illuminant[:, None], axis=0)
            sensitivity *= band_pass_filter / normalization

        # A camera taking filter (UV / haze / colored) sits in front of the lens
        # and multiplies the incoming light spectrum; fold its transmittance into
        # the spectral sensitivity. No renormalization -- the attenuation and
        # color cast are the intended effect. (The midgray print-balance reference
        # flows through this same path, so print exposure balancing sees the
        # filtered response consistently.)
        transmittance = color_filter_transmittance(getattr(self._camera, 'color_filter', 'none'))
        if transmittance is not None:
            sensitivity = sensitivity * transmittance[:, None]

        method = self._settings.rgb_to_raw_method
        if method == "mallett2019_legacy":
            # The pre-registry per-pixel mallett path (green-channel midgray
            # normalization, unclipped out-of-sRGB reflectance). Kept verbatim
            # for old renders; the registry 'mallett2019' is the LUT variant.
            raw = rgb_to_raw_mallett2019(rgb, sensitivity,
                            color_space=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            reference_illuminant=self._film.info.reference_illuminant)
        else:
            # Single generic dispatch: the method string selects the LUT descriptor;
            # the service builds (and caches) its tc_lut, and rgb_to_raw consumes it.
            # Reflectance methods (jakob2019 / otsu2018 / mallett2019 / arctic2026…)
            # and the irradiance method (hanatos2025) flow through the same two calls.
            tc_lut = self._lut_service.get_filming_tc_lut(
                method, sensitivity, self._film.info.reference_illuminant)
            raw = rgb_to_raw(
                method,
                rgb,
                sensitivity,
                color_space=color_space,
                apply_cctf_decoding=apply_cctf_decoding,
                reference_illuminant=self._film.info.reference_illuminant,
                tc_lut=tc_lut,
            )
        return raw
    
    def _compute_density_spectral_midgray_to_balance_print(self):
        rgb_midgray = np.array([[[0.184] * 3]])
        density_spectral_midgray = self._simple_rgb_to_density_spectral(rgb_midgray)
        if self._enlarger_service.print_exposure_compensation:
            neg_exp_comp_ev = self._camera.exposure_compensation_ev
            rgb_midgray_comp = np.array([[[0.184] * 3]]) * 2 ** neg_exp_comp_ev
            density_spectral_midgray_comp = self._simple_rgb_to_density_spectral(rgb_midgray_comp)
        else:
            density_spectral_midgray_comp = None
        return density_spectral_midgray, density_spectral_midgray_comp

    def _simple_rgb_to_density_spectral(self, rgb: np.ndarray) -> np.ndarray:
        raw = self._rgb_to_film_raw(rgb) 
        log_raw = np.log10(raw + 1e-10)
        density_cmy = develop_simple(
            log_raw,
            self._film.data.log_exposure,
            self._film.data.density_curves,
            gamma_factor=self._film_render.density_curve_gamma,
        )
        density_spectral = compute_density_spectral(
            self._film.data.channel_density,
            density_cmy,
            base_density=self._film.data.base_density,
        )
        return density_spectral
    