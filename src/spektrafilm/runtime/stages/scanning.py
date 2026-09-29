from __future__ import annotations

import colour
import numpy as np
from opt_einsum import contract

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.gpu.spectral import spectral_reduce
from spektrafilm.model.diffusion import apply_gaussian_blur, apply_unsharp_mask
from spektrafilm.model.develop import compute_density_spectral
from spektrafilm.model.glare import add_glare
from spektrafilm.model.scan_finishing import (
    apply_scan_finishing_display,
    apply_scan_finishing_linear,
    apply_scan_geometry,
    scan_finishing_is_identity,
    scan_geometry_is_identity,
)
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.utils.gamut_compression import compress_rgb


class ScanningStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        scanner_params,
        io_params,
        settings_params,
        lut_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._scanner = scanner_params
        self._io = io_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._color_reference_service = color_reference_service
        
        self.cmy_to_log_xyz = self._return_callable_cmy_to_log_xyz()
        
        # communicate to the color reference service the callable to convert cmy densities to log xyz
        self._color_reference_service.cmy_to_log_xyz = self.cmy_to_log_xyz
        
    # public methods

    def scan(self, density_channels: np.ndarray) -> np.ndarray:
        rgb = self._density_to_rgb(density_channels, use_lut=self._settings.use_scanner_lut)
        rgb = self._apply_blur_and_unsharp(rgb)
        # Display-referred scan finishing (SCAN tab). Identity at defaults —
        # the historical path below is byte-untouched then. The LINEAR group
        # (scan exposure / analog gains / color separation) runs before the
        # output cctf; the DISPLAY group runs on the encoded output. When the
        # file is written linear the display group runs inside a temporary
        # encode/decode round-trip so the LOOK is independent of the
        # write-time encoding choice (same contract as the encoding itself).
        finishing = getattr(self._scanner, 'finishing', None)
        if not scan_finishing_is_identity(finishing):
            rgb = self._finishing_linear(rgb, finishing)
            if self._io.output_cctf_encoding:
                rgb = self._finishing_display(self._apply_cctf_encoding(rgb), finishing)
            else:
                encoded = colour.RGB_to_RGB(
                    rgb, self._io.output_color_space, self._io.output_color_space,
                    apply_cctf_decoding=False, apply_cctf_encoding=True,
                )
                encoded = self._finishing_display(encoded, finishing)
                rgb = colour.RGB_to_RGB(
                    encoded, self._io.output_color_space, self._io.output_color_space,
                    apply_cctf_decoding=True, apply_cctf_encoding=False,
                )
        else:
            rgb = self._apply_cctf_encoding(rgb)
        # Scan-time geometry LAST: desqueeze -> skew -> crop as a cutout of the
        # rendered film (grain/halation keep full-frame scale; the camera crop
        # in io params is the one that re-frames the film gate).
        if not scan_geometry_is_identity(finishing):
            rgb = apply_scan_geometry(rgb, finishing)
        return rgb

    # private methods

    def _finishing_linear(self, rgb: np.ndarray, finishing) -> np.ndarray:
        # Phase 6D GPU drop-in (guard + error trap inside the dispatcher);
        # None means the unchanged CPU function runs.
        try:
            from spektrafilm.gpu.scan_finishing import scan_finishing_linear_dispatch
            _gpu = scan_finishing_linear_dispatch(rgb, finishing)
            if _gpu is not None:
                return _gpu
        except ImportError:
            pass
        return apply_scan_finishing_linear(rgb, finishing)

    def _finishing_display(self, rgb: np.ndarray, finishing) -> np.ndarray:
        try:
            from spektrafilm.gpu.scan_finishing import scan_finishing_display_dispatch
            _gpu = scan_finishing_display_dispatch(rgb, finishing)
            if _gpu is not None:
                return _gpu
        except ImportError:
            pass
        return apply_scan_finishing_display(rgb, finishing)

    def _density_to_rgb(self, density_channels: np.ndarray, *, use_lut: bool) -> np.ndarray:
        if self._io.scan_film:
            glare = None
            density_min = -np.array(self._film_render.grain.density_min)
            density_max = np.nanmax(self._film.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
        else:
            glare = self._print_render.glare
            density_min = np.nanmin(self._print.data.density_curves, axis=0)
            density_max = np.nanmax(self._print.data.density_curves, axis=0)
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            
        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        log_xyz = self._lut_service.spectral_compute_scanner(
            density_channels,
            spectral_calculation=self.cmy_to_log_xyz,
            data_min=density_min,
            data_max=density_max,
            use_lut=use_lut,
        )
        illuminant_xyz = contract("k,kl->l", scan_illuminant, STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
        # Phase 6D tail: fused exp10 -> glare -> XYZ->RGB in ONE submit (one
        # upload / one readback, replacing three per-step round-trips). Only
        # when black_white_xyz_correction is an identity — else the unchanged
        # per-step path below runs.
        svc = self._color_reference_service
        _bw_identity = not getattr(svc, '_black_correction', True) and not getattr(
            svc, '_white_correction', True)
        rgb = None
        if _bw_identity:
            try:
                from spektrafilm.gpu.scan_tail_resident import scan_tail_dispatch
                rgb = scan_tail_dispatch(
                    log_xyz, glare, illuminant_xyz,
                    self._io.output_color_space, illuminant_xy,
                )
            except ImportError:
                rgb = None
        if rgb is None:
            # GPU drop-in for 10 ** log_xyz (precise software exp2, Gotcha 5 fixed).
            xyz = None
            try:
                from spektrafilm.gpu.elementwise import exp10_scale_dispatch
                xyz = exp10_scale_dispatch(log_xyz, 1.0)
            except ImportError:
                xyz = None
            if xyz is None:
                xyz = 10 ** log_xyz
            xyz = self._color_reference_service.black_white_xyz_correction(xyz)
            # GPU glare (seeded lognormal field + blur + add in one submit;
            # statistical parity like grain, deterministic run-to-run unlike
            # the CPU's unseeded stream). None -> unchanged CPU add_glare.
            _gpu_glare = None
            try:
                from spektrafilm.gpu.glare import add_glare_dispatch
                _gpu_glare = add_glare_dispatch(xyz, illuminant_xyz, glare)
            except ImportError:
                _gpu_glare = None
            xyz = _gpu_glare if _gpu_glare is not None else add_glare(xyz, illuminant_xyz, glare)
            # GPU drop-in for the XYZ->RGB matrix (chromatic adaptation + matrix);
            # the matrix is colour's own, so this is parity-exact with a CPU fallback.
            try:
                from spektrafilm.gpu.color import xyz_to_rgb_dispatch
                rgb = xyz_to_rgb_dispatch(xyz, self._io.output_color_space, illuminant_xy)
            except ImportError:
                rgb = None
            if rgb is None:
                rgb = colour.XYZ_to_RGB(
                    xyz,
                    colourspace=self._io.output_color_space,
                    apply_cctf_encoding=False,
                    illuminant=illuminant_xy,
                )
        # Output gamut compression. Compresses chromaticities the
        # simulation reached that fall outside the output primaries
        # cube; for perceptual algorithms (oklch / oklrab / jzazbz /
        # cam16ucs) the spec's lightness_compression also pulls
        # super-bright pixels back into the cube via a one-sided soft
        # roll-off on the perceptual lightness axis (black stays at 0).
        # With both in place the output is in [0, 1] without a
        # downstream clip; see n100 / n110 for the design and b40 for
        # the smoothness analysis.
        rgb = compress_rgb(
            rgb, self._io.output_gamut_compress,
            output_color_space=self._io.output_color_space,
        )
        return rgb

    def _return_callable_cmy_to_log_xyz(self):
        # Phase 13B: bleach bypass at scan time — the negative's retained
        # silver + leuco cyan when scanning FILM, the print's retained
        # silver when scanning the PRINT. Same exact 10^-S factorization as
        # the enlarger path; still a pure per-CMY function (scanner-LUT safe).
        if self._io.scan_film:
            channel_density = self._film.data.channel_density
            base_density = self._film.data.base_density
            scan_illuminant = standard_illuminant(self._film.info.viewing_illuminant)
            bypass_amount = getattr(self._film_render, 'bleach_bypass_amount', 0.0)
            leuco = getattr(self._film_render, 'bleach_bypass_leuco_cyan', 1.0)
        else:
            channel_density = self._print.data.channel_density
            base_density = self._print.data.base_density
            scan_illuminant = standard_illuminant(self._print.info.viewing_illuminant)
            bypass_amount = getattr(self._print_render, 'bleach_bypass_amount', 0.0)
            leuco = 0.0                     # leuco cyan is negative-side only

        normalization = np.sum(scan_illuminant * STANDARD_OBSERVER_CMFS[:, 1], axis=0)

        def cmy_to_log_xyz(density_cmy: np.ndarray) -> np.ndarray:
            silver = None
            if bypass_amount > 0.0:
                from spektrafilm.model.bleach_bypass import (
                    bypass_effective_cmy, retained_silver_density)
                silver = retained_silver_density(density_cmy, bypass_amount)
                if leuco > 0.0:
                    density_cmy = bypass_effective_cmy(density_cmy, bypass_amount, leuco)
            # Fused spectral block (density -> light -> CMF contraction) via the
            # GPU kernel with automatic CPU fallback; the math is unchanged
            # (spectral_reduce's CPU path delegates to compute_density_spectral
            # and density_to_light). Cheap scalars stay on the host.
            xyz = spectral_reduce(
                density_cmy,
                channel_density,
                base_density,
                scan_illuminant,
                STANDARD_OBSERVER_CMFS[:],
            ) / normalization
            if silver is not None:
                xyz = xyz * 10.0 ** (-silver)
            return np.log10(np.fmax(xyz, 0.0) + 1e-10)
        return cmy_to_log_xyz

    def _apply_blur_and_unsharp(self, rgb: np.ndarray) -> np.ndarray:
        sigma, amount = self._scanner.unsharp_mask
        # GPU fast path: lens blur + unsharp as one resident submit (upload
        # once, both blurs + the (1+a)*x - a*g mix on device, read back once).
        # Falls through to the unchanged per-call path on any miss.
        try:
            from spektrafilm.gpu.scan_blur_resident import scan_blur_unsharp_dispatch
            _gpu = scan_blur_unsharp_dispatch(rgb, self._scanner.lens_blur, sigma, amount)
            if _gpu is not None:
                return _gpu
        except ImportError:
            pass
        rgb = apply_gaussian_blur(rgb, self._scanner.lens_blur)
        if sigma > 0 and amount > 0:
            rgb = apply_unsharp_mask(rgb, sigma=sigma, amount=amount)
        return rgb

    def _apply_cctf_encoding(self, rgb: np.ndarray) -> np.ndarray:
        if self._io.output_cctf_encoding:
            # GPU drop-in for the output cctf encode (sRGB replicated exactly;
            # other colourspaces fall through to colour). clip=False since
            # 0.3.4: gamut compression owns cube containment and the CPU path
            # no longer clips, so signed/OOG values pass through identically.
            try:
                from spektrafilm.gpu.color import cctf_encode_clip_dispatch
                _gpu = cctf_encode_clip_dispatch(
                    rgb, self._io.output_color_space, clip=False
                )
                if _gpu is not None:
                    return _gpu
            except ImportError:
                pass
            rgb = colour.RGB_to_RGB(
                rgb,
                self._io.output_color_space,
                self._io.output_color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True,
            )
        return rgb



