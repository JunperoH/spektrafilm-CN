import copy
from datetime import date
from importlib.metadata import PackageNotFoundError, version as distribution_version
import importlib.resources as pkg_resources
import json
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Mapping

import numpy as np


_PROJECT_URL = 'https://github.com/andreavolpato/spektrafilm'
_PROFILE_LICENSE_URL = f'{_PROJECT_URL}/blob/main/SPEKTRAFILM_LICENSE.txt'


PROFILE_TYPES = frozenset({'negative', 'positive'})
PROFILE_SUPPORTS = frozenset({'film', 'paper'})
PROFILE_STAGES = frozenset({'filming', 'printing'})
PROFILE_USES = frozenset({'still', 'cine'})
PROFILE_ANTIHALATION = frozenset({'strong', 'weak', 'no'})
PROFILE_CHANNEL_MODELS = frozenset({'color', 'bw'})
LEGACY_PROFILE_INFO_KEYS = frozenset({
    'fitted_cmy_midscale_neutral_density',
    'log_exposure_midscale_neutral',
})


def _package_version() -> str:
    try:
        return distribution_version('spektrafilm')
    except PackageNotFoundError:
        return '0+unknown'

def _created_date() -> str:
    return date.today().isoformat()

def _copyright_statement() -> str:
    return f"Copyright (c) {date.today().year} Andrea Volpato. Licensed under CC BY-SA 4.0."

def _empty_vector() -> np.ndarray:
    return np.empty((0,), dtype=float)

def _empty_matrix() -> np.ndarray:
    return np.empty((0, 3), dtype=float)

def _empty_tensor() -> np.ndarray:
    return np.empty((0, 3, 3), dtype=float)


def _empty_layer_matrix() -> np.ndarray:
    return np.empty((3, 0), dtype=float)


@dataclass
class DensityCurvesModel:
    """Parametric model of the density curves.

    `centers`, `amplitudes`, `sigmas` are 2D arrays shaped (n_channels, n_layers).
    n_layers can be 2, 3, ... — set by the array shape.

    `model_type` names the per-layer sigmoid:
      - 'norm_cdfs'      sum of Gaussian CDFs (centers, amplitudes, sigmas);
                         `alphas` is None. Every existing profile JSON stores
                         the historical name 'cdfs' for this family; it is
                         normalized to 'norm_cdfs' on load (alias, NOT a new
                         model — no profile regenerates).
      - 'sept_norm_cdfs' sum of septic-polynomial CDF approximations with a
                         per-layer median-preserving skew `alphas` (|a|<1);
                         a=0 reproduces the symmetric Gaussian fit.
    """
    model_type: str = 'norm_cdfs'
    centers: np.ndarray = field(default_factory=_empty_layer_matrix)
    amplitudes: np.ndarray = field(default_factory=_empty_layer_matrix)
    sigmas: np.ndarray = field(default_factory=_empty_layer_matrix)
    alphas: np.ndarray | None = None

    def __post_init__(self):
        if self.model_type == 'cdfs':
            self.model_type = 'norm_cdfs'
        self.centers = np.asarray(self.centers, dtype=float)
        self.amplitudes = np.asarray(self.amplitudes, dtype=float)
        self.sigmas = np.asarray(self.sigmas, dtype=float)
        if self.alphas is not None:
            alphas = np.asarray(self.alphas, dtype=float)
            self.alphas = alphas if alphas.size else None

    @property
    def n_channels(self) -> int:
        return self.centers.shape[0] if self.centers.ndim == 2 else 0

    @property
    def n_layers(self) -> int:
        return self.centers.shape[1] if self.centers.ndim == 2 else 0


@dataclass
class ProfileMetadata:
    version: str = field(default_factory=_package_version)
    copyright: str = field(default_factory=_copyright_statement)
    created: str = field(default_factory=_created_date)
    license: str = (
        "spektrafilm profile by Andrea Volpato, licensed under CC BY-SA 4.0. "
        "Redistribution and derivatives must credit the author, link the "
        f"project ({_PROJECT_URL}),"
        "preserve this license, and remain CC BY-SA 4.0."
        "Modifications must be noted. Full text of the license and "
        f"attribution requirements: {_PROFILE_LICENSE_URL}."
    )
    citation: str = (
        "If you use this profile in your work, please cite the spektrafilm "
        f"project: {_PROJECT_URL}, see CITATION.cff for details."
    )
    datasource: str = """
    This profile was created by processing raw measurement data from data-sheets and/or scientific papers. Original data are property of the respective holders.
    Film/photo-paper: Kodak and Fujifilm data-sheets, scientific publications, and technical material.
    Reflectance: Otsu (https://github.com/enneract/otsu2018), Munsell (https://zenodo.org/records/3269912), human skin (https://www.nist.gov/programs-projects/reflectance-measurements-human-skin), forest colors (https://zenodo.org/records/3269920), Japan colors (https://zenodo.org/records/5217752).
    All data publicly available.
    """.strip()

@dataclass
class ProfileInfo:
    stock: str = None
    name: str = None
    type: str = 'negative'
    support: str = 'film'
    stage: str = 'filming'
    use: str = 'still'
    antihalation: str = 'weak'
    target_print: str | None = None
    channel_model: str = 'color'
    densitometer: str = 'status_M'
    log_sensitivity_density_over_min: float = 0.2
    reference_illuminant: str = 'D55'
    viewing_illuminant: str = 'D50'
    # Phase 10B custom stocks: partial film_render overrides carried by a
    # baked recipe ({'grain': {...}, 'halation': {...}, 'dir_couplers': {...}});
    # None for every bundled profile. Applied by _apply_film_specifics.
    custom_look: dict | None = None
    # The bundled base a baked recipe derives from (None for bundled
    # profiles). Lookups keyed by stock slug (neutral print filters) fall
    # back to it — the custom stock's spectra ARE the base's.
    custom_base: str | None = None

@dataclass
class Hanatos2025SensitivityAdaptation:
    window_params: np.ndarray = field(default_factory=_empty_vector)
    surface_params: np.ndarray = field(default_factory=_empty_vector)
    spectral_gaussian_blur: float = 0.0 # sigma in nm for gaussian blur of the spectra
    reference_illuminant: str = None # "D55" or "T"
    apply_window: bool = True
    apply_surface: bool = True
    active: bool = None

@dataclass
class ProfileData:
    wavelengths: np.ndarray = field(default_factory=_empty_vector)
    log_sensitivity: np.ndarray = field(default_factory=_empty_matrix)
    hanatos2025_adaptation_window_params: np.ndarray = field(default_factory=_empty_vector)
    hanatos2025_adaptation_surface_params: np.ndarray = field(default_factory=_empty_vector)
    channel_density: np.ndarray = field(default_factory=_empty_matrix)
    base_density: np.ndarray = field(default_factory=_empty_vector)
    midscale_neutral_density: np.ndarray = field(default_factory=_empty_vector)
    log_exposure: np.ndarray = field(default_factory=_empty_vector)
    density_curves: np.ndarray = field(default_factory=_empty_matrix)
    density_curves_layers: np.ndarray = field(default_factory=_empty_tensor)
    density_curves_model: DensityCurvesModel = field(default_factory=DensityCurvesModel)
    # Phase 12A: B&W profiles carry a development-time family (minutes).
    # After the loader collapses the curves to one member, the FULL family
    # stays here so the GUI can enumerate it (empty on color profiles).
    development_time: np.ndarray = field(default_factory=_empty_vector)

    def __post_init__(self):
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.log_sensitivity = np.asarray(self.log_sensitivity, dtype=float)
        self.hanatos2025_adaptation_window_params = np.asarray(self.hanatos2025_adaptation_window_params, dtype=float)
        if self.hanatos2025_adaptation_window_params.size == 0:
            self.hanatos2025_adaptation_window_params = _empty_vector()
        self.hanatos2025_adaptation_surface_params = np.asarray(self.hanatos2025_adaptation_surface_params, dtype=float)
        if self.hanatos2025_adaptation_surface_params.size == 0:
            self.hanatos2025_adaptation_surface_params = _empty_matrix()
        self.channel_density = np.asarray(self.channel_density, dtype=float)
        self.base_density = np.asarray(self.base_density, dtype=float)
        self.midscale_neutral_density = np.asarray(self.midscale_neutral_density, dtype=float)
        self.log_exposure = np.asarray(self.log_exposure, dtype=float)
        self.density_curves = np.asarray(self.density_curves, dtype=float)
        self.density_curves_layers = np.asarray(self.density_curves_layers, dtype=float)
        self.development_time = np.asarray(self.development_time, dtype=float)
        if not isinstance(self.density_curves_model, DensityCurvesModel):
            if isinstance(self.density_curves_model, Mapping):
                self.density_curves_model = DensityCurvesModel(**dict(self.density_curves_model))
            else:
                raise TypeError('density_curves_model must be a DensityCurvesModel or Mapping')


@dataclass
class Profile:
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)
    info: ProfileInfo = field(default_factory=ProfileInfo)
    data: ProfileData = field(default_factory=ProfileData)

    def __post_init__(self):
        if not isinstance(self.metadata, ProfileMetadata):
            raise TypeError('metadata must be a ProfileMetadata instance')
        if not isinstance(self.info, ProfileInfo):
            raise TypeError('info must be a ProfileInfo instance')
        if not isinstance(self.data, ProfileData):
            raise TypeError('data must be a ProfileData instance')

    def clone(self) -> 'Profile':
        return copy.deepcopy(self)

    def update_info(self, **changes) -> 'Profile':
        self.info = replace(self.info, **changes)
        return self

    def update_data(self, **changes) -> 'Profile':
        self.data = replace(self.data, **changes)
        return self

    def update(self, *, info=None, data=None) -> 'Profile':
        if info:
            self.update_info(**info)
        if data:
            self.update_data(**data)
        return self

    def hanatos2025_adaptation(self) -> Hanatos2025SensitivityAdaptation:
        return Hanatos2025SensitivityAdaptation(
            window_params=self.data.hanatos2025_adaptation_window_params,
            surface_params=self.data.hanatos2025_adaptation_surface_params,
            reference_illuminant=self.info.reference_illuminant,
        )
    
    @property
    def is_positive(self) -> bool:
        return self.info.type == 'positive'

    @property
    def is_negative(self) -> bool:
        return self.info.type == 'negative'

    @property
    def is_paper(self) -> bool:
        return self.info.support == 'paper'

    @property
    def is_film(self) -> bool:
        return self.info.support == 'film'
    
    @property
    def is_color(self) -> bool:
        return self.info.channel_model == 'color'
    
    @property
    def is_bw(self) -> bool:
        return self.info.channel_model == 'bw'

    @property
    def is_filming(self) -> bool:
        return self.info.stage == 'filming'

    @property
    def is_printing(self) -> bool:
        return self.info.stage == 'printing'

    @property
    def is_still(self) -> bool:
        return self.info.use == 'still'

    @property
    def is_cine(self) -> bool:
        return self.info.use == 'cine'


def development_time_index(times, requested=None) -> int:
    """Nearest family member; default the floor-middle of the family
    (mirrors the spektrafilm.rs reference / upstream select_development_time)."""
    times = list(times or [])
    if len(times) <= 1:
        return 0
    if requested is None:
        return (len(times) - 1) // 2
    return int(np.argmin([abs(float(t) - float(requested)) for t in times]))


def _resolve_bw_data(data_payload, development_time=None):
    """Phase 12A: collapse a B&W development-time family to one member and
    broadcast the single channel onto the 3-channel engine layout.

    Line-for-line adaptation of spektrafilm.rs resolve_for_render
    (references/spektrafilm.rs/repo/crates/spektrafilm-core/src/profile.rs;
    itself mirroring Andrea's upstream n_channels==1 semantics — Phase 12 is
    a PRE-PORT and gets replaced by the upstream B&W release when it ships):
      - density_curves columns index the family -> keep column idx, then
        broadcast to [v, v, v];
      - base_density is per-wavelength x family -> keep column idx;
      - log_sensitivity -> [s, s, s] (one panchromatic record);
      - channel_density -> [c, 0, 0]: the G/B dye lanes carry NO spectral
        weight, so every spectral integration computes exactly the upstream
        single-channel density x dye_spectrum;
      - the curves model rows (centers/amplitudes/sigmas are [time][layers])
        -> pick idx, replicate x3 channels;
      - density_curves_layers entries may carry the family in a 3rd dim ->
        pick idx, then replicate x3 channels (engine wants (n, layers, ch));
      - development_time keeps the FULL family (the GUI enumerates it);
      - None-valued optional spectra (midscale, hanatos2025 params) are
        sanitized (zeros / absent) — B&W profiles ship without them."""
    data = dict(data_payload)
    times = data.get('development_time') or []
    idx = development_time_index(times, development_time)

    curves = np.asarray(data['density_curves'], dtype=float)
    if curves.ndim == 2 and curves.shape[1] != 3:
        column = curves[:, min(idx, curves.shape[1] - 1)]
    else:
        column = curves[:, 0] if curves.ndim == 2 else curves
    data['density_curves'] = np.repeat(column[:, None], 3, axis=1)

    base = np.asarray(data.get('base_density'), dtype=float)
    if base.ndim == 2:
        data['base_density'] = base[:, min(idx, base.shape[1] - 1)]

    sensitivity = np.asarray(data['log_sensitivity'], dtype=float)
    data['log_sensitivity'] = np.repeat(sensitivity[:, :1], 3, axis=1)

    dye = np.asarray(data['channel_density'], dtype=float)[:, :1]
    data['channel_density'] = np.concatenate(
        [dye, np.zeros_like(dye), np.zeros_like(dye)], axis=1)

    model = data.get('density_curves_model')
    if isinstance(model, Mapping):
        model = dict(model)
        for key in ('centers', 'amplitudes', 'sigmas'):
            rows = model.get(key) or []
            if rows:
                row = rows[min(idx, len(rows) - 1)]
                model[key] = [list(row)] * 3
        data['density_curves_model'] = model

    layers = np.asarray(data.get('density_curves_layers'), dtype=float)
    if layers.ndim == 3:        # (n, layers, family) -> pick the member
        layers = layers[:, :, min(idx, layers.shape[2] - 1)]
    if layers.ndim == 2:        # (n, layers) -> replicate onto 3 channels
        layers = np.repeat(layers[:, :, None], 3, axis=2)
    data['density_curves_layers'] = layers

    wavelengths = np.asarray(data.get('wavelengths'), dtype=float)
    if data.get('midscale_neutral_density') is None:
        data['midscale_neutral_density'] = np.zeros_like(wavelengths)
    for key in ('hanatos2025_adaptation_window_params',
                'hanatos2025_adaptation_surface_params'):
        if data.get(key) is None:
            data.pop(key, None)
    return data


def profile_from_dict(data: Any) -> Profile:
    if isinstance(data, Profile):
        return data

    if not isinstance(data, Mapping):
        raise TypeError('Unsupported profile payload')

    metadata_payload = data.get('metadata', {})
    info_payload = data.get('info', {})
    data_payload = data.get('data', {})
    if not isinstance(metadata_payload, Mapping):
        raise TypeError("Profile 'metadata' must be a mapping")
    if not isinstance(info_payload, Mapping):
        raise TypeError("Profile 'info' must be a mapping")
    if not isinstance(data_payload, Mapping):
        raise TypeError("Profile 'data' must be a mapping")

    info_payload = dict(info_payload)
    for key in LEGACY_PROFILE_INFO_KEYS:
        info_payload.pop(key, None)

    return Profile(
        metadata=ProfileMetadata(**dict(metadata_payload)),
        info=ProfileInfo(**info_payload),
        data=ProfileData(**dict(data_payload)),
    )


def profile_to_dict(data):
    if is_dataclass(data):
        return {k: profile_to_dict(getattr(data, k)) for k in data.__dataclass_fields__}
    if isinstance(data, dict):
        return {k: profile_to_dict(v) for k, v in data.items()}
    if isinstance(data, list):
        return [profile_to_dict(v) for v in data]
    if isinstance(data, tuple):
        return [profile_to_dict(v) for v in data]
    return data


def _json_safe(data):
    if isinstance(data, dict):
        return {k: _json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_json_safe(v) for v in data]
    if isinstance(data, tuple):
        return [_json_safe(v) for v in data]
    if isinstance(data, np.ndarray):
        return _json_safe(data.tolist())
    if isinstance(data, float) and np.isnan(data):
        return None
    return data


def _validate_profile_info(info, stock):
    if info.type not in PROFILE_TYPES:
        raise ValueError(f"Invalid profile '{stock}': unsupported type={info.type!r}")
    if info.support not in PROFILE_SUPPORTS:
        raise ValueError(f"Invalid profile '{stock}': unsupported support={info.support!r}")
    if info.stage not in PROFILE_STAGES:
        raise ValueError(f"Invalid profile '{stock}': unsupported stage={info.stage!r}")
    if info.use not in PROFILE_USES:
        raise ValueError(f"Invalid profile '{stock}': unsupported use={info.use!r}")
    if info.antihalation not in PROFILE_ANTIHALATION:
        raise ValueError(f"Invalid profile '{stock}': unsupported antihalation={info.antihalation!r}")
    if info.channel_model not in PROFILE_CHANNEL_MODELS:
        raise ValueError(f"Invalid profile '{stock}': unsupported channel_model={info.channel_model!r}")


def _validate_profile(profile, stock):
    try:
        _validate_profile_info(profile.info, stock)
        data = profile.data
        valid = (
            data.log_exposure.ndim == 1
            and data.density_curves.ndim == 2
            and data.density_curves.shape[1] == 3
            and data.density_curves.shape[0] == data.log_exposure.shape[0]
            and data.log_sensitivity.ndim == 2
            and data.log_sensitivity.shape[1] == 3
            and data.wavelengths.ndim == 1
            and data.channel_density.ndim == 2
            and data.channel_density.shape[1] == 3
            and data.channel_density.shape[0] == data.wavelengths.shape[0]
            and data.base_density.ndim == 1
            and data.base_density.shape[0] == data.wavelengths.shape[0]
            and data.midscale_neutral_density.ndim == 1
            and data.midscale_neutral_density.shape[0] == data.wavelengths.shape[0]
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        raise ValueError(f"Invalid profile '{stock}'") from None

    if not valid:
        raise ValueError(f"Invalid profile '{stock}'")

def save_profile(profile, suffix=''):
    profile = copy.deepcopy(profile)
    profile.info.stock = profile.info.stock + suffix
    package = pkg_resources.files('spektrafilm.data.profiles')
    filename = profile.info.stock + '.json'
    resource = package / filename
    print('Saving profile to:', filename)
    with resource.open("w") as file:
        json.dump(_json_safe(profile_to_dict(profile)), file, indent=4, allow_nan=False)

def list_profiles():
    """Return the sorted slugs of all bundled profiles (the JSON file
    stems under ``spektrafilm.data.profiles``)."""
    package = pkg_resources.files('spektrafilm.data.profiles')
    return sorted(
        entry.name[:-len('.json')]
        for entry in package.iterdir()
        if entry.name.endswith('.json')
    )


def load_profile(stock, development_time=None):
    # Phase 10B: 'custom_*' slugs resolve through the user-dir recipe store —
    # the recipe's bundled BASE profile is loaded and the recipe baked into a
    # deep copy. Bundled resources are never touched.
    if str(stock).startswith('custom_'):
        from spektrafilm.profiles.custom_stocks import load_custom_profile
        profile = load_custom_profile(stock)
        _validate_profile(profile, stock)
        return profile
    package = pkg_resources.files('spektrafilm.data.profiles')
    filename = stock + '.json'
    resource = package / filename
    with resource.open("r") as file:
        payload = json.load(file)
    # Phase 12A: B&W profiles resolve to one development-time family member
    # and broadcast their single channel BEFORE validation, so downstream
    # code always sees the standard 3-channel shapes.
    info = payload.get('info', {}) if isinstance(payload, Mapping) else {}
    if isinstance(info, Mapping) and info.get('channel_model') == 'bw':
        payload = dict(payload)
        payload['data'] = _resolve_bw_data(payload.get('data', {}),
                                           development_time)
    profile = profile_from_dict(payload)
    _validate_profile(profile, stock)
    return profile


# Split-architecture aliases.
load_processed_profile = load_profile
save_processed_profile = save_profile

__all__ = [
    "DensityCurvesModel",
    "Profile",
    "ProfileData",
    "ProfileInfo",
    "PROFILE_ANTIHALATION",
    "PROFILE_CHANNEL_MODELS",
    "PROFILE_STAGES",
    "PROFILE_SUPPORTS",
    "PROFILE_TYPES",
    "PROFILE_USES",
    "profile_from_dict",
    "profile_to_dict",
    "list_profiles",
    "load_profile",
    "save_profile",
    "load_processed_profile",
    "save_processed_profile",
]
