"""Phase 10B: custom film stocks — the .sfstock recipe store and bake engine.

A custom stock is a RECIPE, not a raw profile: a small versioned JSON that
names a bundled BASE stock and describes how its density curves (and,
optionally, its grain/halation/coupler look) are replaced. At load time the
base profile is deep-copied and the recipe is BAKED into the copy — the
camera-side spectral data (log_sensitivity, channel_density, base_density,
hanatos2025 adaptation) comes from the base stock UNCHANGED. That mismatch
freedom (slide-film dyes + negative-family curves + C41 paper) is the honest
cross-processing model; editing the spectra themselves is OUT OF SCOPE
(the hanatos2025 adaptation params are fitted per stock to its
sensitivities and go stale if they change — a spectral-editing refit
pipeline is a separate later feasibility study).

Storage is a USER-WRITABLE directory (never the package resources — the
frozen build's install folder is not for user data):

    %APPDATA%/spektrafilm/stocks/<slug>.sfstock      (Windows)
    ~/.spektrafilm/stocks/<slug>.sfstock             (fallback)
    $SPEKTRAFILM_STOCKS_DIR                          (override, also the test hook)

Curve modes:
  - 'base'        keep the base stock's curves untouched. An all-neutral
                  recipe ('base' + no fog + no look overrides) bakes to a
                  profile whose data arrays are BIT-IDENTICAL to the base
                  (test-pinned) — only the identity/metadata differ.
  - 'parametric'  replace the curves with the hyperbolic-log family
                  (model/parametric.py) per channel: gamma, speed_offset_ev
                  (shifts log_exposure_0 against the base half-density
                  anchor), density_max, toe_size, shoulder_size, plus an
                  additive toe-weighted fog floor (same form as expiration:
                  D' = D + fog*(1 - D/D_max), so D_max is preserved).
  - 'spline'      drawn control points (Phase 10D; format reserved here).

Morph compatibility: push/pull chemistry and expiration act THROUGH
DensityCurvesModel, so baking always fits a fresh norm_cdfs decomposition to
the generated (fog-free) curve — the morph engines compose on custom stocks
exactly as on bundled ones. NOTE the same axis separation as Phase 7: the
experimental chemistry morph regenerates curves FROM the model, so recipe
fog (a sampled-array axis, like aging's fog) rides on top only while
chemistry is inactive; aging's own fog still applies after the morph.

Licensing: bundled profiles are CC BY-SA 4.0. Every save and every bake
stamps derived-from metadata (base slug, modification note, license
preserved) IN CODE — see _stamp_recipe_license / _stamp_profile_metadata.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from spektrafilm.profiles.io import (
    Profile,
    PROFILE_ANTIHALATION,
    PROFILE_TYPES,
    PROFILE_USES,
    list_profiles,
    load_profile as _load_bundled_profile,
)

__all__ = [
    "CUSTOM_SLUG_PREFIX",
    "STOCK_FILE_SUFFIX",
    "SCHEMA_VERSION",
    "StockRecipe",
    "ParametricCurves",
    "bake_recipe_into_profile",
    "delete_custom_stock",
    "is_custom_slug",
    "list_custom_stocks",
    "load_custom_profile",
    "load_custom_stock",
    "make_slug",
    "save_custom_stock",
    "spline_fit_residual",
    "user_stocks_dir",
]

CUSTOM_SLUG_PREFIX = "custom_"
STOCK_FILE_SUFFIX = ".sfstock"
SCHEMA_VERSION = 1

_CURVE_MODES = ("base", "parametric", "spline")
# Look-override targets: partial field dicts applied onto params.film_render
# at digest time (the same moment _apply_film_specifics runs).
_LOOK_TARGETS = ("grain", "halation", "dir_couplers")


# --------------------------------------------------------------------------- #
# recipe dataclasses
# --------------------------------------------------------------------------- #
def _neutral_triplet() -> tuple:
    return (1.0, 1.0, 1.0)


def _zero_triplet() -> tuple:
    return (0.0, 0.0, 0.0)


@dataclass
class ParametricCurves:
    """Per-channel (R, G, B) parameters of the hyperbolic-log curve family."""

    gamma: tuple = (0.65, 0.65, 0.65)
    speed_offset_ev: tuple = _zero_triplet()
    density_max: tuple = (2.2, 2.2, 2.2)
    toe_size: tuple = (0.3, 0.3, 0.3)
    shoulder_size: tuple = (0.3, 0.3, 0.3)
    fog: tuple = _zero_triplet()


@dataclass
class StockRecipe:
    name: str
    base_stock: str
    slug: str = ""
    type: str = ""                    # '' = inherit from the base stock
    use: str = ""
    antihalation: str = ""
    target_print: str | None = None
    curve_mode: str = "base"
    parametric: ParametricCurves = field(default_factory=ParametricCurves)
    spline_points: tuple = ()          # Phase 10D: per-channel (x, y) points
    look: dict = field(default_factory=dict)   # {'grain': {...}, ...} partial
    schema_version: int = SCHEMA_VERSION
    app_version: str = ""
    license: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.slug:
            self.slug = make_slug(self.name)


def make_slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("Custom stock name must contain letters or digits.")
    return CUSTOM_SLUG_PREFIX + cleaned


def is_custom_slug(slug: str) -> bool:
    return str(slug).startswith(CUSTOM_SLUG_PREFIX)


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
def user_stocks_dir() -> Path:
    override = os.environ.get("SPEKTRAFILM_STOCKS_DIR")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "spektrafilm" / "stocks"
    return Path.home() / ".spektrafilm" / "stocks"


def _recipe_path(slug: str) -> Path:
    return user_stocks_dir() / f"{slug}{STOCK_FILE_SUFFIX}"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("spektrafilm")
    except Exception:
        return "0+unknown"


def _stamp_recipe_license(recipe: StockRecipe) -> None:
    """Derived-work stamping, in code on every save (CC BY-SA 4.0)."""
    recipe.license = {
        "license": "CC BY-SA 4.0",
        "derived_from": recipe.base_stock,
        "modification": (
            f"Custom stock recipe '{recipe.name}' derived from the spektrafilm "
            f"profile '{recipe.base_stock}' (curve mode: {recipe.curve_mode}; "
            "spectral data unchanged)."
        ),
        "attribution": (
            "Base profile by Andrea Volpato, spektrafilm project "
            "(https://github.com/andreavolpato/spektrafilm), CC BY-SA 4.0. "
            "This derivative preserves that license."
        ),
        "stamped": date.today().isoformat(),
    }


def save_custom_stock(recipe: StockRecipe) -> Path:
    _validate_recipe(recipe)
    if recipe.base_stock in (recipe.slug,):
        raise ValueError("A custom stock cannot use itself as base.")
    if recipe.slug in list_profiles():
        raise ValueError(
            f"Slug '{recipe.slug}' collides with a bundled profile.")
    recipe.app_version = _package_version()
    _stamp_recipe_license(recipe)
    path = _recipe_path(recipe.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "sfstock", **asdict(recipe)}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    return path


def load_custom_stock(slug: str) -> StockRecipe:
    path = _recipe_path(slug)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _recipe_from_dict(data)


def list_custom_stocks() -> list[str]:
    directory = user_stocks_dir()
    if not directory.is_dir():
        return []
    return sorted(
        entry.stem for entry in directory.iterdir()
        if entry.suffix == STOCK_FILE_SUFFIX and entry.is_file()
    )


def delete_custom_stock(slug: str) -> bool:
    path = _recipe_path(slug)
    if path.is_file():
        path.unlink()
        return True
    return False


def _recipe_from_dict(data: dict[str, Any]) -> StockRecipe:
    if not isinstance(data, dict):
        raise ValueError("A .sfstock file must contain a JSON object.")
    if data.get("format", "sfstock") != "sfstock":
        raise ValueError("Not a .sfstock recipe file.")
    version = int(data.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Recipe schema_version {version} is newer than this app "
            f"supports ({SCHEMA_VERSION}).")
    parametric_data = data.get("parametric") or {}
    parametric = ParametricCurves(**{
        key: tuple(float(v) for v in value)
        for key, value in parametric_data.items()
        if key in ParametricCurves.__dataclass_fields__
    })
    recipe = StockRecipe(
        name=str(data.get("name", "")),
        base_stock=str(data.get("base_stock", "")),
        slug=str(data.get("slug", "")),
        type=str(data.get("type", "") or ""),
        use=str(data.get("use", "") or ""),
        antihalation=str(data.get("antihalation", "") or ""),
        target_print=data.get("target_print"),
        curve_mode=str(data.get("curve_mode", "base")),
        parametric=parametric,
        spline_points=tuple(
            tuple(tuple(float(v) for v in point) for point in channel)
            for channel in (data.get("spline_points", ()) or ())
        ),
        look=dict(data.get("look", {}) or {}),
        schema_version=version,
        app_version=str(data.get("app_version", "")),
        license=dict(data.get("license", {}) or {}),
    )
    _validate_recipe(recipe)
    return recipe


def _validate_recipe(recipe: StockRecipe) -> None:
    if not recipe.name:
        raise ValueError("Recipe needs a name.")
    if not recipe.base_stock:
        raise ValueError("Recipe needs a base_stock slug.")
    if is_custom_slug(recipe.base_stock):
        raise ValueError("base_stock must be a bundled profile, not a custom stock.")
    if not is_custom_slug(recipe.slug):
        raise ValueError(f"Custom slugs must start with '{CUSTOM_SLUG_PREFIX}'.")
    if recipe.curve_mode not in _CURVE_MODES:
        raise ValueError(f"curve_mode must be one of {_CURVE_MODES}.")
    if recipe.type and recipe.type not in PROFILE_TYPES:
        raise ValueError(f"type must be one of {sorted(PROFILE_TYPES)} or ''.")
    if recipe.use and recipe.use not in PROFILE_USES:
        raise ValueError(f"use must be one of {sorted(PROFILE_USES)} or ''.")
    if recipe.antihalation and recipe.antihalation not in PROFILE_ANTIHALATION:
        raise ValueError(
            f"antihalation must be one of {sorted(PROFILE_ANTIHALATION)} or ''.")
    if recipe.curve_mode == "spline":
        points = recipe.spline_points
        if len(points) != 3:
            raise ValueError(
                "spline curve_mode needs per-channel points: a 3-tuple of "
                "point lists (R, G, B).")
        for ch, channel_points in enumerate(points):
            if len(channel_points) < 2:
                raise ValueError(
                    f"spline channel {ch} needs at least 2 control points.")
            for point in channel_points:
                if len(point) != 2:
                    raise ValueError(
                        f"spline channel {ch} has a malformed point {point!r}.")
    unknown_look = set(recipe.look) - set(_LOOK_TARGETS)
    if unknown_look:
        raise ValueError(f"Unknown look override targets: {sorted(unknown_look)}.")
    for target, overrides in recipe.look.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"look['{target}'] must be a dict of field overrides.")


# --------------------------------------------------------------------------- #
# baking
# --------------------------------------------------------------------------- #
def _base_half_density_anchor(profile: Profile) -> np.ndarray:
    """Per-channel log exposure where the BASE curve crosses half of its
    density range — the anchor speed_offset_ev shifts against."""
    x = np.asarray(profile.data.log_exposure, dtype=float)
    curves = np.asarray(profile.data.density_curves, dtype=float)
    anchors = np.empty(curves.shape[1])
    for channel in range(curves.shape[1]):
        y = curves[:, channel]
        half = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
        finite = np.isfinite(y)
        yc, xc = y[finite], x[finite]
        if yc[0] > yc[-1]:      # positive stock: falling curve
            yc, xc = yc[::-1], xc[::-1]
        anchors[channel] = float(np.interp(half, yc, xc))
    return anchors


def _parametric_curves(recipe: StockRecipe, profile: Profile) -> np.ndarray:
    from spektrafilm.model.parametric import parametric_density_curves_model

    p = recipe.parametric
    x = np.asarray(profile.data.log_exposure, dtype=float)
    profile_type = recipe.type or profile.info.type
    anchors = _base_half_density_anchor(profile)

    gamma = np.asarray(p.gamma, dtype=float)
    density_max = np.asarray(p.density_max, dtype=float)
    # anchor: the family's half-density point sits at logE0 + dmax/(2*gamma);
    # speed_offset_ev shifts it (positive EV = faster film = less exposure).
    log_exposure_0 = (
        anchors
        - density_max / (2.0 * gamma)
        - np.asarray(p.speed_offset_ev, dtype=float) * math.log10(2.0)
    )
    if profile_type == "positive":
        # falling curves: evaluate the rising family on mirrored exposure
        x_eval = (2.0 * np.mean(anchors)) - x
    else:
        x_eval = x
    curves = parametric_density_curves_model(
        x_eval,
        gamma,
        log_exposure_0,
        density_max,
        np.asarray(p.toe_size, dtype=float),
        np.asarray(p.shoulder_size, dtype=float),
    )
    return curves


def _spline_curves(recipe: StockRecipe, profile: Profile) -> np.ndarray:
    """Phase 10D: sample the drawn free-mode points onto the profile's
    log-exposure grid. Per channel: PCHIP monotone interpolation through the
    x-sorted points, FLAT extension outside the control range, non-negative,
    and monotone in the stock's direction (rising for negatives, falling for
    positives) — the physics constraints of the free tier. The drawn points
    are the BAKED truth; the norm_cdfs fit-back (engine truth for the morph
    engines) happens in bake_recipe_into_profile."""
    from scipy.interpolate import PchipInterpolator

    x_grid = np.asarray(profile.data.log_exposure, dtype=float)
    profile_type = recipe.type or profile.info.type
    curves = np.empty((x_grid.size, 3))
    for ch in range(3):
        points = sorted(
            ((float(px), float(py)) for px, py in recipe.spline_points[ch]),
            key=lambda p: p[0])
        xs = np.array([p[0] for p in points])
        ys = np.maximum(np.array([p[1] for p in points]), 0.0)
        xs, keep = np.unique(xs, return_index=True)
        ys = ys[keep]
        # monotone in the stock's direction (PCHIP then preserves it)
        ys = (np.maximum.accumulate(ys) if profile_type != 'positive'
              else np.minimum.accumulate(ys))
        if xs.size < 2:
            curves[:, ch] = ys[0] if ys.size else 0.0
            continue
        sampled = PchipInterpolator(xs, ys)(np.clip(x_grid, xs[0], xs[-1]))
        curves[:, ch] = np.maximum(sampled, 0.0)
    return curves


def spline_fit_residual(log_exposure, drawn_curves, model, profile_type) -> tuple:
    """Per-channel RMS between the drawn curves (baked truth) and the
    norm_cdfs fit (engine truth) — the free tier's 'how film-like is this
    drawing' number, shown in the editor."""
    from spektrafilm.utils.morph_curves import apply_print_curves_morph_with_layers

    generated, _layers = apply_print_curves_morph_with_layers(
        log_exposure, model, SimpleNamespace(active=False),
        profile_type=profile_type)
    diff = np.asarray(drawn_curves, dtype=float) - np.asarray(generated, dtype=float)
    return tuple(float(np.sqrt(np.nanmean(diff[:, ch] ** 2))) for ch in range(3))


def _apply_fog(curves: np.ndarray, fog: tuple) -> np.ndarray:
    """Toe-weighted additive floor, expiration's form: D_max preserved."""
    fog_arr = np.asarray(fog, dtype=float)
    if not np.any(fog_arr > 0.0):
        return curves
    out = curves.copy()
    for channel in range(out.shape[1]):
        f = float(fog_arr[channel])
        if f <= 0.0:
            continue
        y = out[:, channel]
        d_max = float(np.nanmax(y))
        if d_max <= 0.0:
            continue
        f = min(f, 0.9 * d_max)
        out[:, channel] = y + f * (1.0 - y / d_max)
    return out


def _stamp_profile_metadata(profile: Profile, recipe: StockRecipe) -> None:
    metadata = profile.metadata
    metadata.copyright = (
        f"Derivative work: custom stock '{recipe.name}' derived from the "
        f"spektrafilm profile '{recipe.base_stock}'. {metadata.copyright}"
    )
    metadata.license = (
        "Derived profile — CC BY-SA 4.0 preserved from the base profile. "
        f"Modification: recipe-baked curves (mode: {recipe.curve_mode}), "
        "spectral data unchanged. " + metadata.license
    )
    metadata.created = date.today().isoformat()


def bake_recipe_into_profile(base_profile: Profile, recipe: StockRecipe) -> Profile:
    """Deep-copy the base profile and bake the recipe into the copy.

    Identity/metadata are always stamped; the DATA arrays change only for
    what the recipe actually overrides, so an all-neutral recipe ('base'
    curves, no look) keeps every array bit-identical to the base.
    """
    _validate_recipe(recipe)
    profile = copy.deepcopy(base_profile)
    profile.info.stock = recipe.slug
    profile.info.name = recipe.name
    if recipe.type:
        profile.info.type = recipe.type
    if recipe.use:
        profile.info.use = recipe.use
    if recipe.antihalation:
        profile.info.antihalation = recipe.antihalation
    if recipe.target_print is not None:
        profile.info.target_print = recipe.target_print
    profile.info.custom_look = {
        target: dict(overrides) for target, overrides in recipe.look.items()
    } if recipe.look else None
    profile.info.custom_base = recipe.base_stock
    _stamp_profile_metadata(profile, recipe)

    if recipe.curve_mode == "base":
        return profile

    from spektrafilm.model.curve_fit import fit_density_curves_model
    from spektrafilm.utils.morph_curves import apply_print_curves_morph_with_layers

    profile_type = profile.info.type
    if recipe.curve_mode == "spline":
        # Phase 10D free tier: the DRAWN points are the baked truth (no fog
        # axis — the drawing includes its own floor); the fitted model is
        # the engine truth the morph engines (push/pull, aging) act through.
        curves = _spline_curves(recipe, profile)
        model, _residuals = fit_density_curves_model(
            profile.data.log_exposure, curves,
            n_layers=3, profile_type=profile_type,
        )
    else:
        curves = _parametric_curves(recipe, profile)
        model, _residuals = fit_density_curves_model(
            profile.data.log_exposure, curves,
            n_layers=3, profile_type=profile_type,
        )
        curves = _apply_fog(curves, recipe.parametric.fog)

    _total, layers = apply_print_curves_morph_with_layers(
        profile.data.log_exposure,
        model,
        SimpleNamespace(active=False),
        profile_type=profile_type,
    )
    profile.data.density_curves = curves
    profile.data.density_curves_layers = layers
    profile.data.density_curves_model = model
    return profile


def fit_parametric_curves(profile: Profile) -> ParametricCurves:
    """Fit the hyperbolic-log family to a profile's curves — the seed the
    stock editor (Phase 10C) opens with, so its handles start ON the base
    stock's characteristic curve. Deterministic (least_squares, fixed
    initialization), per channel; speed_offset_ev stays 0 by construction
    because the anchor is the base half-density crossing itself."""
    from scipy.optimize import least_squares

    from spektrafilm.model.parametric import parametric_density_curves_model

    x = np.asarray(profile.data.log_exposure, dtype=float)
    curves = np.asarray(profile.data.density_curves, dtype=float)
    profile_type = profile.info.type
    anchors = _base_half_density_anchor(profile)
    x_eval = (2.0 * np.mean(anchors)) - x if profile_type == "positive" else x

    gammas, d_maxes, toes, shoulders, fogs = [], [], [], [], []
    for channel in range(curves.shape[1]):
        y = curves[:, channel]
        finite = np.isfinite(y)
        yc, xc = y[finite], x_eval[finite]
        order = np.argsort(xc)
        xc, yc = xc[order], yc[order]
        fog0 = float(yc.min())
        d_range = max(float(yc.max()) - fog0, 1e-3)
        slope = np.gradient(yc, xc)
        gamma0 = max(float(np.nanmax(slope)), 0.1)
        anchor = float(anchors[channel])

        def model(v):
            gamma, d_max, toe, shoulder, loge0 = v
            curve = parametric_density_curves_model(
                xc, [gamma] * 3, [loge0] * 3, [d_max] * 3, [toe] * 3, [shoulder] * 3,
            )[:, 0]
            return curve + fog0 - yc

        guess = np.array([gamma0, d_range, 0.3, 0.3,
                          anchor - d_range / (2.0 * gamma0)])
        lower = np.array([0.05, 0.1, 0.05, 0.05, float(xc.min()) - 5.0])
        upper = np.array([5.0, 6.0, 3.0, 3.0, float(xc.max()) + 5.0])
        fit = least_squares(model, np.clip(guess, lower, upper),
                            bounds=(lower, upper), max_nfev=2000)
        gamma, d_max, toe, shoulder, _loge0 = fit.x
        gammas.append(round(float(gamma), 3))
        d_maxes.append(round(float(d_max), 3))
        toes.append(round(float(toe), 3))
        shoulders.append(round(float(shoulder), 3))
        fogs.append(round(max(fog0, 0.0), 3))

    return ParametricCurves(
        gamma=tuple(gammas),
        speed_offset_ev=(0.0, 0.0, 0.0),
        density_max=tuple(d_maxes),
        toe_size=tuple(toes),
        shoulder_size=tuple(shoulders),
        fog=tuple(fogs),
    )


def load_custom_profile(slug: str) -> Profile:
    """Resolve a custom slug: load its recipe, load the bundled base, bake.

    A missing base stock raises ValueError with a message naming both slugs
    (the GUI turns that into a relink prompt in Phase 10C)."""
    recipe = load_custom_stock(slug)
    try:
        base = _load_bundled_profile(recipe.base_stock)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Custom stock '{slug}' needs base profile "
            f"'{recipe.base_stock}', which is not available.") from exc
    return bake_recipe_into_profile(base, recipe)
