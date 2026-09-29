from __future__ import annotations

import datetime
import importlib.resources as _real_pkg_resources
import pathlib as _pathlib


class _PkgResourcesShim:
    @staticmethod
    def files(package: str):
        # Resolve a dotted package path to a real directory on disk,
        # relative to the installed/compiled spektrafilm package.
        parts = package.split('.')
        # io.py lives at spektrafilm/utils/io.py; go up to the spektrafilm parent
        base = _pathlib.Path(__file__).resolve().parent.parent.parent
        return base.joinpath(*parts)


pkg_resources = _PkgResourcesShim()

import json
from dataclasses import dataclass

import exiv2
import numpy as np
import OpenImageIO as oiio
import scipy.interpolate

################################################################################
# Image metadata
################################################################################


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    exif: exiv2.ExifData
    iptc: exiv2.IptcData
    xmp: exiv2.XmpData


def read_image_metadata(filename: str) -> ImageMetadata | None:
    """Read content metadata (EXIF, IPTC, XMP) from an image file.

    Uses the Exiv2 library to read metadata from any format including RAW files.

    Parameters
    ----------
    filename : str
        Path to the image file.

    Returns
    -------
    ImageMetadata or None
        The image metadata, or ``None`` if the file cannot be opened.
    """
    try:
        image = exiv2.ImageFactory.open(filename)
        image.readMetadata()
    except Exception:
        return None

    return ImageMetadata(
        exif=image.exifData(),
        iptc=image.iptcData(),
        xmp=image.xmpData(),
    )


def read_camera_lens_model(metadata: "ImageMetadata | None") -> tuple[str, str]:
    """(camera_model, lens_model) strings from a read ImageMetadata, for
    prefilling the Metadata panel. Camera model is prefixed with the make when
    the model does not already include it. Best-effort; ('', '') on any
    problem so the panel never breaks."""
    if metadata is None:
        return "", ""
    wanted = {
        "Exif.Image.Make": "",
        "Exif.Image.Model": "",
        "Exif.Photo.LensModel": "",
        "Exif.Image.LensModel": "",
    }
    def _value(datum) -> str:
        # Prefer the value-only string; exiv2's str(datum) is "key: value".
        for accessor in ("toString", "print"):
            method = getattr(datum, accessor, None)
            if callable(method):
                try:
                    return str(method()).strip()
                except Exception:
                    pass
        text = str(datum).strip()
        key = ""
        try:
            key = datum.key()
        except Exception:
            key = ""
        if key and text.startswith(key):
            text = text[len(key):].lstrip(": ").strip()
        return text

    try:
        for datum in metadata.exif:
            try:
                key = datum.key()
            except Exception:
                continue
            if key in wanted and not wanted[key]:
                value = _value(datum)
                if value:
                    wanted[key] = value
    except Exception:
        return "", ""

    make = wanted["Exif.Image.Make"]
    model = wanted["Exif.Image.Model"]
    if make and model and make.split()[0].lower() not in model.lower():
        camera = f"{make} {model}".strip()
    else:
        camera = model or make
    lens = wanted["Exif.Photo.LensModel"] or wanted["Exif.Image.LensModel"]
    return camera, lens


def write_image_metadata(
    filename: str,
    source_metadata: ImageMetadata | None = None,
    *,
    saving_color_space: str | None = None,
    saving_cctf_encoding: bool = True,
    custom_tags: dict[str, str] | None = None,
) -> None:
    """Write metadata to an image file after pixel data has been saved.

    Copies any source EXIF, IPTC and XMP tags, then sets overridden tags
    (Orientation, DateTime, Software, pixel dimensions). When
    ``saving_color_space`` is given, also tags the file with the EXIF
    ColorSpace / Interoperability fields that match the saved color space and
    records the human-readable profile name in ``Xmp.photoshop.ICCProfile``.

    Parameters
    ----------
    filename : str
        Path to the output image file (must already exist on disk).
    source_metadata : ImageMetadata, optional
        Metadata returned by ``read_image_metadata`` to copy from the original
        file. Pass ``None`` when there is no source file.
    saving_color_space : str, optional
        Human-readable name of the color space the pixels were encoded in
        (e.g. ``"sRGB"``, ``"Adobe RGB (1998)"``, ``"Display P3"``).
    saving_cctf_encoding : bool, default True
        Whether the saved pixels carry the color space's encoding transfer
        function. ``False`` (linear data) is appended to the recorded profile
        name so downstream tools can flag it.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "exr":
        return

    image_input = oiio.ImageInput.open(filename)
    if image_input is None:
        raise RuntimeError(f"Could not open image file with OpenImageIO: {filename}")

    try:
        spec = image_input.spec()
    finally:
        image_input.close()
    destination = exiv2.ImageFactory.open(filename)
    destination.readMetadata()

    if source_metadata is not None:
        destination.setExifData(source_metadata.exif)
        destination.setIptcData(source_metadata.iptc)
        destination.setXmpData(source_metadata.xmp)

    destination_exif = destination.exifData()

    destination_exif["Exif.Image.Orientation"] = 1
    destination_exif["Exif.Image.DateTime"] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    destination_exif["Exif.Image.Software"] = "spektrafilm"
    destination_exif["Exif.Photo.PixelXDimension"] = spec.width
    destination_exif["Exif.Photo.PixelYDimension"] = spec.height

    # User-supplied metadata (Artist/Copyright/Description/UserComment from the
    # Metadata panel); each value overrides any copied source tag. Individual
    # failures are non-fatal so one bad tag never blocks the write.
    if custom_tags:
        for key, value in custom_tags.items():
            if value is None or str(value) == "":
                continue
            try:
                destination_exif[key] = str(value)
            except Exception:
                pass

    if saving_color_space is not None:
        _set_color_space_tags(
            destination_exif,
            destination.xmpData(),
            saving_color_space,
            saving_cctf_encoding,
        )

    destination.writeMetadata()


# EXIF Photo.ColorSpace values per EXIF 2.32 spec.
_EXIF_COLORSPACE_SRGB = 1
_EXIF_COLORSPACE_UNCALIBRATED = 65535


# Maps (color_space_name, cctf_encoded) -> path inside spektrafilm/data/icc/.
# Filenames preserve the upstream names so they stay traceable to the source
# repos (see data/icc/README.md). Missing entries / files are silently skipped.
_ICC_FILENAMES: dict[tuple[str, bool], str] = {
    # Elle Stone — established RGB working spaces, V2 for broad compatibility.
    ("sRGB", True): "ellelstone/sRGB-elle-V2-srgbtrc.icc",
    ("sRGB", False): "ellelstone/sRGB-elle-V2-g10.icc",
    ("Adobe RGB (1998)", True): "ellelstone/ClayRGB-elle-V2-g22.icc",
    ("Adobe RGB (1998)", False): "ellelstone/ClayRGB-elle-V2-g10.icc",
    ("ProPhoto RGB", True): "ellelstone/LargeRGB-elle-V2-g18.icc",
    ("ProPhoto RGB", False): "ellelstone/LargeRGB-elle-V2-g10.icc",
    ("ITU-R BT.2020", True): "ellelstone/Rec2020-elle-V2-rec709.icc",
    ("ITU-R BT.2020", False): "ellelstone/Rec2020-elle-V2-g10.icc",
    # ACES2065-1 is scene-linear; both flags map to the linear ACES (AP0) file.
    ("ACES2065-1", True): "ellelstone/ACES-elle-V2-g10.icc",
    ("ACES2065-1", False): "ellelstone/ACES-elle-V2-g10.icc",
    # Saucecontrol — P3 variants Elle Stone's set doesn't cover.
    # No compact linear P3 ICC ships upstream; linear variants fall through.
    ("Display P3", True): "saucecontrol/DisplayP3-v2-micro.icc",
    ("DCI-P3", True): "saucecontrol/DCI-P3-v4.icc",
}


def icc_pair_available(color_space: str, cctf_encoding: bool) -> bool:
    """Return True if a bundled ICC profile exists for this (space, encoding).

    The export panel uses this to grey out an encoding choice that has no
    embeddable ICC for the selected colour space (e.g. linear DCI-P3, or an
    encoded ACES2065-1). When False, the file is still written and tagged via
    EXIF/XMP by ``write_image_metadata``; only the embedded ICC is skipped.
    """
    return (color_space, cctf_encoding) in _ICC_FILENAMES


def _load_icc_profile(color_space: str, cctf_encoding: bool) -> bytes | None:
    relative_path = _ICC_FILENAMES.get((color_space, cctf_encoding))
    if relative_path is None:
        return None
    resource = pkg_resources.files("spektrafilm.data.icc").joinpath(*relative_path.split("/"))
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError):
        return None


def _set_color_space_tags(
    exif_data: "exiv2.ExifData",
    xmp_data: "exiv2.XmpData",
    saving_color_space: str,
    saving_cctf_encoding: bool,
) -> None:
    if saving_color_space == "sRGB" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_SRGB
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R98"
    elif saving_color_space == "Adobe RGB (1998)" and saving_cctf_encoding:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED
        exif_data["Exif.Iop.InteroperabilityIndex"] = "R03"
    else:
        exif_data["Exif.Photo.ColorSpace"] = _EXIF_COLORSPACE_UNCALIBRATED

    profile_name = saving_color_space if saving_cctf_encoding else f"{saving_color_space} (linear)"
    xmp_data["Xmp.photoshop.ICCProfile"] = profile_name


################################################################################
# 16-bit PNG I/O
################################################################################

def load_image_oiio(filename):
    # Open the image file
    in_img = oiio.ImageInput.open(filename)
    if not in_img:
        raise IOError("Could not open image file: " + filename)
    
    try:
        spec = in_img.spec()
        
        # Determine the native pixel format:
        # Use "uint16" for PNG and "half" for EXR if applicable.
        if spec.format == oiio.TypeDesc("uint8"): # for compatibility
            read_type = oiio.TypeDesc("uint8")
        elif spec.format == oiio.TypeDesc("uint16"):
            read_type = oiio.TypeDesc("uint16")
        elif spec.format == oiio.TypeDesc("half"):
            read_type = oiio.TypeDesc("half")
        elif spec.format == oiio.TypeDesc("float"):
            read_type = oiio.TypeDesc("float")
        else:
            # Fallback: use "uint16" by default. You might choose "float" if desired.
            read_type = oiio.TypeDesc("uint16")
        
        # Read the image data using the chosen type
        pixels = in_img.read_image(read_type)
        if pixels is None:
            raise Exception("Failed to read image data from " + filename)
        
        # Convert the raw data to a NumPy array and reshape it
        np_pixels = np.array(pixels)
        np_pixels = np_pixels.reshape(spec.height, spec.width, spec.nchannels)
        
        if spec.format == oiio.TypeDesc("uint16"):
            np_pixels = np.double(np_pixels)/(2**16-1)
        if spec.format == oiio.TypeDesc("uint8"):
            np_pixels = np.double(np_pixels)/(2**8-1)

        # Grayscale (1 channel) and gray+alpha (2 channels) sources: the
        # whole pipeline is 3-channel — replicate the luminance so a gray
        # TIFF/PNG imports as a neutral RGB image instead of a 1-channel
        # array that breaks the [..., :3] slices downstream.
        if np_pixels.ndim == 3 and np_pixels.shape[2] in (1, 2):
            np_pixels = np.repeat(np_pixels[:, :, :1], 3, axis=2)

        return np_pixels
    finally:
        in_img.close()

def save_image_oiio(
    filename,
    image_data,
    bit_depth=16,
    *,
    color_space: str | None = None,
    cctf_encoding: bool = True,
):
    """Save a 3-channel image to disk via OpenImageIO.

    Pixel format per extension:

    - ``.jpg`` / ``.jpeg``: clipped to [0, 1] and written as uint8.
      ``bit_depth`` is ignored.
    - ``.png``: ``bit_depth`` selects the encoding (clipped to [0, 1]) —
      16 → uint16 (true 16-bit, scaled to [0, 65535]),
      8 → uint8 (scaled to [0, 255]). 16-bit PNG comes straight from the
      float render, never an 8-bit buffer promoted to 16-bit.
    - ``.tif`` / ``.tiff``: ``bit_depth`` selects the encoding —
      8 → uint8 (clipped, scaled to [0, 255]),
      16 → uint16 (clipped, scaled to [0, 65535]),
      32 → float32 (raw, no clip/scale). Written with ZIP/deflate
      compression.
    - ``.exr``: ``bit_depth`` selects the encoding —
      16 → half (float16), 32 → float32. Always raw, no clip/scale.

    With the default ``bit_depth=16`` this gives half EXR and uint16 TIFF —
    the idiomatic precision for each format. Pass ``bit_depth=32`` for
    float32 in either, or ``bit_depth=8`` for uint8 TIFF.

    When ``color_space`` is provided and a matching ICC profile exists in
    ``spektrafilm/data/icc/`` (see the table in ``_ICC_FILENAMES``), the
    profile bytes are embedded into the file's native ICC slot:
    JPEG APP2 marker, PNG iCCP chunk, or TIFF ICCProfile tag. EXR carries
    its own color metadata so ICC embedding is skipped there. Missing
    profiles fall back to no embedding — the EXIF/XMP color-space tagging
    written by ``write_image_metadata`` still labels the file.

    Parameters
    ----------
    filename : str
        Output path; the extension selects the file format.
    image_data : np.ndarray
        Image data with shape ``(height, width, 3)``. Floating-point input
        is assumed to be in [0, 1] for integer-encoded formats.
    bit_depth : int, default 16
        Precision selector for TIFF and EXR (see above). Ignored for JPEG
        and PNG.
    color_space : str, optional
        Name of the color space the pixels are encoded in (e.g. ``"sRGB"``,
        ``"Display P3"``). Used to look up the ICC profile to embed.
    cctf_encoding : bool, default True
        Whether the pixels carry the color space's encoding transfer
        function. Affects which ICC variant is embedded (encoded vs linear).
    """
    # Extract image dimensions and number of channels
    height, width, nchannels = image_data.shape

    # Determine file type based on extension
    ext = filename.split('.')[-1].lower()
    
    # Create an ImageSpec with the proper data type
    if ext == "png":
        # ``bit_depth`` selects the encoding (assumes image_data in [0, 1]):
        #   16 -> uint16 (true 16-bit, scaled to [0, 65535]) -- the high-precision
        #         scan base; comes straight from the float render, never an 8-bit
        #         buffer promoted to 16-bit.
        #    8 -> uint8 (scaled to [0, 255]).
        if bit_depth == 16:
            img = (np.clip(image_data, 0, 1) * 65535.0).astype(np.uint16)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint16"))
        elif bit_depth == 8:
            img = (np.clip(image_data, 0, 1) * 255.0).astype(np.uint8)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
        else:
            raise ValueError(f"Unsupported bit_depth for PNG: {bit_depth}")
        data_to_write = img
    elif ext in {"jpg", "jpeg"}:
        img_uint8 = np.clip(image_data, 0, 1) * 255.0
        img_uint8 = img_uint8.astype(np.uint8)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
        data_to_write = img_uint8
    elif ext=="exr" and bit_depth==16:
        # Convert the image data to 16-bit half precision.
        # Note: numpy's float16 is used here; OpenImageIO accepts "half" for 16-bit floats.
        img_half = image_data.astype(np.float16)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("half"))
        data_to_write = img_half
    elif ext=='exr' and bit_depth==32:
        # Convert the image data to 32-bit float precision.
        # Note: numpy's float32 is used here; OpenImageIO accepts "float" for 32-bit floats.
        img_float = image_data.astype(np.float32)
        spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("float"))
        data_to_write = img_float
    elif ext in {"tif", "tiff"}:
        if bit_depth == 8:
            img = (np.clip(image_data, 0, 1) * 255.0).astype(np.uint8)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint8"))
        elif bit_depth == 16:
            img = (np.clip(image_data, 0, 1) * 65535.0).astype(np.uint16)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("uint16"))
        elif bit_depth == 32:
            img = image_data.astype(np.float32)
            spec = oiio.ImageSpec(width, height, nchannels, oiio.TypeDesc("float"))
        else:
            raise ValueError(f"Unsupported bit_depth for TIFF: {bit_depth}")
        # ZIP/deflate is lossless and works for all bit depths (LZW is faster but
        # integer-only); a TIFF of a 4K float image is ~100 MB uncompressed.
        spec.attribute("Compression", "zip")
        data_to_write = img
    else:
        raise ValueError("Unsupported file extension: " + ext)

    if color_space is not None and ext != "exr":
        icc_bytes = _load_icc_profile(color_space, cctf_encoding)
        if icc_bytes is not None:
            icc_array = np.frombuffer(icc_bytes, dtype=np.uint8)
            spec.attribute(
                "ICCProfile",
                oiio.TypeDesc(f"uint8[{icc_array.size}]"),
                icc_array,
            )

    # Create an ImageOutput for writing the file
    out = oiio.ImageOutput.create(filename)
    if not out:
        raise IOError("Could not create output image: " + filename)
    
    try:
        out.open(filename, spec)
        # Write the image data; write_image accepts the NumPy array directly.
        out.write_image(data_to_write)
    finally:
        out.close()

################################################################################
# Neutral filter values
################################################################################

NEUTRAL_PRINT_FILTERS_FILENAME = 'neutral_print_filters.json'


def save_neutral_print_filters(neutral_print_filters):
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("w") as file:
        json.dump(neutral_print_filters, file, indent=4)


def read_neutral_print_filters():
    package = pkg_resources.files('spektrafilm.data.filters')
    resource = package / NEUTRAL_PRINT_FILTERS_FILENAME
    with resource.open("r") as file:
        return json.load(file)

################################################################################
# Profiles
################################################################################

def load_dichroic_filters(wavelengths, brand='thorlabs'):
    channels = ['c','m','y']
    filters = np.zeros((np.size(wavelengths), 3))
    for i, channel in enumerate(channels):
        package = pkg_resources.files('spektrafilm.data.filters.dichroics')
        filename = brand+'/filter_'+channel+'.csv'
        resource = package / filename
        with resource.open("r") as file:
            data = np.loadtxt(file, delimiter=',')
            unique_index = np.unique(data[:,0], return_index=True)[1]
            data = data[unique_index,:]
            # filters[:,i] = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/100)(wavelengths)
            filters[:,i] = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/100)(wavelengths)
    return filters

def load_filter(wavelengths, name='KG3', brand='schott', filter_type='heat_absorbing', percent_transmittance=False):
    transmittance = np.zeros_like(wavelengths)
    package = pkg_resources.files('spektrafilm.data.filters.'+filter_type)
    filename = brand+'/'+name+'.csv'
    resource = package / filename
    if percent_transmittance: scale = 100
    else: scale = 1
    with resource.open("r") as file:
        data = np.loadtxt(file, delimiter=',')
        unique_index = np.unique(data[:,0], return_index=True)[1]
        data = data[unique_index,:]
        # transmittance = scipy.interpolate.CubicSpline(data[:,0], data[:,1]/scale)(wavelengths)
        transmittance = scipy.interpolate.Akima1DInterpolator(data[:,0], data[:,1]/scale)(wavelengths)
    return transmittance
