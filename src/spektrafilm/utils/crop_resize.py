import numpy as np
import skimage.transform


def apply_desqueeze(image, desqueeze=0.0):
    """Anamorphic desqueeze, applied BEFORE skew and crop (restores the true
    scene geometry of footage shot through an anamorphic element).

    ``desqueeze`` in [-2, 2]: positive stretches the WIDTH by the factor
    (2.0 = classic 2x anamorphic), negative stretches the HEIGHT by |factor|
    (for material squeezed the other way). 0, 1 and -1 are identity (off).
    Values between 0 and 1 compress instead of stretching. Bicubic, matching
    the upscale path.
    """
    factor = float(desqueeze)
    if factor in (0.0, 1.0, -1.0):
        return image
    if factor > 0:
        scale = (1.0, factor)
    else:
        scale = (-factor, 1.0)
    return skimage.transform.rescale(image, scale, channel_axis=2, order=3)


def apply_skew(image, skew_deg=0.0):
    """Rotate the full frame by ``skew_deg`` (degrees, CCW positive, +-45)
    about the image center, BEFORE the crop — so the crop rectangle stays
    axis-aligned while the user straightens or tilts the framing.

    Matches NegPy's fine-rotation transform: full-frame affine at scale 1.0
    (no zoom-to-fill), bilinear interpolation, edges replicated. Corners that
    rotate out of frame are meant to be cut away by the crop.
    """
    angle = float(skew_deg)
    if angle == 0.0:
        return image
    return skimage.transform.rotate(
        image, angle, resize=False, center=None, order=1, mode='edge')


def crop_geometry_output_shape(shape_hw, desqueeze=0.0):
    """(height, width) after apply_desqueeze (skew preserves the shape).

    Mirrors skimage.transform.rescale's output rounding so pixel mapping and
    the actual transform agree.
    """
    height, width = int(shape_hw[0]), int(shape_hw[1])
    factor = float(desqueeze)
    if factor in (0.0, 1.0, -1.0):
        return height, width
    if factor > 0:
        return height, int(round(width * factor))
    return int(round(height * -factor)), width


def map_point_through_crop_geometry(point_xy, shape_hw, desqueeze=0.0, skew_deg=0.0):
    """Map a PRE-transform pixel position (x, y) to its POST-transform pixel
    position after apply_desqueeze + apply_skew (the same order
    crop_and_rescale applies them). This is how an interactively drawn crop
    corner is re-projected so the axis-aligned crop cuts the region the user
    actually pointed at (NegPy's map_coords_to_geometry role).

    Desqueeze scales the coordinate with the image; skew rotates it about the
    image center. skimage.transform.rotate(image, angle) displays the content
    rotated CCW (for y-down arrays a feature moves by the CW rotation matrix
    in (x, y) pixel coords), about center ((w-1)/2, (h-1)/2) — validated
    against a rendered marker in the tests.
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    factor = float(desqueeze)
    if factor not in (0.0, 1.0, -1.0):
        # skimage.rescale samples output pixel x' at input (x'+0.5)/f - 0.5,
        # so a feature at x lands at (x + 0.5) * f - 0.5.
        if factor > 0:
            x = (x + 0.5) * factor - 0.5
        else:
            y = (y + 0.5) * -factor - 0.5
    height, width = crop_geometry_output_shape(shape_hw, desqueeze)
    angle = np.deg2rad(float(skew_deg))
    if angle != 0.0:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        dx = x - cx
        dy = y - cy
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Positive skew shows the content rotated CCW on screen; in the y-down
        # (x, y) pixel frame that is the CW coordinate rotation. Signs verified
        # against a rendered marker through apply_skew (see the tests).
        x = cx + dx * cos_a + dy * sin_a
        y = cy - dx * sin_a + dy * cos_a
    return x, y


def crop_image(image, center=(0.5,0.5), size=(0.1, 0.1)):
    """
    Crop an image based on a specified fraction and center.

    Parameters:
    image (numpy.ndarray): The input image to be cropped.
    center (tuple of float, optional): The center of the cropping area as a tuple of two floats (x, y). 
                                      Each value should be between 0 and 1. Default is (0.5, 0.5).
    size (tuple of float, optional): The normalize size of the cropped area as fraction of the long side, (x,y). Default is (0.1, 0.1).

    Returns:
    numpy.ndarray: The cropped image.
    """
    center = np.flip(center)
    shape = image.shape[0:2]
    cn = np.round(shape*np.array(center))
    sz = np.round(np.double(np.max(shape))*np.flip(np.array(size)))
    x0 = np.round(cn - sz/2)
    sz = np.int64(sz)
    x0 = np.int64(x0)
    x0[x0<0] = 0
    if x0[0]+sz[0]>shape[0]: x0[0] = shape[0]-sz[0]
    if x0[1]+sz[1]>shape[1]: x0[1] = shape[1]-sz[1]
    image_crop = image[x0[0]:x0[0]+sz[0], x0[1]:x0[1]+sz[1],:]
    return image_crop

# def resize_image(image, resize_factor=1.0): #TBD
#     """
#     Resize the given image by a specified factor.
#     Parameters:
#     image (numpy.ndarray): The image to be resized.
#     resize_factor (float, optional): The factor by which to resize the image. 
#                                      Default is 1.0 (no resizing).
#     Returns:
#     numpy.ndarray: The resized image.
#     """
#     # zoom(image, zoom=(resize_factor, resize_factor, 1.0))
#     return skimage.transform.rescale(image, resize_factor, channel_axis=2)