"""
Image conversion and processing helpers for CSCI 1430 demos.

DPG texture format conversion, geometric transforms, and brightness adjustment.
For webcam capture / frame acquisition, see demo_webcam.py.
For pinhole camera math / 3D rendering, see demo_3d.py.
"""

import cv2
import numpy as np


def convert_cv_to_dpg(image, clip=False):
    """Convert OpenCV image to Dear PyGui texture format.

    Args:
        image: OpenCV image (BGR or grayscale), uint8 or float
        clip: If True, clip float values to [0, 1] range before conversion

    Returns:
        Flattened float32 RGBA array suitable for dpg.set_value()
    """
    if clip:
        image = np.clip(image, 0, 1)

    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    image = image.astype(np.float32) / 255.0
    return image.flatten()


def convert_cv_to_dpg_float(image):
    """Convert float [0,1] image to Dear PyGui texture format.

    Unlike convert_cv_to_dpg which expects uint8, this handles float images
    directly without the /255 normalization.

    Args:
        image: Float image with values in [0, 1] range (grayscale or BGR)

    Returns:
        Flattened float32 RGBA array suitable for dpg.set_value()
    """
    image = np.clip(image, 0, 1)

    if len(image.shape) == 2:  # Grayscale
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        rgba[:, :, 0] = image
        rgba[:, :, 1] = image
        rgba[:, :, 2] = image
        rgba[:, :, 3] = 1.0
        return rgba.flatten()
    else:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGBA)
        return (image.astype(np.float32) / 255.0).flatten()


def crop_to_square(image, target_size=None):
    """Crop image to square from center, optionally resize.

    Args:
        image: Input image (any shape)
        target_size: Optional int to resize square to

    Returns:
        Square image
    """
    h, w = image.shape[:2]

    if w > h:
        crop = (w - h) // 2
        image = image[:, crop:crop + h]
    elif h > w:
        crop = (h - w) // 2
        image = image[crop:crop + w, :]

    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return image


def apply_affine_transform(image, rotation=0, scale=1.0, translate_x=0, translate_y=0,
                           border_mode=cv2.BORDER_REFLECT):
    """Apply rotation, scale, and translation to image.

    Args:
        image: Input image
        rotation: Rotation angle in degrees
        scale: Scale factor
        translate_x: X translation as percentage of width (-100 to 100)
        translate_y: Y translation as percentage of height (-100 to 100)
        border_mode: OpenCV border mode for out-of-bounds pixels

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    M[0, 2] += translate_x * w / 100.0
    M[1, 2] += translate_y * h / 100.0
    return cv2.warpAffine(image, M, (w, h), borderMode=border_mode)


def apply_brightness(image, scale=1.0, shift=0.0):
    """Apply brightness adjustment to image.

    Args:
        image: Input image (uint8)
        scale: Multiplicative factor
        shift: Additive offset

    Returns:
        Adjusted image (uint8)
    """
    result = image.astype(np.float32) * scale + shift
    return np.clip(result, 0, 255).astype(np.uint8)
