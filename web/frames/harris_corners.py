"""
Web frame module for liveHarrisCorners demo.
Harris corner detection with affine transforms and brightness adjustments.
"""

import cv2
import numpy as np


def _apply_harris(gray, block_size, ksize, k, threshold):
    """Run Harris corner detection, return image with corners marked red."""
    ksize_odd = ksize | 1  # Ensure odd
    harris = cv2.cornerHarris(gray, block_size, ksize_odd, k)
    # Threshold to find corners
    mask = harris > threshold * harris.max() if harris.max() > 0 else np.zeros_like(harris, dtype=bool)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[mask] = [0, 0, 255]  # Red corners (BGR)
    n_corners = int(np.sum(mask))
    return out, n_corners


def _apply_affine(img, rotation, scale, tx, ty):
    """Apply rotation, scale, and translation to image."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _apply_brightness(img, scale, shift):
    """Apply brightness scale and shift."""
    result = img.astype(np.float32) * scale + shift
    return np.clip(result, 0, 255).astype(np.uint8)


WEB_CONFIG = {
    "title": "Harris Corner Detection",
    "description": (
        "Detect corners using Harris corner detector with tunable parameters. "
        "Apply affine transforms and brightness changes to test corner invariance."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "original",    "label": "Original + Corners",    "width": 320, "height": 240},
        {"id": "transformed", "label": "Transformed + Corners", "width": 320, "height": 240},
    ],
    "controls": {
        "block_size": {
            "type": "int", "min": 2, "max": 10, "step": 1,
            "default": 2, "label": "Block Size", "format": "d",
        },
        "ksize": {
            "type": "int", "min": 3, "max": 31, "step": 2,
            "default": 5, "label": "Sobel K", "format": "d",
        },
        "k": {
            "type": "float", "min": 0.01, "max": 0.3, "step": 0.01,
            "default": 0.07, "label": "Harris k",
        },
        "threshold": {
            "type": "float", "min": 0.001, "max": 0.1, "step": 0.001,
            "default": 0.01, "label": "Threshold", "format": ".3f",
        },
        "rotation": {
            "type": "float", "min": -180.0, "max": 180.0, "step": 1.0,
            "default": 0.0, "label": "Rotate", "format": ".0f",
        },
        "scale": {
            "type": "float", "min": 0.25, "max": 4.0, "step": 0.05,
            "default": 1.0, "label": "Scale",
        },
        "translate_x": {
            "type": "float", "min": -50.0, "max": 50.0, "step": 1.0,
            "default": 0.0, "label": "Translate X", "format": ".0f",
        },
        "translate_y": {
            "type": "float", "min": -50.0, "max": 50.0, "step": 1.0,
            "default": 0.0, "label": "Translate Y", "format": ".0f",
        },
        "brightness_scale": {
            "type": "float", "min": 0.5, "max": 2.0, "step": 0.05,
            "default": 1.0, "label": "Brightness Scale",
        },
        "brightness_shift": {
            "type": "float", "min": -100.0, "max": 100.0, "step": 1.0,
            "default": 0.0, "label": "Brightness Shift", "format": ".0f",
        },
        "pause": {
            "type": "bool", "default": False, "label": "Pause",
        },
    },
    "layout": {"rows": [["original", "transformed"]]},
}


def web_frame(state):
    img = state["input_image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris on original
    orig_out, n_orig = _apply_harris(
        gray, state["block_size"], state["ksize"], state["k"], state["threshold"])

    # Apply transforms
    transformed = _apply_affine(
        gray, state["rotation"], state["scale"],
        state["translate_x"], state["translate_y"])
    transformed = _apply_brightness(
        transformed, state["brightness_scale"], state["brightness_shift"])

    # Harris on transformed
    trans_out, n_trans = _apply_harris(
        transformed, state["block_size"], state["ksize"], state["k"], state["threshold"])

    status = (
        f"Original: {n_orig} corners  |  Transformed: {n_trans} corners  |  "
        f"Block: {state['block_size']}  |  k: {state['k']:.3f}"
    )
    return {"original": orig_out, "transformed": trans_out, "status": status}
