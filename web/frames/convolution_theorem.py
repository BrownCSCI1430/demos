"""
Web frame module for liveConvolutionTheorem demo.
Demonstrates convolution theorem: spatial convolution = frequency multiplication.
"""

import numpy as np
import cv2

from utils.demo_fft import process_convolution, process_deconvolution, visualize_fft_amplitude
from utils.demo_kernels import create_kernel, pad_kernel_to_image_size

# Module-level state
_random_kernel = None
_random_kernel_size = None

KERNELS = ["Box", "Gaussian", "Sharpen", "Edge Horizontal", "Edge Vertical", "Random"]

# Map display names to create_kernel names
_KERNEL_MAP = {
    "Box": "Box Blur",
    "Gaussian": "Gaussian",
    "Sharpen": "Sharpen",
    "Edge Horizontal": "Edge (Sobel X)",
    "Edge Vertical": "Edge (Sobel Y)",
    "Random": "Random",
}


def _create_random_kernel(size):
    """Create or return cached random kernel."""
    global _random_kernel, _random_kernel_size
    if _random_kernel is None or _random_kernel_size != size:
        kernel = np.random.randn(size, size).astype(np.float64)
        s = kernel.sum()
        _random_kernel = kernel / s if s != 0 else kernel / (size * size)
        _random_kernel_size = size
    return _random_kernel


def _visualize_kernel(kernel, target_size):
    """Render kernel as a grayscale image for display."""
    kh, kw = kernel.shape
    vis = kernel.copy()
    vmin, vmax = vis.min(), vis.max()
    if vmax > vmin:
        vis = (vis - vmin) / (vmax - vmin)
    else:
        vis = np.full_like(vis, 0.5)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.resize(vis, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


WEB_CONFIG = {
    "title": "Convolution Theorem",
    "description": (
        "Demonstrates the convolution theorem: spatial convolution equals "
        "frequency multiplication. Includes deconvolution mode with regularization."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "input",       "label": "Input Image",         "width": 200, "height": 200},
        {"id": "kernel_img",  "label": "Kernel",              "width": 200, "height": 200},
        {"id": "result",      "label": "Result",              "width": 200, "height": 200},
        {"id": "input_fft",   "label": "F(Image)",            "width": 200, "height": 200},
        {"id": "kernel_fft",  "label": "F(Kernel)",           "width": 200, "height": 200},
        {"id": "product_fft", "label": "F(Image) * F(Kernel)", "width": 200, "height": 200},
    ],
    "controls": {
        "mode": {
            "type": "choice", "options": ["Convolution", "Deconvolution"],
            "default": "Convolution", "label": "Mode",
        },
        "kernel_type": {
            "type": "choice", "options": KERNELS,
            "default": "Gaussian", "label": "Kernel",
        },
        "kernel_size": {
            "type": "int", "min": 3, "max": 51, "step": 2,
            "default": 15, "label": "Kernel Size", "format": "d",
        },
        "sigma": {
            "type": "float", "min": 0.1, "max": 15.0, "step": 0.1,
            "default": 3.0, "label": "Sigma",
        },
        "use_regularization": {
            "type": "bool", "default": False, "label": "Regularization",
            "visible_when": {"mode": ["Deconvolution"]},
        },
        "reg_lambda": {
            "type": "float", "min": 0.0001, "max": 0.5, "step": 0.001,
            "default": 0.01, "label": "Reg. Lambda", "format": ".4f",
            "visible_when": {"mode": ["Deconvolution"]},
        },
        "regenerate": {
            "type": "button", "label": "Regenerate Random Kernel",
            "visible_when": {"kernel_type": ["Random"]},
        },
        "pause": {
            "type": "bool", "default": False, "label": "Pause",
        },
    },
    "layout": {
        "rows": [
            ["input", "kernel_img", "result"],
            ["input_fft", "kernel_fft", "product_fft"],
        ],
    },
}


def _float_to_bgr(im_float):
    gray = np.clip(im_float * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def web_button(button_id):
    """Handle button clicks."""
    global _random_kernel
    if button_id == "regenerate":
        _random_kernel = None  # Force regeneration


def web_frame(state):
    img = state["input_image"]

    # Crop to square and convert to float grayscale
    h, w = img.shape[:2]
    sz = min(h, w)
    y0, x0 = (h - sz) // 2, (w - sz) // 2
    crop = img[y0:y0+sz, x0:x0+sz]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    proc_sz = 200
    gray = cv2.resize(gray, (proc_sz, proc_sz))
    im = gray.astype(np.float64) / 255.0

    # Build kernel
    ktype = state["kernel_type"]
    ksize = state["kernel_size"] | 1  # Ensure odd

    if ktype == "Random":
        kernel = _create_random_kernel(ksize)
    else:
        mapped_name = _KERNEL_MAP.get(ktype, ktype)
        kernel = create_kernel(mapped_name, ksize, sigma=state["sigma"])

    # Pad kernel to image size
    kernel_padded = pad_kernel_to_image_size(kernel, im.shape)

    mode = state["mode"]
    is_deconv = mode == "Deconvolution"

    if is_deconv:
        # Convolve first, then deconvolve
        blurred, _, _, _ = process_convolution(im, kernel_padded)
        reg_val = state["reg_lambda"] if state["use_regularization"] else 0.01
        result, blurred_fft, kernel_fft, result_fft = process_deconvolution(
            blurred, kernel_padded, regularization=reg_val)
        input_display = blurred
        image_fft = blurred_fft
        product_fft = result_fft
    else:
        result, image_fft, kernel_fft, product_fft = process_convolution(im, kernel_padded)
        input_display = im

    # Visualizations
    input_vis = _float_to_bgr(np.clip(input_display, 0, 1))
    kernel_vis = _visualize_kernel(kernel, proc_sz)
    result_vis = _float_to_bgr(np.clip(result, 0, 1))

    img_fft_vis = _float_to_bgr(visualize_fft_amplitude(image_fft, im.shape, use_log=True))
    ker_fft_vis = _float_to_bgr(visualize_fft_amplitude(np.fft.fft2(kernel_padded), im.shape, use_log=False))
    prod_fft_vis = _float_to_bgr(visualize_fft_amplitude(product_fft, im.shape, use_log=True))

    op = "÷" if is_deconv else "×"
    status = f"Mode: {mode}  |  Kernel: {ktype} {ksize}×{ksize}  |  F(Image) {op} F(Kernel)"

    return {
        "input": input_vis,
        "kernel_img": kernel_vis,
        "result": result_vis,
        "input_fft": img_fft_vis,
        "kernel_fft": ker_fft_vis,
        "product_fft": prod_fft_vis,
        "status": status,
    }
