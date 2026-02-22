"""
Web frame module for liveFilter demo.
Image filtering with interactive kernel editor.
"""

import cv2
import numpy as np

from utils.demo_kernels import (
    KERNEL_PRESETS, SIGMA_KERNELS, ZERO_DC_KERNELS,
    create_kernel,
)

# ── Module-level state ──
_kernel_type = "Box Blur"
_kernel_size = 3
_kernel_values = create_kernel("Box Blur", 3)
_gaussian_sigma = 1.0
_hovered_cell = None

EDITOR_SIZE = 240  # Pixel size of kernel editor canvas


def _value_to_color(value, min_val, max_val):
    """Map kernel value to grayscale intensity."""
    if min_val == max_val:
        return 128
    if min_val >= 0:
        intensity = int((value / max_val) * 255) if max_val > 0 else 128
    else:
        abs_max = max(abs(min_val), abs(max_val))
        normalized = (value + abs_max) / (2 * abs_max)
        intensity = int(normalized * 255)
    return max(0, min(255, intensity))


def _draw_kernel_editor():
    """Render kernel grid as a BGR image."""
    global _kernel_values
    if _kernel_values is None:
        return np.zeros((EDITOR_SIZE, EDITOR_SIZE, 3), dtype=np.uint8)

    size = _kernel_values.shape[0]
    cell_size = EDITOR_SIZE / size
    img = np.full((EDITOR_SIZE, EDITOR_SIZE, 3), 40, dtype=np.uint8)

    vmin, vmax = _kernel_values.min(), _kernel_values.max()

    for row in range(size):
        for col in range(size):
            x1 = int(col * cell_size) + 1
            y1 = int(row * cell_size) + 1
            x2 = int((col + 1) * cell_size) - 1
            y2 = int((row + 1) * cell_size) - 1

            val = _kernel_values[row, col]
            gray = _value_to_color(val, vmin, vmax)
            color = (gray, gray, gray)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # Highlight hovered cell
            if _hovered_cell == (row, col):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw value text if cells are large enough
            if cell_size > 20:
                text = f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}"
                text_color = (0, 0, 0) if gray > 128 else (255, 255, 255)
                font_scale = min(cell_size / 80, 0.5)
                font_scale = max(font_scale, 0.25)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, 1)
                cx = (x1 + x2) // 2 - tw // 2
                cy = (y1 + y2) // 2 + th // 2
                cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, text_color, 1, cv2.LINE_AA)

    return img


def _get_cell_from_xy(x, y):
    """Map canvas pixel coords to (row, col) or None."""
    if _kernel_values is None:
        return None
    size = _kernel_values.shape[0]
    cell_size = EDITOR_SIZE / size
    col = int(x / cell_size)
    row = int(y / cell_size)
    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None


def _update_kernel_from_preset():
    """Rebuild kernel from current preset/size/sigma."""
    global _kernel_values
    _kernel_values = create_kernel(_kernel_type, _kernel_size, sigma=_gaussian_sigma)


# Custom presets list (adds "Custom" option)
_ALL_PRESETS = KERNEL_PRESETS + ["Custom"]


WEB_CONFIG = {
    "title": "Image Filtering",
    "description": (
        "Apply convolution filters with adjustable kernel size. "
        "Click on kernel cells to edit values manually."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "input",         "label": "Input (Grayscale)", "width": 320, "height": 240},
        {"id": "filtered",      "label": "Filtered",          "width": 320, "height": 240},
        {"id": "kernel_editor", "label": "Kernel Editor",     "width": EDITOR_SIZE, "height": EDITOR_SIZE},
    ],
    "controls": {
        "kernel_type": {
            "type": "choice", "options": _ALL_PRESETS,
            "default": "Box Blur", "label": "Kernel",
        },
        "kernel_size": {
            "type": "int", "min": 3, "max": 15, "step": 2,
            "default": 3, "label": "Kernel Size", "format": "d",
        },
        "sigma": {
            "type": "float", "min": 0.1, "max": 5.0, "step": 0.1,
            "default": 1.0, "label": "Sigma",
            "visible_when": {"kernel_type": list(SIGMA_KERNELS)},
        },
        "normalize": {
            "type": "bool", "default": True, "label": "Normalize",
        },
    },
    "mouse": ["kernel_editor"],
    "layout": {"rows": [["input", "filtered"], ["kernel_editor"]]},
}


def web_mouse(event):
    """Handle mouse events on kernel editor canvas."""
    global _hovered_cell, _kernel_values, _kernel_type

    if event["canvas"] != "kernel_editor":
        return

    cell = _get_cell_from_xy(event["x"], event["y"])

    if event["type"] == "mousemove":
        _hovered_cell = cell
        return

    if event["type"] in ("click", "contextmenu") and cell is not None:
        row, col = cell
        if event["ctrl"]:
            # Ctrl+click: set to 0
            _kernel_values[row, col] = 0.0
        else:
            delta = 0.1
            if event["shift"]:
                delta = 1.0
            if event["type"] == "contextmenu" or event["button"] == 2:
                delta = -delta
            _kernel_values[row, col] += delta
        _kernel_type = "Custom"


def web_frame(state):
    global _kernel_type, _kernel_size, _gaussian_sigma, _kernel_values

    img = state["input_image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check if preset or size changed
    new_type = state["kernel_type"]
    new_size = state["kernel_size"] | 1  # Ensure odd
    new_sigma = state.get("sigma", 1.0)

    if (new_type != _kernel_type and new_type != "Custom") or new_size != _kernel_size:
        _kernel_type = new_type
        _kernel_size = new_size
        _gaussian_sigma = new_sigma
        _update_kernel_from_preset()
    elif new_type in SIGMA_KERNELS and new_sigma != _gaussian_sigma:
        _gaussian_sigma = new_sigma
        _update_kernel_from_preset()

    # Apply filter
    kernel = _kernel_values.copy()
    normalize = state["normalize"]
    if normalize and _kernel_type not in ZERO_DC_KERNELS and kernel.sum() != 0:
        kernel = kernel / kernel.sum()
    kernel = kernel.astype(np.float32)
    filtered = cv2.filter2D(gray, -1, kernel)

    # Render kernel editor
    editor_img = _draw_kernel_editor()

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    n = _kernel_values.shape[0]
    status = f"Kernel: {_kernel_type} {n}×{n}  |  Sum: {_kernel_values.sum():.3f}"

    return {
        "input": gray_bgr,
        "filtered": filtered_bgr,
        "kernel_editor": editor_img,
        "status": status,
    }
