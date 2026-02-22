"""
Web frame module for liveFourier demo.
Fourier Transform visualization with multiple modes.
"""

import numpy as np
import cv2

from liveFourier import process_fft, transform_image

# ── Module-level state for animation ──
_frame_counter = 0
_magnitude = 1
_orientation = 0.0
_magnitude_frame_counter = 0
_amplitude_cutoff = 1
_cutoff_direction = 1
_cutoff_frame_counter = 0

MODE_NAMES = ["Normal FFT", "DC Only", "Rotating Dot", "Frequency Reconstruction"]

WEB_CONFIG = {
    "title": "Fourier Transform",
    "description": (
        "Visualize 2D Fourier transforms with multiple modes: "
        "normal FFT, DC only, rotating dot, and frequency reconstruction."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "input",     "label": "Input",             "width": 240, "height": 240},
        {"id": "inverse",   "label": "Inverse FT",        "width": 240, "height": 240},
        {"id": "amplitude", "label": "Fourier Amplitude",  "width": 240, "height": 240},
        {"id": "phase",     "label": "Fourier Phase",      "width": 240, "height": 240},
    ],
    "controls": {
        "mode": {
            "type": "choice", "options": MODE_NAMES,
            "default": "Normal FFT", "label": "Mode",
        },
        "intensity_shift": {
            "type": "float", "min": -1.0, "max": 1.0, "step": 0.01,
            "default": 0.0, "label": "Intensity Shift",
        },
        "intensity_scale": {
            "type": "float", "min": 0.0, "max": 3.0, "step": 0.01,
            "default": 1.0, "label": "Intensity Scale",
        },
        "rotation": {
            "type": "float", "min": -180.0, "max": 180.0, "step": 1.0,
            "default": 0.0, "label": "Rotation", "format": ".0f",
        },
        "scale": {
            "type": "float", "min": 0.25, "max": 4.0, "step": 0.05,
            "default": 1.0, "label": "Scale",
        },
        "translate_x": {
            "type": "float", "min": -100.0, "max": 100.0, "step": 1.0,
            "default": 0.0, "label": "Translate X", "format": ".0f",
        },
        "translate_y": {
            "type": "float", "min": -100.0, "max": 100.0, "step": 1.0,
            "default": 0.0, "label": "Translate Y", "format": ".0f",
        },
        "dc_shift": {
            "type": "float", "min": -1.0, "max": 1.0, "step": 0.01,
            "default": 0.0, "label": "DC Shift",
        },
        "amplitude_scalar": {
            "type": "float", "min": 0.1, "max": 5.0, "step": 0.1,
            "default": 1.0, "label": "Amplitude Scale",
        },
        "phase_offset": {
            "type": "float", "min": -3.14, "max": 3.14, "step": 0.01,
            "default": 0.0, "label": "Phase Offset",
        },
        "dc_zero": {
            "type": "bool", "default": False, "label": "Zero DC",
        },
        "animate_magnitude": {
            "type": "bool", "default": True, "label": "Animate Magnitude",
            "visible_when": {"mode": ["Rotating Dot"]},
        },
        "animate_orientation": {
            "type": "bool", "default": True, "label": "Animate Orientation",
            "visible_when": {"mode": ["Rotating Dot"]},
        },
        "pause": {
            "type": "bool", "default": False, "label": "Pause",
        },
    },
    "layout": {"rows": [["input", "inverse"], ["amplitude", "phase"]]},
}


def _float_to_bgr(im_float):
    """Convert [0,1] float image to uint8 BGR (grayscale 3-channel)."""
    gray = np.clip(im_float * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def web_frame(state):
    global _frame_counter, _magnitude, _orientation
    global _magnitude_frame_counter
    global _amplitude_cutoff, _cutoff_direction, _cutoff_frame_counter

    from liveFourier import state as fft_state

    img = state["input_image"]
    mode_name = state["mode"]
    mode_idx = MODE_NAMES.index(mode_name) if mode_name in MODE_NAMES else 0

    # Crop to square and convert to float grayscale
    h, w = img.shape[:2]
    sz = min(h, w)
    y0, x0 = (h - sz) // 2, (w - sz) // 2
    crop = img[y0:y0+sz, x0:x0+sz]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Resize to processing size
    proc_sz = 240
    gray = cv2.resize(gray, (proc_sz, proc_sz))
    im = gray.astype(np.float32) / 255.0

    # Apply intensity adjustments
    im = im * state["intensity_scale"] + state["intensity_shift"]
    im = np.clip(im, 0, 1)

    # Apply spatial transforms
    im = transform_image(im, state["rotation"],
                          state["translate_x"], state["translate_y"],
                          state["scale"])

    # Sync module-level state for process_fft
    fft_state.mode = mode_idx
    fft_state.dc_shift = state["dc_shift"]
    fft_state.dc_zero = state["dc_zero"]
    fft_state.amplitude_scalar = state["amplitude_scalar"]
    fft_state.phase_offset = state["phase_offset"]

    # Animation for Rotating Dot (mode 2)
    if mode_idx == 2:
        if state.get("animate_orientation", True):
            _orientation += np.pi / 600
            if _orientation > 2 * np.pi:
                _orientation -= 2 * np.pi
        if state.get("animate_magnitude", True):
            _magnitude_frame_counter += 1
            if _magnitude_frame_counter % 10 == 0:
                _magnitude += 1
                if _magnitude >= 50:
                    _magnitude = 1
        fft_state.magnitude = _magnitude
        fft_state.orientation = _orientation

    # Animation for Frequency Reconstruction (mode 3)
    if mode_idx == 3:
        _cutoff_frame_counter += 1
        if _cutoff_frame_counter % 10 == 0:
            _amplitude_cutoff += _cutoff_direction
            if _amplitude_cutoff <= 1:
                _cutoff_direction = 1
            elif _amplitude_cutoff > proc_sz // 3:
                _cutoff_direction = -1
        fft_state.amplitude_cutoff = _amplitude_cutoff
        fft_state.cutoff_direction = _cutoff_direction

    # Run FFT processing
    new_image, amplitude_vis, phase_vis = process_fft(im)

    # Input is blanked for modes 1, 2
    if mode_idx in (1, 2):
        input_display = np.zeros_like(im)
    else:
        input_display = im

    status = f"Mode: {mode_name}"
    if mode_idx == 2:
        status += f"  |  Mag: {_magnitude}  |  Orient: {_orientation:.2f}"
    elif mode_idx == 3:
        status += f"  |  Cutoff: {_amplitude_cutoff}"

    return {
        "input": _float_to_bgr(input_display),
        "inverse": _float_to_bgr(np.clip(new_image, 0, 1)),
        "amplitude": _float_to_bgr(amplitude_vis),
        "phase": _float_to_bgr(phase_vis),
        "status": status,
    }
