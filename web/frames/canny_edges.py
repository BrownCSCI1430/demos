"""
Web frame module for liveCannyEdges demo.
Real-time Canny edge detection on webcam input.
"""

import cv2
import numpy as np


WEB_CONFIG = {
    "title": "Canny Edge Detection",
    "description": (
        "Real-time Canny edge detection with adjustable blur sigma "
        "and threshold parameters. Uses your webcam as input."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "input", "label": "Input (grayscale)", "width": 320, "height": 240},
        {"id": "edges", "label": "Canny Edges",       "width": 320, "height": 240},
    ],
    "controls": {
        "blur_sigma": {
            "type": "float", "min": 0.1, "max": 10.0, "step": 0.1,
            "default": 1.0, "label": "Blur \u03c3",
        },
        "canny_low": {
            "type": "int", "min": 1, "max": 255, "step": 1,
            "default": 10, "label": "Threshold Low", "format": "d",
        },
        "canny_high": {
            "type": "int", "min": 1, "max": 255, "step": 1,
            "default": 70, "label": "Threshold High", "format": "d",
        },
    },
    "layout": {"rows": [["input", "edges"]]},
}


def web_frame(state):
    """Process one frame: blur + Canny edge detection."""
    img = state["input_image"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), state["blur_sigma"])

    low = state["canny_low"]
    high = max(low, state["canny_high"])
    edges = cv2.Canny(blur, low, high)

    # Convert to BGR for canvas rendering
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    status = (
        f"\u03c3 = {state['blur_sigma']:.1f}  |  "
        f"Thresholds: {low} / {high}"
    )

    return {"input": gray_bgr, "edges": edges_bgr, "status": status}
