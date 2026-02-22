"""
Web frame module for liveHOGPersonDetector demo.
HOG-based person detection using OpenCV's built-in descriptor.
"""

import cv2
import numpy as np

# Initialize HOG detector once at module level
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

WEB_CONFIG = {
    "title": "HOG Person Detection",
    "description": (
        "Detect people using Histogram of Oriented Gradients (HOG) features "
        "with OpenCV's built-in SVM classifier."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "detection", "label": "Detection", "width": 320, "height": 240},
    ],
    "controls": {
        "win_stride": {
            "type": "int", "min": 4, "max": 16, "step": 4,
            "default": 8, "label": "Win Stride", "format": "d",
        },
        "scale": {
            "type": "float", "min": 1.01, "max": 1.5, "step": 0.01,
            "default": 1.05, "label": "Scale",
        },
        "threshold": {
            "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
            "default": 0.0, "label": "Hit Threshold",
        },
        "show_boxes": {
            "type": "bool", "default": True, "label": "Show Boxes",
        },
    },
    "layout": {"rows": [["detection"]]},
}


def web_frame(state):
    img = state["input_image"]
    stride = state["win_stride"]
    scale = state["scale"]
    threshold = state["threshold"]
    show_boxes = state["show_boxes"]

    output = img.copy()
    rects, weights = _hog.detectMultiScale(
        img,
        winStride=(stride, stride),
        scale=scale,
        hitThreshold=threshold,
    )

    n_detections = 0
    if show_boxes and len(rects) > 0:
        for (x, y, w, h) in rects:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            n_detections += 1
    elif len(rects) > 0:
        n_detections = len(rects)

    status = (
        f"People Detected: {n_detections}  |  "
        f"Stride: {stride}  |  Scale: {scale:.2f}"
    )
    return {"detection": output, "status": status}
