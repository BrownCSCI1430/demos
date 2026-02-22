"""
Web frame module for liveSIFTMatching demo.
SIFT feature matching between query images and live camera feed.
"""

import os
import cv2
import numpy as np

from liveSIFTMatching import compute_sift_matches

# ── Module-level setup ──
_detector = cv2.SIFT_create()

# Discover query images from data/
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
_QUERY_FILES = []
_QUERY_NAMES = []
_data_dir = "data"

if os.path.isdir(_data_dir):
    for f in sorted(os.listdir(_data_dir)):
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
            _QUERY_FILES.append(os.path.join(_data_dir, f))
            _QUERY_NAMES.append(f)

if not _QUERY_NAMES:
    _QUERY_NAMES = ["(none)"]
    _QUERY_FILES = [""]

# Load first query image
_current_query_name = _QUERY_NAMES[0]
_query_image = None
if _QUERY_FILES[0] and os.path.exists(_QUERY_FILES[0]):
    _query_image = cv2.imread(_QUERY_FILES[0], cv2.IMREAD_GRAYSCALE)
    if _query_image is not None:
        _query_image = cv2.resize(_query_image, (240, 240))


WEB_CONFIG = {
    "title": "SIFT Feature Matching",
    "description": (
        "Match SIFT features between a query image and live camera feed. "
        "Green lines show good matches passing the ratio test."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "matches", "label": "SIFT Matches (Query | Live)", "width": 560, "height": 240},
    ],
    "controls": {
        "query_image": {
            "type": "choice", "options": _QUERY_NAMES,
            "default": _QUERY_NAMES[0], "label": "Query Image",
        },
        "match_distance": {
            "type": "float", "min": 0.1, "max": 1.0, "step": 0.01,
            "default": 0.55, "label": "Distance Ratio",
        },
        "show_matches": {
            "type": "bool", "default": True, "label": "Show Matches",
        },
    },
    "layout": {"rows": [["matches"]]},
}


def web_frame(state):
    global _current_query_name, _query_image

    img = state["input_image"]

    # Reload query image if changed
    query_name = state["query_image"]
    if query_name != _current_query_name:
        _current_query_name = query_name
        if query_name in _QUERY_NAMES:
            idx = _QUERY_NAMES.index(query_name)
            path = _QUERY_FILES[idx]
            if path and os.path.exists(path):
                _query_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if _query_image is not None:
                    # Resize to match camera height
                    cam_h = img.shape[0]
                    qh, qw = _query_image.shape[:2]
                    if qh > 0 and qw > 0:
                        aspect = qw / qh
                        new_w = int(cam_h * aspect)
                        _query_image = cv2.resize(_query_image, (new_w, cam_h))

    output, good_matches = compute_sift_matches(
        img, _query_image, _detector,
        state["match_distance"], state["show_matches"])

    # Ensure output is BGR
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    status = f"Matches: {len(good_matches)}  |  Distance Ratio: {state['match_distance']:.2f}"
    return {"matches": output, "status": status}
