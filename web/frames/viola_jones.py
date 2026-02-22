"""
Web frame module for liveViolaJones demo.
Face detection using Haar cascade classifiers.
"""

import cv2
import numpy as np

# Available cascades — these ship with opencv-python
CASCADES = {
    "frontalface_default": "haarcascade_frontalface_default.xml",
    "frontalface_alt": "haarcascade_frontalface_alt.xml",
    "frontalface_alt2": "haarcascade_frontalface_alt2.xml",
    "profileface": "haarcascade_profileface.xml",
    "eye": "haarcascade_eye.xml",
    "smile": "haarcascade_smile.xml",
}

# Module-level state
_current_cascade_name = "frontalface_default"
_cascade = None


def _load_cascade(name):
    global _cascade, _current_cascade_name
    xml_file = CASCADES.get(name)
    if xml_file is None:
        return
    # Try cv2.data.haarcascades path (standard opencv-python location)
    import os
    path = os.path.join(cv2.data.haarcascades, xml_file)
    if not os.path.exists(path):
        # Fallback: try data/ directory (bundled copies)
        path = os.path.join("data", xml_file)
    cc = cv2.CascadeClassifier(path)
    if not cc.empty():
        _cascade = cc
        _current_cascade_name = name
    else:
        print(f"Warning: Failed to load cascade {name} from {path}")


# Load default cascade at import time
_load_cascade("frontalface_default")


WEB_CONFIG = {
    "title": "Viola-Jones Face Detection",
    "description": (
        "Detect faces using Haar cascade classifiers with multiple cascade options."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "detection", "label": "Detection", "width": 320, "height": 240},
    ],
    "controls": {
        "cascade_type": {
            "type": "choice",
            "options": list(CASCADES.keys()),
            "default": "frontalface_default",
            "label": "Cascade",
        },
        "scale_factor": {
            "type": "float", "min": 1.01, "max": 2.0, "step": 0.01,
            "default": 1.1, "label": "Scale Factor",
        },
        "min_neighbors": {
            "type": "int", "min": 1, "max": 10, "step": 1,
            "default": 3, "label": "Min Neighbors", "format": "d",
        },
        "min_size": {
            "type": "int", "min": 10, "max": 200, "step": 5,
            "default": 30, "label": "Min Size", "format": "d",
        },
        "show_boxes": {
            "type": "bool", "default": True, "label": "Show Boxes",
        },
    },
    "layout": {"rows": [["detection"]]},
}


def web_frame(state):
    global _current_cascade_name

    # Reload cascade if changed
    if state["cascade_type"] != _current_cascade_name:
        _load_cascade(state["cascade_type"])

    img = state["input_image"]
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    n_detections = 0
    if _cascade is not None:
        min_sz = state["min_size"]
        rects = _cascade.detectMultiScale(
            gray,
            scaleFactor=state["scale_factor"],
            minNeighbors=state["min_neighbors"],
            minSize=(min_sz, min_sz),
        )

        if state["show_boxes"] and len(rects) > 0:
            for (x, y, w, h) in rects:
                cv2.rectangle(output, (x, y), (x + w, y + h), (127, 0, 255), 2)
        n_detections = len(rects) if len(rects) > 0 else 0

    status = (
        f"Detections: {n_detections}  |  "
        f"Scale: {state['scale_factor']:.2f}  |  "
        f"Neighbors: {state['min_neighbors']}"
    )
    return {"detection": output, "status": status}
