"""
Web frame module for liveStarter demo.
Minimal template — passes camera input through process_frame().
"""

from liveStarter import process_frame

WEB_CONFIG = {
    "title": "Starter Template",
    "description": (
        "Minimal template for image processing experiments. "
        "Edit process_frame() in liveStarter.py to see your changes."
    ),
    "camera": {"width": 320, "height": 240},
    "outputs": [
        {"id": "input",  "label": "Input (Raw Webcam)", "width": 320, "height": 240},
        {"id": "output", "label": "Output (Your Processing)", "width": 320, "height": 240},
    ],
    "controls": {},
    "layout": {"rows": [["input", "output"]]},
}


def web_frame(state):
    img = state["input_image"]
    result = process_frame(img)
    return {"input": img, "output": result}
