"""
Live Viola-Jones Face Detection Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveViolaJones.py
This version: Dear PyGui sliders for interactive face detection parameters
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg
from utils.demo_webcam import init_camera_demo, cleanup_camera_demo, get_frame
from utils.demo_ui import (
    setup_viewport, make_state_updater, make_reset_callback, make_camera_callback,
    add_global_controls, control_panel, create_parameter_table, add_parameter_row,
    poll_collapsible_panels, auto_resize_images, create_blank_texture,
)

# Default values
DEFAULTS = {
    "scale_factor": 1.1,
    "min_neighbors": 3,
    "min_size": 30,
    "show_boxes": True,
    "cascade_type": "frontalface_default",
    "pause": False,
    "ui_scale": 1.5,
}

GUIDE_VIOLA_JONES = [
    {"title": "Haar-like features",
     "body": "Rectangular features measure intensity differences between adjacent "
             "regions. They capture edges, lines, and center-surround patterns "
             "characteristic of faces (e.g., eyes are darker than cheeks)."},
    {"title": "Integral image",
     "body": "A summed area table makes computing any rectangle sum O(1) \u2014 "
             "just four lookups regardless of rectangle size. This enables "
             "evaluating thousands of Haar features in real time."},
    {"title": "AdaBoost cascade",
     "body": "AdaBoost selects the most discriminative features and combines weak "
             "classifiers into strong ones. Arranged in a cascade: early stages "
             "use few features to quickly reject obvious non-faces (most windows). "
             "Later stages use more features for difficult cases."},
    {"title": "Cascade types and parameters",
     "body": "Different cascades detect different objects: frontal face, profile, "
             "eyes, smiles. Try switching cascades to see the difference.\n"
             "Scale Factor: image pyramid step (smaller = slower but finer).\n"
             "Min Neighbors: overlap filtering (higher = fewer false positives).\n"
             "Min Size: smallest detectable face in pixels."},
]


# Cascade options
CASCADES = {
    "frontalface_default": "haarcascade_frontalface_default.xml",
    "frontalface_alt": "haarcascade_frontalface_alt.xml",
    "frontalface_alt2": "haarcascade_frontalface_alt2.xml",
    "profileface": "haarcascade_profileface.xml",
    "eye": "haarcascade_eye.xml",
    "smile": "haarcascade_smile.xml",
}


class State:
    cap = None
    frame_width = 640
    frame_height = 480
    scale_factor = DEFAULTS["scale_factor"]
    min_neighbors = DEFAULTS["min_neighbors"]
    min_size = DEFAULTS["min_size"]
    show_boxes = DEFAULTS["show_boxes"]
    cascade_type = DEFAULTS["cascade_type"]
    face_cascade = None
    use_camera = True
    cat_mode = False
    pause = False
    fallback_image = None


state = State()


def load_cascade(cascade_name):
    """Load a Haar cascade classifier"""
    cascade_file = cv2.data.haarcascades + CASCADES.get(cascade_name, CASCADES["frontalface_default"])
    state.face_cascade = cv2.CascadeClassifier(cascade_file)


_IMAGE_LAYOUT = None  # set after camera init

def update_image_sizes():
    auto_resize_images(_IMAGE_LAYOUT, margin_w=50, margin_h=350)


def update_cascade_type(sender, value):
    state.cascade_type = value
    load_cascade(value)


def main():
    global _IMAGE_LAYOUT
    frame_width, frame_height = init_camera_demo(state, "Viola-Jones Face Detection Demo")
    aspect = frame_width / frame_height
    _IMAGE_LAYOUT = [("detection_image", 1.0, aspect)]

    # Initialize cascade
    load_cascade(state.cascade_type)

    with dpg.texture_registry():
        create_blank_texture(frame_width, frame_height, "detection_texture")

    with dpg.window(label="Viola-Jones Face Detection Demo", tag="main_window"):
        def _extra_reset():
            load_cascade(DEFAULTS["cascade_type"])

        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            camera_callback=make_camera_callback(state),
            reset_extra=_extra_reset,
            guide=GUIDE_VIOLA_JONES, guide_title="Viola-Jones Face Detection",
        )

        dpg.add_separator()

        # Detection parameters
        with dpg.group(horizontal=True):
            with control_panel("Detection Parameters", width=400, height=200,
                               color=(150, 200, 255)):
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Cascade", items=list(CASCADES.keys()),
                        default_value=state.cascade_type,
                        callback=update_cascade_type,
                        tag="cascade_type_combo", width=150,
                    )
                    dpg.add_checkbox(
                        label="Show Boxes", default_value=state.show_boxes,
                        callback=make_state_updater(state, "show_boxes"),
                        tag="show_boxes_checkbox",
                    )
                with create_parameter_table():
                    add_parameter_row(
                        "Scale Factor", "scale_factor_slider", DEFAULTS["scale_factor"],
                        1.01, 2.0, make_state_updater(state, "scale_factor"),
                        make_reset_callback(state, "scale_factor", "scale_factor_slider", DEFAULTS["scale_factor"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Min Neighbors", "min_neighbors_slider", DEFAULTS["min_neighbors"],
                        1, 10, make_state_updater(state, "min_neighbors"),
                        make_reset_callback(state, "min_neighbors", "min_neighbors_slider", DEFAULTS["min_neighbors"]),
                        slider_type="int")
                    add_parameter_row(
                        "Min Size", "min_size_slider", DEFAULTS["min_size"],
                        10, 200, make_state_updater(state, "min_size"),
                        make_reset_callback(state, "min_size", "min_size_slider", DEFAULTS["min_size"]),
                        slider_type="int")

        dpg.add_separator()
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        dpg.add_text("Viola-Jones Detection")
        dpg.add_image("detection_texture", tag="detection_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - Viola-Jones Detection",
                   frame_width + 100, frame_height + 400,
                   "main_window", update_image_sizes, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        if not state.pause:
            frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode,
                              (state.frame_width, state.frame_height))
            if frame is None:
                dpg.render_dearpygui_frame()
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            if state.face_cascade is not None:
                detected = state.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=state.scale_factor,
                    minNeighbors=state.min_neighbors,
                    minSize=(state.min_size, state.min_size)
                )
            else:
                detected = []

            if state.show_boxes:
                for (x, y, w, h) in detected:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)

            dpg.set_value("detection_texture", convert_cv_to_dpg(frame))

            status = f"Detections: {len(detected)}  |  Scale: {state.scale_factor:.2f}  |  Neighbors: {state.min_neighbors}"
            dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    cleanup_camera_demo(state)


if __name__ == "__main__":
    main()
