"""
Live Viola-Jones Face Detection Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveViolaJones.py
This version: Dear PyGui sliders for interactive face detection parameters
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg, init_camera, load_fallback_image, get_frame
from utils.demo_ui import (
    load_fonts, setup_viewport, make_state_updater, make_reset_callback,
    add_global_controls, control_panel, create_parameter_table, add_parameter_row,
    poll_collapsible_panels,
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


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 50
    available_height = vp_height - 350

    aspect_ratio = state.frame_width / state.frame_height

    img_width = available_width
    img_height = int(img_width / aspect_ratio)

    if img_height > available_height:
        img_height = available_height
        img_width = int(img_height * aspect_ratio)

    if dpg.does_item_exist("detection_image"):
        dpg.configure_item("detection_image", width=img_width, height=img_height)


def on_viewport_resize():
    update_image_sizes()


def update_cascade_type(sender, value):
    state.cascade_type = value
    load_cascade(value)


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Viola-Jones Face Detection Demo')
    parser.add_argument('--width', type=int, default=None, help='Camera width')
    parser.add_argument('--height', type=int, default=None, help='Camera height')
    args = parser.parse_args()

    # Initialize camera with optional resolution
    state.cap, state.frame_width, state.frame_height, state.use_camera = \
        init_camera(width=args.width, height=args.height)

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.frame_height, state.frame_width = state.fallback_image.shape[:2]
        state.cat_mode = True

    # Initialize cascade
    load_cascade(state.cascade_type)

    frame_width, frame_height = state.frame_width, state.frame_height

    dpg.create_context()

    load_fonts()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="detection_texture")

    with dpg.window(label="Viola-Jones Face Detection Demo", tag="main_window"):
        def _extra_reset():
            load_cascade(DEFAULTS["cascade_type"])

        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
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
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

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

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
