"""
Live Viola-Jones Face Detection Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveViolaJones.py
This version: Dear PyGui sliders for interactive face detection parameters
"""

import cv2
import os
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg, init_camera, load_fallback_image, get_frame
from utils.demo_ui import setup_viewport, make_state_updater, make_reset_callback

# Default values
DEFAULTS = {
    "scale_factor": 1.1,
    "min_neighbors": 3,
    "min_size": 30,
    "show_boxes": True,
    "cascade_type": "frontalface_default",
    "ui_scale": 1.5,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        state.frame_width = 640
        state.frame_height = 480

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.cat_mode = True

    # Initialize cascade
    load_cascade(state.cascade_type)

    frame_width, frame_height = state.frame_width, state.frame_height

    dpg.create_context()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="detection_texture")

    with dpg.window(label="Viola-Jones Face Detection Demo", tag="main_window"):
        # Global controls row (custom for this demo due to cascade combo)
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="Cascade",
                items=list(CASCADES.keys()),
                default_value=state.cascade_type,
                callback=update_cascade_type,
                tag="cascade_combo",
                width=150
            )
            dpg.add_slider_float(
                label="UI Scale",
                default_value=DEFAULTS["ui_scale"],
                min_value=1.0, max_value=3.0,
                callback=lambda s, v: dpg.set_global_font_scale(v),
                width=100
            )
            dpg.add_spacer(width=20)
            dpg.add_checkbox(
                label="Show Boxes",
                default_value=state.show_boxes,
                callback=make_state_updater(state, "show_boxes")
            )
            dpg.add_checkbox(
                label="Cat Mode",
                default_value=state.cat_mode,
                callback=make_state_updater(state, "cat_mode"),
                tag="cat_mode_checkbox",
                enabled=state.use_camera
            )
            if not state.use_camera:
                dpg.add_text("(no webcam)", color=(255, 100, 100))

        dpg.add_separator()

        # Detection parameters
        with dpg.collapsing_header(label="Detection Parameters", default_open=True):
            with dpg.table(header_row=False,
                           borders_innerV=False, borders_outerV=False,
                           borders_innerH=False, borders_outerH=False,
                           policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=30)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=20)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                with dpg.table_row():
                    dpg.add_text("Scale Factor")
                    dpg.add_slider_float(tag="scale_slider", default_value=state.scale_factor,
                                         min_value=1.01, max_value=2.0,
                                         callback=make_state_updater(state, "scale_factor"),
                                         width=80, format="%.2f")
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "scale_factor", "scale_slider", DEFAULTS["scale_factor"]),
                                   width=25)
                    dpg.add_spacer(width=20)
                    dpg.add_text("Min Neighbors")
                    dpg.add_slider_int(tag="neighbors_slider", default_value=state.min_neighbors,
                                       min_value=1, max_value=10,
                                       callback=make_state_updater(state, "min_neighbors"),
                                       width=80)
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "min_neighbors", "neighbors_slider", DEFAULTS["min_neighbors"]),
                                   width=25)

                with dpg.table_row():
                    dpg.add_text("Min Size")
                    dpg.add_slider_int(tag="minsize_slider", default_value=state.min_size,
                                       min_value=10, max_value=200,
                                       callback=make_state_updater(state, "min_size"),
                                       width=80)
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "min_size", "minsize_slider", DEFAULTS["min_size"]),
                                   width=25)
                    dpg.add_spacer(width=20)
                    dpg.add_spacer(width=80)
                    dpg.add_spacer(width=100)
                    dpg.add_spacer(width=30)

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
        frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode,
                          (state.frame_width, state.frame_height))
        if frame is None:
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
