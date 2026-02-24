"""
Live HOG Person Detector Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveHOGPersonDetector.py
This version: Dear PyGui sliders for interactive HOG control
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
    "win_stride": 8,
    "scale": 1.05,
    "hit_threshold": 0.0,
    "show_boxes": True,
    "pause": False,
    "ui_scale": 1.5,
}

GUIDE_HOG = [
    {"title": "HOG features",
     "body": "Histogram of Oriented Gradients describes local shape via gradient "
             "orientation histograms in cells (8\u00d78 pixels), normalized across "
             "overlapping blocks (2\u00d72 cells). This captures edge structure while "
             "being robust to illumination changes."},
    {"title": "Sliding window detection",
     "body": "A fixed-size window (64\u00d7128 pixels) slides across the image at "
             "multiple scales. At each position, the HOG descriptor is extracted "
             "and classified by a pre-trained SVM."},
    {"title": "SVM classifier",
     "body": "The detector uses a linear SVM trained on labeled HOG features "
             "(person vs non-person). The decision boundary separates the two "
             "classes in the 3780-dimensional HOG feature space."},
    {"title": "Parameters",
     "body": "Hit Threshold: SVM decision boundary offset. Higher = fewer false "
             "positives but may miss some detections.\n"
             "Win Stride: step size for the sliding window (smaller = denser search).\n"
             "Scale: image pyramid step size for multi-scale detection."},
]


class State:
    cap = None
    frame_width = 640
    frame_height = 480
    win_stride = DEFAULTS["win_stride"]
    scale = DEFAULTS["scale"]
    hit_threshold = DEFAULTS["hit_threshold"]
    show_boxes = DEFAULTS["show_boxes"]
    use_camera = True
    cat_mode = False
    pause = False
    fallback_image = None


state = State()


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 50
    available_height = vp_height - 300

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


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='HOG Person Detection Demo')
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

    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    frame_width, frame_height = state.frame_width, state.frame_height

    dpg.create_context()

    load_fonts()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="detection_texture")

    with dpg.window(label="HOG Person Detection Demo", tag="main_window"):
        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            guide=GUIDE_HOG, guide_title="HOG Person Detection",
        )

        dpg.add_separator()

        # Detection parameters
        with dpg.group(horizontal=True):
            with control_panel("Detection Parameters", width=350, height=170,
                               color=(150, 200, 255)):
                dpg.add_checkbox(
                    label="Show Boxes", default_value=state.show_boxes,
                    callback=make_state_updater(state, "show_boxes"),
                    tag="show_boxes_checkbox",
                )
                with create_parameter_table():
                    add_parameter_row(
                        "Win Stride", "win_stride_slider", DEFAULTS["win_stride"],
                        4, 16, make_state_updater(state, "win_stride"),
                        make_reset_callback(state, "win_stride", "win_stride_slider", DEFAULTS["win_stride"]),
                        slider_type="int")
                    add_parameter_row(
                        "Scale", "scale_slider", DEFAULTS["scale"],
                        1.01, 1.5, make_state_updater(state, "scale"),
                        make_reset_callback(state, "scale", "scale_slider", DEFAULTS["scale"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Hit Threshold", "hit_threshold_slider", DEFAULTS["hit_threshold"],
                        0.0, 1.0, make_state_updater(state, "hit_threshold"),
                        make_reset_callback(state, "hit_threshold", "hit_threshold_slider", DEFAULTS["hit_threshold"]),
                        format_str="%.2f")

        dpg.add_separator()
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        dpg.add_text("HOG Person Detection")
        dpg.add_image("detection_texture", tag="detection_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - HOG Person Detection",
                   frame_width + 100, frame_height + 350,
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

            # Detect people
            stride = (state.win_stride, state.win_stride)
            boxes, weights = hog.detectMultiScale(frame, winStride=stride, scale=state.scale, hitThreshold=state.hit_threshold)

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

            if state.show_boxes:
                for (xA, yA, xB, yB) in boxes:
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            dpg.set_value("detection_texture", convert_cv_to_dpg(frame))

            status = f"People Detected: {len(boxes)}  |  Stride: {state.win_stride}  |  Scale: {state.scale:.2f}"
            dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
