"""
Live HOG Person Detector Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveHOGPersonDetector.py
This version: Dear PyGui sliders for interactive HOG control
"""

import cv2
import os
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg, init_camera, load_fallback_image, get_frame
from utils.demo_ui import load_fonts, setup_viewport, make_state_updater, make_reset_callback, add_global_controls

# Default values
DEFAULTS = {
    "win_stride": 8,
    "scale": 1.05,
    "hit_threshold": 0.0,
    "show_boxes": True,
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
        state.frame_width = 640
        state.frame_height = 480

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
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
        def _extra_reset():
            if dpg.does_item_exist("stride_slider"):
                dpg.set_value("stride_slider", DEFAULTS["win_stride"])

        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            reset_extra=_extra_reset,
            guide=GUIDE_HOG, guide_title="HOG Person Detection",
        )

        dpg.add_separator()

        # Detection parameters
        with dpg.collapsing_header(label="Detection Parameters", default_open=True):
            dpg.add_checkbox(
                label="Show Boxes", default_value=state.show_boxes,
                callback=make_state_updater(state, "show_boxes"),
                tag="show_boxes_checkbox",
            )
            with dpg.table(header_row=False,
                           borders_innerV=False, borders_outerV=False,
                           borders_innerH=False, borders_outerH=False,
                           policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column()  # label (auto-fit)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=30)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=20)
                dpg.add_table_column()  # label (auto-fit)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                with dpg.table_row():
                    dpg.add_text("Win Stride")
                    dpg.add_slider_int(tag="stride_slider", default_value=state.win_stride,
                                       min_value=4, max_value=16,
                                       callback=make_state_updater(state, "win_stride"),
                                       width=80)
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "win_stride", "stride_slider", DEFAULTS["win_stride"]),
                                   width=25)
                    dpg.add_spacer(width=20)
                    dpg.add_text("Scale")
                    dpg.add_slider_float(tag="scale_slider", default_value=state.scale,
                                         min_value=1.01, max_value=1.5,
                                         callback=make_state_updater(state, "scale"),
                                         width=80, format="%.2f")
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "scale", "scale_slider", DEFAULTS["scale"]),
                                   width=25)

                with dpg.table_row():
                    dpg.add_text("Hit Threshold")
                    dpg.add_slider_float(tag="threshold_slider", default_value=state.hit_threshold,
                                         min_value=0.0, max_value=1.0,
                                         callback=make_state_updater(state, "hit_threshold"),
                                         width=80, format="%.2f")
                    dpg.add_button(label="R",
                                   callback=make_reset_callback(state, "hit_threshold", "threshold_slider", DEFAULTS["hit_threshold"]),
                                   width=25)
                    dpg.add_spacer(width=20)
                    dpg.add_spacer(width=80)
                    dpg.add_spacer(width=100)
                    dpg.add_spacer(width=30)

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
        frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode,
                          (state.frame_width, state.frame_height))
        if frame is None:
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
