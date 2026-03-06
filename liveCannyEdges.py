"""
Live Canny Edge Detection Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveCannyEdges.py (keyboard controls)
This version: Dear PyGui sliders and toggles for interactive control
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg
from utils.demo_webcam import init_camera_demo, cleanup_camera_demo, get_frame
from utils.demo_ui import (
    add_global_controls, setup_viewport, make_state_updater, make_camera_callback,
    make_reset_callback, control_panel, create_parameter_table, add_parameter_row,
    poll_collapsible_panels, auto_resize_images, create_blank_texture,
)

# Default values
DEFAULTS = {
    "blur_sigma": 1.0,
    "canny_thresh_low": 10,
    "canny_thresh_high": 70,
    "pause": False,
    "ui_scale": 1.5,
}

GUIDE_CANNY = [
    {"title": "Gaussian blur preprocessing",
     "body": "Smooth the image to reduce noise before detecting edges. "
             "Higher sigma = stronger smoothing = less noise but may blur "
             "fine edges. The blur kernel size is fixed at 11x11."},
    {"title": "Gradient computation",
     "body": "Sobel filters in x and y directions compute the gradient magnitude "
             "and direction at each pixel. Strong gradients indicate potential edges."},
    {"title": "Non-maximum suppression",
     "body": "Thins gradient ridges to 1-pixel-wide edges by keeping only "
             "local maxima along the gradient direction. This prevents thick, "
             "blurry edge responses."},
    {"title": "Hysteresis thresholding",
     "body": "Two thresholds classify edge pixels:\n"
             "  - Below Low: discard (not an edge)\n"
             "  - Above High: strong edge (always kept)\n"
             "  - Between: kept only if connected to a strong edge\n"
             "Widen the gap for more edges; narrow it for fewer. "
             "Try: low=10, high=70 (default) vs low=50, high=200 (strict)."},
]


# Global state
class State:
    blur_sigma = DEFAULTS["blur_sigma"]
    canny_thresh_low = DEFAULTS["canny_thresh_low"]
    canny_thresh_high = DEFAULTS["canny_thresh_high"]
    cap = None
    frame_width = 0
    frame_height = 0
    input_ratio = 0.25  # Input image is 1/4 the size of canny image
    use_camera = True
    cat_mode = False
    pause = False
    fallback_image = None


state = State()


def update_image_sizes():
    r = state.input_ratio
    aspect = state.frame_width / state.frame_height
    layout = [("input_image", r / (1 + r), aspect), ("canny_image", 1 / (1 + r), aspect)]
    auto_resize_images(layout, margin_w=50, margin_h=240)


# Canny threshold callbacks need special logic to keep low <= high
def update_canny_low(sender, value):
    state.canny_thresh_low = int(value)
    if state.canny_thresh_low > state.canny_thresh_high:
        state.canny_thresh_high = state.canny_thresh_low
        dpg.set_value("canny_thresh_high_slider", state.canny_thresh_high)


def update_canny_high(sender, value):
    state.canny_thresh_high = int(value)
    if state.canny_thresh_high < state.canny_thresh_low:
        state.canny_thresh_low = state.canny_thresh_high
        dpg.set_value("canny_thresh_low_slider", state.canny_thresh_low)


def process_frame():
    """Capture and process a single frame"""
    img = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode,
                    (state.frame_width, state.frame_height))
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), state.blur_sigma)
    canny = cv2.Canny(blur, state.canny_thresh_low, state.canny_thresh_high)

    return gray, canny


def main():
    frame_width, frame_height = init_camera_demo(state, "Canny Edge Detection Demo")

    with dpg.texture_registry():
        create_blank_texture(frame_width, frame_height, "input_texture")
        create_blank_texture(frame_width, frame_height, "canny_texture")

    # Create main window
    with dpg.window(label="Canny Edge Detection Demo", tag="main_window"):
        # Global controls
        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            camera_callback=make_camera_callback(state),
            guide=GUIDE_CANNY, guide_title="Canny Edge Detection",
        )

        dpg.add_separator()

        # Edge Detection parameters
        with dpg.group(horizontal=True):
            with control_panel("Edge Detection", width=350, height=140,
                               color=(150, 200, 255)):
                with create_parameter_table():
                    add_parameter_row(
                        "Blur Sigma", "blur_sigma_slider", DEFAULTS["blur_sigma"],
                        0.1, 10.0, make_state_updater(state, "blur_sigma"),
                        make_reset_callback(state, "blur_sigma", "blur_sigma_slider", DEFAULTS["blur_sigma"]),
                        format_str="%.1f")
                    add_parameter_row(
                        "Canny Low", "canny_thresh_low_slider", DEFAULTS["canny_thresh_low"],
                        1, 255, update_canny_low,
                        make_reset_callback(state, "canny_thresh_low", "canny_thresh_low_slider", DEFAULTS["canny_thresh_low"]),
                        slider_type="int")
                    add_parameter_row(
                        "Canny High", "canny_thresh_high_slider", DEFAULTS["canny_thresh_high"],
                        1, 255, update_canny_high,
                        make_reset_callback(state, "canny_thresh_high", "canny_thresh_high_slider", DEFAULTS["canny_thresh_high"]),
                        slider_type="int")

        dpg.add_separator()
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        # Image display
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Input Stream")
                dpg.add_image("input_texture", tag="input_image")
            with dpg.group():
                dpg.add_text("Canny Edges")
                dpg.add_image("canny_texture", tag="canny_image")

    # Setup viewport
    initial_width = int(frame_width * (1 + state.input_ratio) + 100)
    setup_viewport("CSCI 1430 - Canny Edge Detection", initial_width, frame_height + 280,
                   "main_window", update_image_sizes, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        if not state.pause:
            gray, canny = process_frame()

            if gray is not None and canny is not None:
                dpg.set_value("input_texture", convert_cv_to_dpg(gray))
                dpg.set_value("canny_texture", convert_cv_to_dpg(canny))

                status = f"Blur Sigma: {state.blur_sigma:.1f}  |  Canny Low: {state.canny_thresh_low}  |  Canny High: {state.canny_thresh_high}"
                dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    cleanup_camera_demo(state)


if __name__ == "__main__":
    main()
