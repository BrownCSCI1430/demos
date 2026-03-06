"""
Live Starter Demo - CSCI 1430 Brown University
A minimal template for students to experiment with image processing.

This demo shows:
- Left: Raw webcam feed
- Right: Your processed output (edit the process_frame function!)
"""

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg
from utils.demo_webcam import init_camera_demo, cleanup_camera_demo, get_frame
from utils.demo_ui import (
    add_global_controls, setup_viewport, make_state_updater, make_camera_callback,
    poll_collapsible_panels, auto_resize_images, create_blank_texture,
)

# Default values
DEFAULTS = {
    "pause": False,
    "ui_scale": 1.5,
}

GUIDE_STARTER = [
    {"title": "What is this demo?",
     "body": "A minimal template for experimenting with image processing. "
             "Left panel = raw webcam feed, right panel = your process_frame() output. "
             "Edit the code while it runs to see changes in real time."},
    {"title": "Edit process_frame()",
     "body": "Open liveStarter.py and find the process_frame() function. "
             "Replace 'output = img.copy()' with any OpenCV operation:\n"
             "  - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  (grayscale)\n"
             "  - cv2.GaussianBlur(img, (15,15), 0)      (blur)\n"
             "  - cv2.Canny(gray, 50, 150)                (edges)\n"
             "  - 255 - img                               (invert)\n"
             "  - img[:, :, 2]                            (red channel only)"},
    {"title": "Experiment",
     "body": "Save and re-run to see your changes. Chain multiple operations "
             "to build a pipeline. Toggle Cat Mode to work without a webcam."},
]


class State:
    cap = None
    frame_width = 0
    frame_height = 0
    use_camera = True
    cat_mode = False
    pause = False
    fallback_image = None


state = State()


_IMAGE_LAYOUT = None  # set after camera init

def update_image_sizes():
    auto_resize_images(_IMAGE_LAYOUT, margin_w=80, margin_h=200)


def process_frame(img):
    """
    ============================================================================
    STUDENT TODO: Edit this function to process the input image!
    ============================================================================

    This function receives the raw webcam frame and should return a processed
    version. The output will be displayed on the right side of the window.

    Args:
        img: Input image as a NumPy array (BGR format, shape: height x width x 3)

    Returns:
        Processed image as a NumPy array (can be BGR or grayscale)

    Example transformations to try:

    1. Convert to grayscale:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    2. Apply Gaussian blur:
        return cv2.GaussianBlur(img, (15, 15), 0)

    3. Detect edges with Canny:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)

    4. Flip the image:
        return cv2.flip(img, 1)  # 1 = horizontal, 0 = vertical

    5. Invert colors:
        return 255 - img

    6. Apply a threshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh

    7. Extract a single color channel:
        return img[:, :, 2]  # Red channel (BGR order: 0=Blue, 1=Green, 2=Red)

    ============================================================================
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE: Replace the line below with your own image processing!
    # -------------------------------------------------------------------------

    output = img.copy()  # Currently just copies the input (no processing)

    # -------------------------------------------------------------------------
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return output


def main():
    global _IMAGE_LAYOUT
    frame_width, frame_height = init_camera_demo(state, "Starter Template Demo")
    aspect = frame_width / frame_height
    _IMAGE_LAYOUT = [("input_image", 0.5, aspect), ("output_image", 0.5, aspect)]

    with dpg.texture_registry():
        create_blank_texture(frame_width, frame_height, "input_texture")
        create_blank_texture(frame_width, frame_height, "output_texture")

    # Create main window
    with dpg.window(label="Starter Demo", tag="main_window"):
        # Global controls
        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            camera_callback=make_camera_callback(state),
            guide=GUIDE_STARTER, guide_title="Starter Template",
        )

        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Instructions
        dpg.add_text("Edit the process_frame() function in liveStarter.py to see your changes!",
                     color=(100, 255, 100))

        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Image display - side by side
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Input (Raw Webcam)")
                dpg.add_image("input_texture", tag="input_image")
            dpg.add_spacer(width=20)
            with dpg.group():
                dpg.add_text("Output (Your Processing)")
                dpg.add_image("output_texture", tag="output_image")

    # Setup viewport
    initial_width = int(frame_width * 2 + 120)
    setup_viewport("CSCI 1430 - Starter Demo", initial_width, frame_height + 200,
                   "main_window", update_image_sizes, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        if not state.pause:
            img = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)

            if img is not None:
                # Process the frame (student code goes in process_frame!)
                output = process_frame(img)

                # Update the display textures
                dpg.set_value("input_texture", convert_cv_to_dpg(img))
                dpg.set_value("output_texture", convert_cv_to_dpg(output))

        dpg.render_dearpygui_frame()

    cleanup_camera_demo(state)


if __name__ == "__main__":
    main()
