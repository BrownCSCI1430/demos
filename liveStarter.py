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

from utils.demo_utils import convert_cv_to_dpg, resize_with_letterbox, init_camera, load_fallback_image
from utils.demo_ui import add_global_controls, setup_viewport, make_state_updater

# Default values
DEFAULTS = {
    "ui_scale": 1.5,
}


class State:
    cap = None
    frame_width = 0
    frame_height = 0
    use_camera = True
    cat_mode = False
    fallback_image = None


state = State()


def update_image_sizes():
    """Update image display sizes based on viewport dimensions"""
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 80
    available_height = vp_height - 200

    aspect_ratio = state.frame_width / state.frame_height

    # Each image gets half the available width
    img_width = int(available_width / 2)
    img_height = int(img_width / aspect_ratio)

    if img_height > available_height:
        img_height = available_height
        img_width = int(img_height * aspect_ratio)

    if dpg.does_item_exist("input_image"):
        dpg.configure_item("input_image", width=img_width, height=img_height)
    if dpg.does_item_exist("output_image"):
        dpg.configure_item("output_image", width=img_width, height=img_height)


def on_viewport_resize():
    update_image_sizes()


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


def get_frame():
    """Capture a single frame from camera or fallback image"""
    if state.use_camera and not state.cat_mode:
        if state.cap is None or not state.cap.isOpened():
            return None
        ret, img = state.cap.read()
        if not ret:
            return None
    else:
        if state.fallback_image is None:
            return None
        img = state.fallback_image.copy()
        img = resize_with_letterbox(img, state.frame_width, state.frame_height)

    return img


def main():
    # Initialize camera
    state.cap, state.frame_width, state.frame_height, state.use_camera = init_camera()

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.frame_height, state.frame_width = state.fallback_image.shape[:2]
        state.cat_mode = True

    frame_width, frame_height = state.frame_width, state.frame_height

    # Initialize Dear PyGui
    dpg.create_context()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                            format=dpg.mvFormat_Float_rgba, tag="input_texture")
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                            format=dpg.mvFormat_Float_rgba, tag="output_texture")

    # Create main window
    with dpg.window(label="Starter Demo", tag="main_window"):
        # Global controls
        add_global_controls(DEFAULTS, state, make_state_updater(state, "cat_mode"))

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
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        img = get_frame()

        if img is not None:
            # Process the frame (student code goes in process_frame!)
            output = process_frame(img)

            # Update the display textures
            dpg.set_value("input_texture", convert_cv_to_dpg(img))
            dpg.set_value("output_texture", convert_cv_to_dpg(output))

        dpg.render_dearpygui_frame()

    # Cleanup
    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
