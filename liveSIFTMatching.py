"""
Live SIFT Matching Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveSIFTMatching.py (keyboard controls)
This version: Dear PyGui controls for interactive SIFT feature matching
"""

import cv2
import os
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import init_camera, load_fallback_image, get_frame, DATA_DIR
from utils.demo_ui import setup_viewport, make_state_updater, make_reset_callback

# Default values
DEFAULTS = {
    "match_distance": 0.75,
    "show_matches": True,
    "ui_scale": 1.5,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class State:
    cap = None
    frame_width = 640
    frame_height = 480
    query_image = None
    query_index = 0
    match_distance = DEFAULTS["match_distance"]
    show_matches = DEFAULTS["show_matches"]
    images = []
    image_names = []
    use_camera = True
    cat_mode = False
    fallback_image = None


state = State()


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 50
    available_height = vp_height - 220

    aspect_ratio = 2.0

    img_width = available_width
    img_height = int(img_width / aspect_ratio)

    if img_height > available_height:
        img_height = available_height
        img_width = int(img_height * aspect_ratio)

    if dpg.does_item_exist("matching_image"):
        dpg.configure_item("matching_image", width=img_width, height=img_height)


def on_viewport_resize():
    update_image_sizes()


def update_query_image(sender, value):
    idx = state.image_names.index(value) if value in state.image_names else 0
    state.query_index = idx
    if idx < len(state.images):
        state.query_image = cv2.imread(state.images[idx], 0)


def main():
    # Find image files in data directory
    file_prefix = DATA_DIR + '/'

    image_files = ['macmillan115_2.jpg', 'macmillan115_3.jpg', 'macmillan115_4.jpg',
                   'macmillan115_1.jpg', 'macmillan115_5.jpg',
                   'salomon001_1.jpg', 'salomon001_2.jpg', 'salomon001_3.jpg', 'salomon001_4.jpg',
                   'twenty_jackson.png', 'twenty_tubman.png', 'dollar.jpg']

    state.images = [file_prefix + f for f in image_files if os.path.exists(file_prefix + f)]
    state.image_names = [os.path.basename(f) for f in state.images]

    if not state.images:
        print("Warning: No query images found in Demos folder")
        state.images = [""]
        state.image_names = ["No images"]

    # Load first query image
    if state.images[0] and os.path.exists(state.images[0]):
        state.query_image = cv2.imread(state.images[0], 0)
    else:
        state.query_image = np.zeros((100, 100), dtype=np.uint8)

    # Initialize camera
    state.cap, state.frame_width, state.frame_height, state.use_camera = init_camera()

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.cat_mode = True

    # Initialize SIFT
    detector = cv2.SIFT_create()

    dpg.create_context()

    # Create texture - use fixed size for stable rendering
    texture_width, texture_height = 1280, 480
    with dpg.texture_registry():
        blank_data = [0.0] * (texture_width * texture_height * 4)
        dpg.add_raw_texture(texture_width, texture_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="matching_texture")

    with dpg.window(label="SIFT Matching Demo", tag="main_window"):
        # Top row controls
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="Query",
                items=state.image_names,
                default_value=state.image_names[0] if state.image_names else "",
                callback=update_query_image,
                tag="query_combo",
                width=150
            )
            dpg.add_slider_float(
                label="Distance Ratio",
                default_value=state.match_distance,
                min_value=0.1, max_value=1.0,
                callback=make_state_updater(state, "match_distance"),
                tag="distance_slider",
                width=120
            )
            dpg.add_button(label="R",
                          callback=make_reset_callback(state, "match_distance", "distance_slider", DEFAULTS["match_distance"]))
            dpg.add_checkbox(
                label="Show Matches",
                default_value=state.show_matches,
                callback=make_state_updater(state, "show_matches")
            )
            dpg.add_slider_float(
                label="UI",
                default_value=DEFAULTS["ui_scale"],
                min_value=1.0, max_value=3.0,
                callback=lambda s, v: dpg.set_global_font_scale(v),
                width=80
            )
            dpg.add_spacer(width=20)
            dpg.add_checkbox(
                label="Cat Mode",
                default_value=state.cat_mode,
                callback=make_state_updater(state, "cat_mode"),
                tag="cat_mode_checkbox",
                enabled=state.use_camera
            )
            if not state.use_camera:
                dpg.add_text("(no webcam)", color=(255, 100, 100))

        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        dpg.add_text("SIFT Feature Matching (Query | Live)")
        dpg.add_image("matching_texture", tag="matching_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - SIFT Matching",
                   1100, 650,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if state.query_image is not None and state.query_image.size > 0:
            kp1, des1 = detector.detectAndCompute(state.query_image, None)
            kp2, des2 = detector.detectAndCompute(gray, None)

            good_matches = []
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                for m, n in matches:
                    if m.distance < state.match_distance * n.distance:
                        good_matches.append([m])

            if state.show_matches:
                output = cv2.drawMatchesKnn(state.query_image, kp1, gray, kp2, good_matches,
                                           flags=2, outImg=None,
                                           matchColor=(0, 155, 0),
                                           singlePointColor=(0, 255, 255))
            else:
                output = cv2.drawMatchesKnn(state.query_image, kp1, gray, kp2, [],
                                           flags=2, outImg=None,
                                           singlePointColor=(0, 255, 255))
        else:
            output = gray
            good_matches = []

        # Update texture - resize output to fixed texture size
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        output = cv2.resize(output, (texture_width, texture_height))
        output_rgba = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)
        output_data = (output_rgba.astype(np.float32) / 255.0).flatten()
        dpg.set_value("matching_texture", output_data)

        status = f"Good Matches: {len(good_matches)}  |  Distance Ratio: {state.match_distance:.2f}"
        dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
