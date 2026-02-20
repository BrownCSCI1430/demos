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
from utils.demo_ui import load_fonts, setup_viewport, make_state_updater, make_reset_callback

# Default values
DEFAULTS = {
    "match_distance": 0.55,
    "show_matches": True,
    "ui_scale": 1.5,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class State:
    cap = None
    frame_width = 640
    frame_height = 480
    texture_width = 1280  # Will be updated dynamically
    texture_height = 480
    texture_needs_recreate = False  # Flag to recreate texture in main loop
    texture_registry = None  # Store texture registry reference
    texture_counter = 0  # Counter for unique texture tags
    current_texture_tag = "matching_texture_0"  # Current texture tag in use
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

    # Use actual texture aspect ratio instead of hardcoded value
    aspect_ratio = state.texture_width / state.texture_height

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
        # Normalize to camera height while preserving aspect ratio
        if state.query_image is not None and state.query_image.size > 0:
            query_height = state.frame_height
            h, w = state.query_image.shape[:2]

            # Validate dimensions
            if h <= 0 or w <= 0:
                print(f"Warning: Invalid image dimensions {w}x{h}")
                return

            aspect_ratio = w / h
            query_width = int(query_height * aspect_ratio)

            # Ensure minimum dimensions
            if query_width < 1 or query_height < 1:
                print(f"Warning: Calculated dimensions too small {query_width}x{query_height}")
                return

            state.query_image = cv2.resize(state.query_image, (query_width, query_height))

            # Update texture dimensions for side-by-side display
            state.texture_width = query_width + state.frame_width

            # Set flag to recreate texture in main loop
            # Note: Don't call update_image_sizes() here - it will be called after texture recreation
            state.texture_needs_recreate = True


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='SIFT Feature Matching Demo')
    parser.add_argument('--width', type=int, default=None, help='Camera width')
    parser.add_argument('--height', type=int, default=None, help='Camera height')
    args = parser.parse_args()

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

    # Initialize camera first to get correct dimensions
    state.cap, state.frame_width, state.frame_height, state.use_camera = \
        init_camera(width=args.width, height=args.height)

    # Load first query image after camera initialization
    if state.images[0] and os.path.exists(state.images[0]):
        state.query_image = cv2.imread(state.images[0], 0)
        # Normalize query image to match camera height while preserving aspect ratio
        if state.query_image is not None and state.query_image.size > 0:
            query_height = state.frame_height
            h, w = state.query_image.shape[:2]
            if h > 0 and w > 0:
                aspect_ratio = w / h
                query_width = int(query_height * aspect_ratio)
                if query_width > 0 and query_height > 0:
                    state.query_image = cv2.resize(state.query_image, (query_width, query_height))
    else:
        state.query_image = np.zeros((100, 100), dtype=np.uint8)

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.cat_mode = True

    # Initialize SIFT
    detector = cv2.SIFT_create()

    # Calculate dynamic texture size based on camera and query dimensions
    query_width = state.query_image.shape[1] if state.query_image is not None and state.query_image.size > 0 else state.frame_width
    state.texture_width = query_width + state.frame_width  # Side-by-side width
    state.texture_height = state.frame_height  # Fixed height matches camera

    dpg.create_context()

    load_fonts()

    # Create texture registry and initial texture
    texture_width, texture_height = state.texture_width, state.texture_height
    state.texture_registry = dpg.add_texture_registry()
    blank_data = [0.0] * (texture_width * texture_height * 4)
    state.current_texture_tag = "matching_texture_0"
    dpg.add_raw_texture(texture_width, texture_height, blank_data,
                      format=dpg.mvFormat_Float_rgba, tag=state.current_texture_tag,
                      parent=state.texture_registry)

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
            dpg.add_combo(
                label="UI",
                items=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"],
                default_value=str(DEFAULTS["ui_scale"]),
                callback=lambda s, v: dpg.set_global_font_scale(float(v)),
                width=60
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
        dpg.add_image(state.current_texture_tag, tag="matching_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - SIFT Matching",
                   1100, 650,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        # Recreate texture when query image changes
        if state.texture_needs_recreate:
            # Validate texture dimensions before recreating
            if state.texture_width < 1 or state.texture_height < 1:
                print(f"Warning: Invalid texture dimensions {state.texture_width}x{state.texture_height}, skipping recreation")
                state.texture_needs_recreate = False
                continue

            # Save old texture tag
            old_texture_tag = state.current_texture_tag

            # Create new texture with unique tag BEFORE deleting old one
            state.texture_counter += 1
            new_texture_tag = f"matching_texture_{state.texture_counter}"
            blank_data = [0.0] * (state.texture_width * state.texture_height * 4)
            dpg.add_raw_texture(state.texture_width, state.texture_height, blank_data,
                              format=dpg.mvFormat_Float_rgba, tag=new_texture_tag,
                              parent=state.texture_registry)

            # Rebind image widget to the new texture
            if dpg.does_item_exist("matching_image"):
                dpg.configure_item("matching_image", texture_tag=new_texture_tag)

            # Update current texture tag
            state.current_texture_tag = new_texture_tag

            # Now delete the old texture
            if dpg.does_item_exist(old_texture_tag):
                dpg.delete_item(old_texture_tag)

            # Update display size to match new aspect ratio
            update_image_sizes()

            state.texture_needs_recreate = False
            continue  # Skip frame processing during recreation to prevent timing issues

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

                for match in matches:
                    if len(match) == 2:  # Only process if we got 2 matches
                        m, n = match
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

        # Update texture - resize output to match texture size
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        output = cv2.resize(output, (state.texture_width, state.texture_height))
        output_rgba = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)
        output_data = (output_rgba.astype(np.float32) / 255.0).flatten()

        # CRITICAL: Skip texture update if recreation is in progress to prevent buffer size mismatch
        if not state.texture_needs_recreate:
            dpg.set_value(state.current_texture_tag, output_data)

        status = f"Good Matches: {len(good_matches)}  |  Distance Ratio: {state.match_distance:.2f}"
        dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
