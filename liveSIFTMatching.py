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

from utils.demo_utils import convert_cv_to_dpg
from utils.demo_webcam import init_camera_demo, cleanup_camera_demo, get_frame, DATA_DIR
from utils.demo_ui import (
    setup_viewport, make_state_updater, make_reset_callback, make_camera_callback,
    add_global_controls, control_panel, create_parameter_table, add_parameter_row,
    poll_collapsible_panels, auto_resize_images,
)

# Default values
DEFAULTS = {
    "match_distance": 0.55,
    "show_matches": True,
    "pause": False,
    "ui_scale": 1.5,
}

GUIDE_SIFT = [
    {"title": "Scale-invariant features",
     "body": "SIFT finds keypoints that are stable across scale, rotation, and "
             "illumination changes by searching a scale-space pyramid built from "
             "Difference-of-Gaussians. Unlike Harris corners, SIFT features "
             "persist at multiple resolutions."},
    {"title": "Descriptors",
     "body": "Each keypoint gets a 128-dimensional descriptor encoding the local "
             "gradient distribution in a 16x16 patch around it. The descriptor is "
             "normalized and aligned to the dominant gradient orientation, making "
             "it robust to rotation and illumination changes."},
    {"title": "KNN matching and Lowe's ratio test",
     "body": "For each query descriptor, find the 2 nearest neighbors in the live "
             "frame. Keep the match only if the best distance is significantly "
             "less than the second-best (controlled by the Distance Ratio slider). "
             "Lower ratio = stricter matching = fewer but more reliable matches."},
    {"title": "Try it",
     "body": "Select different query images and move the camera to match features. "
             "Try rotating or tilting the view. SIFT matches survive geometric "
             "changes that would break simpler methods like template matching. "
             "Adjust the distance ratio to trade precision for recall."},
]


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
    pause = False
    fallback_image = None


state = State()


def update_image_sizes():
    aspect = state.texture_width / state.texture_height
    auto_resize_images([("matching_image", 1.0, aspect)], margin_w=50, margin_h=220)


def compute_sift_matches(frame, query_image, detector, match_distance, show_matches):
    """Run SIFT matching between query and live frame.

    Returns (output_bgr, good_matches_list).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if query_image is None or query_image.size == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []

    kp1, des1 = detector.detectAndCompute(query_image, None)
    kp2, des2 = detector.detectAndCompute(gray, None)

    good_matches = []
    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < match_distance * n.distance:
                    good_matches.append([m])

    draw_matches = good_matches if show_matches else []
    output = cv2.drawMatchesKnn(query_image, kp1, gray, kp2, draw_matches,
                                flags=2, outImg=None,
                                matchColor=(0, 155, 0),
                                singlePointColor=(0, 255, 255))
    return output, good_matches


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

    # Initialize camera, fallback, DPG context, and fonts
    frame_width, frame_height = init_camera_demo(state, "SIFT Feature Matching Demo")

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

    # Initialize SIFT
    detector = cv2.SIFT_create()

    # Calculate dynamic texture size based on camera and query dimensions
    query_width = state.query_image.shape[1] if state.query_image is not None and state.query_image.size > 0 else state.frame_width
    state.texture_width = query_width + state.frame_width  # Side-by-side width
    state.texture_height = state.frame_height  # Fixed height matches camera

    # Create texture registry and initial texture
    texture_width, texture_height = state.texture_width, state.texture_height
    state.texture_registry = dpg.add_texture_registry()
    blank_data = [0.0] * (texture_width * texture_height * 4)
    state.current_texture_tag = "matching_texture_0"
    dpg.add_raw_texture(texture_width, texture_height, blank_data,
                      format=dpg.mvFormat_Float_rgba, tag=state.current_texture_tag,
                      parent=state.texture_registry)

    with dpg.window(label="SIFT Matching Demo", tag="main_window"):
        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            camera_callback=make_camera_callback(state),
            guide=GUIDE_SIFT, guide_title="SIFT Feature Matching",
        )

        dpg.add_separator()

        with dpg.group(horizontal=True):
            with control_panel("Matching", width=500, height=140,
                               color=(150, 200, 255)):
                dpg.add_combo(
                    label="Query",
                    items=state.image_names,
                    default_value=state.image_names[0] if state.image_names else "",
                    callback=update_query_image,
                    tag="query_combo",
                    width=150
                )
                with create_parameter_table():
                    add_parameter_row(
                        "Distance Ratio", "match_distance_slider", DEFAULTS["match_distance"],
                        0.1, 1.0, make_state_updater(state, "match_distance"),
                        make_reset_callback(state, "match_distance", "match_distance_slider", DEFAULTS["match_distance"]),
                        format_str="%.2f")
                dpg.add_checkbox(
                    label="Show Matches",
                    default_value=state.show_matches,
                    callback=make_state_updater(state, "show_matches")
                )

        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        dpg.add_text("SIFT Feature Matching (Query | Live)")
        dpg.add_image(state.current_texture_tag, tag="matching_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - SIFT Matching",
                   1100, 650,
                   "main_window", update_image_sizes, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
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

        if not state.pause:
            frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)
            if frame is None:
                dpg.render_dearpygui_frame()
                continue

            output, good_matches = compute_sift_matches(
                frame, state.query_image, detector,
                state.match_distance, state.show_matches)

            # Update texture - resize output to match texture size
            if len(output.shape) == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

            output = cv2.resize(output, (state.texture_width, state.texture_height))

            # CRITICAL: Skip texture update if recreation is in progress to prevent buffer size mismatch
            if not state.texture_needs_recreate:
                dpg.set_value(state.current_texture_tag, convert_cv_to_dpg(output))

            status = f"Good Matches: {len(good_matches)}  |  Distance Ratio: {state.match_distance:.2f}"
            dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    cleanup_camera_demo(state)


if __name__ == "__main__":
    main()
