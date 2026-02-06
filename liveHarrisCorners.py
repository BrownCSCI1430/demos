"""
Live Harris Corners Demo with Dear PyGui controls
CSCI 1430 - Brown University

Original: liveHarrisCorners.py
This version: Dear PyGui sliders for interactive Harris corner detection
"""

import cv2
import os
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import (convert_cv_to_dpg, resize_with_letterbox, init_camera, load_fallback_image,
                              apply_affine_transform, apply_brightness)
from utils.demo_ui import setup_viewport, make_state_updater, make_reset_callback

# Default values
DEFAULTS = {
    "block_size": 2,
    "ksize": 5,
    "k": 0.07,
    "threshold": 0.01,
    "ui_scale": 1.5,
    "rotation": 0.0,
    "scale": 1.0,
    "translate_x": 0.0,
    "translate_y": 0.0,
    "brightness_scale": 1.0,
    "brightness_shift": 0.0,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class State:
    cap = None
    frame_width = 0
    frame_height = 0
    block_size = DEFAULTS["block_size"]
    ksize = DEFAULTS["ksize"]
    k = DEFAULTS["k"]
    threshold = DEFAULTS["threshold"]
    use_camera = True
    cat_mode = False
    fallback_image = None
    paused = False
    paused_frame = None
    rotation = DEFAULTS["rotation"]
    scale = DEFAULTS["scale"]
    translate_x = DEFAULTS["translate_x"]
    translate_y = DEFAULTS["translate_y"]
    brightness_scale = DEFAULTS["brightness_scale"]
    brightness_shift = DEFAULTS["brightness_shift"]


state = State()


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = (vp_width - 70) // 2
    available_height = vp_height - 260

    aspect_ratio = state.frame_width / state.frame_height

    img_width = available_width
    img_height = int(img_width / aspect_ratio)

    if img_height > available_height:
        img_height = available_height
        img_width = int(img_height * aspect_ratio)

    if dpg.does_item_exist("original_image"):
        dpg.configure_item("original_image", width=img_width, height=img_height)
    if dpg.does_item_exist("transformed_image"):
        dpg.configure_item("transformed_image", width=img_width, height=img_height)


def on_viewport_resize():
    update_image_sizes()


# Sobel ksize must be odd
def update_ksize(sender, value):
    state.ksize = value if value % 2 == 1 else value + 1
    dpg.set_value(sender, state.ksize)


def toggle_pause(sender, value):
    state.paused = value
    if not value:
        state.paused_frame = None


# Multi-value reset for translate
def reset_translate():
    state.translate_x = DEFAULTS["translate_x"]
    state.translate_y = DEFAULTS["translate_y"]
    dpg.set_value("translate_x_slider", state.translate_x)
    dpg.set_value("translate_y_slider", state.translate_y)


# Multi-value reset for brightness
def reset_brightness():
    state.brightness_scale = DEFAULTS["brightness_scale"]
    state.brightness_shift = DEFAULTS["brightness_shift"]
    dpg.set_value("brightness_scale_slider", state.brightness_scale)
    dpg.set_value("brightness_shift_slider", state.brightness_shift)


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

    dpg.create_context()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="original_texture")
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                          format=dpg.mvFormat_Float_rgba, tag="transformed_texture")

    with dpg.window(label="Harris Corners Demo", tag="main_window"):
        # Global controls row
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                label="UI Scale",
                default_value=DEFAULTS["ui_scale"],
                min_value=1.0, max_value=3.0,
                callback=lambda s, v: dpg.set_global_font_scale(v),
                width=100
            )
            dpg.add_spacer(width=20)
            dpg.add_checkbox(
                label="Cat Mode",
                default_value=state.cat_mode,
                callback=make_state_updater(state, "cat_mode"),
                tag="cat_mode_checkbox",
                enabled=state.use_camera
            )
            dpg.add_checkbox(
                label="Pause",
                default_value=state.paused,
                callback=toggle_pause,
                tag="pause_checkbox"
            )
            if not state.use_camera:
                dpg.add_text("(no webcam)", color=(255, 100, 100))

        dpg.add_separator()

        # Horizontal sections using child_window containers
        with dpg.group(horizontal=True):
            # Column 1: Harris Detection
            with dpg.child_window(width=250, height=160, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Harris Detection", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("Block Size")
                            dpg.add_slider_int(tag="block_slider", default_value=state.block_size,
                                               min_value=2, max_value=10,
                                               callback=make_state_updater(state, "block_size"), width=80)
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "block_size", "block_slider", DEFAULTS["block_size"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Sobel K")
                            dpg.add_slider_int(tag="ksize_slider", default_value=state.ksize,
                                               min_value=3, max_value=31, callback=update_ksize, width=80)
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "ksize", "ksize_slider", DEFAULTS["ksize"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Harris k")
                            dpg.add_slider_float(tag="k_slider", default_value=state.k,
                                                 min_value=0.01, max_value=0.3,
                                                 callback=make_state_updater(state, "k"), width=80, format="%.3f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "k", "k_slider", DEFAULTS["k"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Threshold")
                            dpg.add_slider_float(tag="threshold_slider", default_value=state.threshold,
                                                 min_value=0.001, max_value=0.1,
                                                 callback=make_state_updater(state, "threshold"), width=80, format="%.4f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "threshold", "threshold_slider", DEFAULTS["threshold"]),
                                           width=25)

            dpg.add_spacer(width=10)

            # Column 2: Transforms
            with dpg.child_window(width=300, height=160, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Transforms", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=130)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("Rotate")
                            dpg.add_slider_float(tag="rotation_slider", default_value=state.rotation,
                                                 min_value=-180.0, max_value=180.0,
                                                 callback=make_state_updater(state, "rotation"), width=80, format="%.1f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "rotation", "rotation_slider", DEFAULTS["rotation"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Scale")
                            dpg.add_slider_float(tag="scale_slider", default_value=state.scale,
                                                 min_value=0.25, max_value=4.0,
                                                 callback=make_state_updater(state, "scale"), width=80, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "scale", "scale_slider", DEFAULTS["scale"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Translate X")
                            dpg.add_slider_float(tag="translate_x_slider", default_value=state.translate_x,
                                                 min_value=-50.0, max_value=50.0,
                                                 callback=make_state_updater(state, "translate_x"), width=80, format="%.1f")
                            dpg.add_button(label="R", callback=reset_translate, width=25)

                        with dpg.table_row():
                            dpg.add_text("Translate Y")
                            dpg.add_slider_float(tag="translate_y_slider", default_value=state.translate_y,
                                                 min_value=-50.0, max_value=50.0,
                                                 callback=make_state_updater(state, "translate_y"), width=80, format="%.1f")
                            dpg.add_spacer(width=25)

            dpg.add_spacer(width=10)

            # Column 3: Brightness
            with dpg.child_window(width=250, height=160, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Brightness", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("Scale")
                            dpg.add_slider_float(tag="brightness_scale_slider", default_value=state.brightness_scale,
                                                 min_value=0.5, max_value=2.0,
                                                 callback=make_state_updater(state, "brightness_scale"), width=80, format="%.2f")
                            dpg.add_button(label="R", callback=reset_brightness, width=25)

                        with dpg.table_row():
                            dpg.add_text("Shift")
                            dpg.add_slider_float(tag="brightness_shift_slider", default_value=state.brightness_shift,
                                                 min_value=-100.0, max_value=100.0,
                                                 callback=make_state_updater(state, "brightness_shift"), width=80, format="%.1f")
                            dpg.add_spacer(width=25)

        dpg.add_separator()
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        # Side-by-side images
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Original")
                dpg.add_image("original_texture", tag="original_image")
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_text("Transformed")
                dpg.add_image("transformed_texture", tag="transformed_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - Harris Corners",
                   max(frame_width * 2 + 150, 850), frame_height + 280,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        # Get frame
        if state.use_camera and not state.cat_mode:
            ret, img = state.cap.read()
            if not ret:
                continue
        else:
            img = state.fallback_image.copy()
            if img.shape[1] != state.frame_width or img.shape[0] != state.frame_height:
                img = resize_with_letterbox(img, state.frame_width, state.frame_height)

        # Handle pause
        if state.paused:
            if state.paused_frame is None:
                state.paused_frame = img.copy()
            img_orig = state.paused_frame.copy()
        else:
            img_orig = img.copy()

        # Process original image
        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        harris_orig = cv2.cornerHarris(gray_orig.astype(np.float32), state.block_size, state.ksize, state.k)
        harris_orig = cv2.dilate(harris_orig, None)
        corner_mask_orig = harris_orig > state.threshold * harris_orig.max()
        num_corners_orig = np.sum(corner_mask_orig)
        img_orig[corner_mask_orig] = [0, 0, 255]

        # Process transformed image
        img_trans = apply_affine_transform(img, state.rotation, state.scale, state.translate_x, state.translate_y)
        img_trans = apply_brightness(img_trans, state.brightness_scale, state.brightness_shift)
        gray_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)
        harris_trans = cv2.cornerHarris(gray_trans.astype(np.float32), state.block_size, state.ksize, state.k)
        harris_trans = cv2.dilate(harris_trans, None)
        corner_mask_trans = harris_trans > state.threshold * harris_trans.max()
        num_corners_trans = np.sum(corner_mask_trans)
        img_trans[corner_mask_trans] = [0, 0, 255]

        # Update textures
        dpg.set_value("original_texture", convert_cv_to_dpg(img_orig))
        dpg.set_value("transformed_texture", convert_cv_to_dpg(img_trans))

        status = f"Original: {num_corners_orig} corners  |  Transformed: {num_corners_trans} corners  |  Block: {state.block_size}  |  k: {state.k:.3f}"
        dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
