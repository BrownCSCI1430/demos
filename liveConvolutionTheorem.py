"""
Live Convolution Theorem Demo with Dear PyGui controls
CSCI 1430 - Brown University

Demonstrates the convolution theorem: convolution in spatial domain
equals multiplication in frequency domain (and vice versa for deconvolution).
"""

import cv2
import math
import os
import numpy as np
import dearpygui.dearpygui as dpg
from skimage import img_as_float
from skimage.color import rgb2gray

from utils.demo_utils import init_camera, load_fallback_image, convert_cv_to_dpg_float, crop_to_square, get_frame
from utils.demo_ui import (
    load_fonts, setup_viewport, make_state_updater, make_reset_callback,
    add_global_controls, control_panel, create_parameter_table, add_parameter_row,
    poll_collapsible_panels,
)
from utils.demo_kernels import create_kernel, pad_kernel_to_image_size, create_gaussian_kernel_fft, visualize_kernel
from utils.demo_fft import visualize_fft_amplitude, process_convolution, process_deconvolution

# Default values
DEFAULTS = {
    "kernel_type": "Gaussian",
    "kernel_size": 15,
    "gaussian_sigma": 3.0,
    "mode": "Convolution",
    "use_regularization": False,
    "regularization": 0.01,
    "pause": False,
    "ui_scale": 1.5,
}


GUIDE_CONV_THEOREM = [
    {"title": "The convolution theorem",
     "body": "Convolution in the spatial domain equals element-wise multiplication "
             "in the frequency domain:\n"
             "  FFT(f \u2217 g) = FFT(f) \u00b7 FFT(g)\n"
             "Both sides are shown simultaneously so you can verify they match."},
    {"title": "Kernel frequency response",
     "body": "Gaussian = low-pass filter (attenuates high frequencies).\n"
             "Sharpen = amplifies high frequencies.\n"
             "Edge kernels = zero DC response (output sums to zero).\n"
             "Watch the kernel's FFT to see what frequencies it passes or blocks."},
    {"title": "Convolution mode",
     "body": "Top row: spatial convolution via cv2.filter2D.\n"
             "Bottom row: frequency multiplication (FFT \u2192 multiply \u2192 inverse FFT).\n"
             "The results should match, demonstrating the convolution theorem."},
    {"title": "Deconvolution",
     "body": "Switch to Deconvolution mode to invert a blur by dividing in the "
             "frequency domain. But dividing by small values (where the kernel's "
             "FFT is near zero) amplifies noise dramatically."},
    {"title": "Regularization",
     "body": "Stabilizes deconvolution using Wiener-style filtering:\n"
             "  H*/(|H|\u00b2 + \u03b5)  instead of  1/H\n"
             "The slider controls \u03b5:\n"
             "  Too small \u2192 noise explosion\n"
             "  Too large \u2192 image stays blurry\n"
             "Find the sweet spot that recovers detail without amplifying noise."},
]

# Available kernels
KERNELS = ["Box", "Gaussian", "Sharpen", "Edge Horizontal", "Edge Vertical", "Random"]


class State:
    cap = None
    use_camera = True
    cat_mode = False
    frame_size = 200
    fallback_image = None
    pause = False

    kernel_type = DEFAULTS["kernel_type"]
    kernel_size = DEFAULTS["kernel_size"]
    gaussian_sigma = DEFAULTS["gaussian_sigma"]

    random_kernel = None
    random_kernel_size = None

    mode = DEFAULTS["mode"]

    use_regularization = DEFAULTS["use_regularization"]
    regularization = DEFAULTS["regularization"]

    blurred_image = None
    original_for_deconv = None


state = State()


def create_random_kernel(size):
    """Create a random kernel, caching it in state."""
    if state.random_kernel is None or state.random_kernel_size != size:
        kernel = np.random.randn(size, size).astype(np.float64)
        kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel / (size * size)
        state.random_kernel = kernel
        state.random_kernel_size = size
    return state.random_kernel


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 200
    available_height = vp_height - 420

    size = min(available_width // 3, available_height // 2)

    for tag in ["input_image", "kernel_image", "result_image",
                "image_fft", "kernel_fft", "product_fft"]:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=size, height=size)


def on_viewport_resize():
    update_image_sizes()


def draw_convolution_symbol(drawlist_tag):
    """Draw an asterisk symbol for convolution"""
    dpg.delete_item(drawlist_tag, children_only=True)
    cx, cy = 30, 30
    r = 20
    color = (255, 255, 100, 255)
    thickness = 3
    for i in range(3):
        angle = i * math.pi / 3
        x1 = cx + r * math.cos(angle)
        y1 = cy + r * math.sin(angle)
        x2 = cx - r * math.cos(angle)
        y2 = cy - r * math.sin(angle)
        dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=thickness, parent=drawlist_tag)


def draw_deconvolution_symbol(drawlist_tag):
    """Draw a circle with a slash for deconvolution"""
    dpg.delete_item(drawlist_tag, children_only=True)
    cx, cy = 30, 30
    r = 20
    color = (255, 255, 100, 255)
    thickness = 3
    dpg.draw_circle((cx, cy), r, color=color, thickness=thickness, parent=drawlist_tag)
    offset = r * math.cos(math.pi / 4)
    dpg.draw_line((cx - offset, cy + offset), (cx + offset, cy - offset),
                  color=color, thickness=thickness, parent=drawlist_tag)


def update_mode(sender, value):
    state.mode = value
    is_deconv = state.mode == "Deconvolution"
    if dpg.does_item_exist("regularization_panel"):
        dpg.configure_item("regularization_panel", show=is_deconv)
    if state.mode == "Convolution":
        if dpg.does_item_exist("spatial_operator_drawing"):
            draw_convolution_symbol("spatial_operator_drawing")
        if dpg.does_item_exist("operator_label"):
            dpg.set_value("operator_label", "\u00d7")
        if dpg.does_item_exist("input_label"):
            dpg.set_value("input_label", "Input Image")
        if dpg.does_item_exist("result_label"):
            dpg.set_value("result_label", "Convolved Result")
        if dpg.does_item_exist("image_fft_label"):
            dpg.set_value("image_fft_label", "F(Image)")
        if dpg.does_item_exist("product_fft_label"):
            dpg.set_value("product_fft_label", "F(Image) \u00d7 F(Kernel)")
    else:
        if dpg.does_item_exist("spatial_operator_drawing"):
            draw_deconvolution_symbol("spatial_operator_drawing")
        if dpg.does_item_exist("operator_label"):
            dpg.set_value("operator_label", "\u00f7")
        if dpg.does_item_exist("input_label"):
            dpg.set_value("input_label", "Blurred Input")
        if dpg.does_item_exist("result_label"):
            dpg.set_value("result_label", "Deconvolved Result")
        if dpg.does_item_exist("image_fft_label"):
            dpg.set_value("image_fft_label", "F(Blurred)")
        if dpg.does_item_exist("product_fft_label"):
            dpg.set_value("product_fft_label", "F(Blurred) \u00f7 F(Kernel)")


def regenerate_random_kernel():
    """Generate a new random kernel"""
    state.random_kernel = None


def update_kernel_type(sender, value):
    state.kernel_type = value
    if dpg.does_item_exist("randomize_button"):
        dpg.configure_item("randomize_button", show=(value == "Random"))


def update_kernel_size(sender, value):
    state.kernel_size = value if value % 2 == 1 else value + 1
    dpg.set_value(sender, state.kernel_size)


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Convolution Theorem Demo')
    parser.add_argument('--width', type=int, default=320, help='Camera width')
    parser.add_argument('--height', type=int, default=240, help='Camera height')
    args = parser.parse_args()

    # Initialize camera with optional resolution
    state.cap, _, _, state.use_camera = init_camera(width=args.width, height=args.height)

    if not state.use_camera:
        print("Warning: No camera found, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.cat_mode = True

    dpg.create_context()

    load_fonts()

    # Create a large font for operator symbols (=, ×, ÷)
    large_font = None
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            with dpg.font_registry():
                large_font = dpg.add_font(font_path, 48)
            break

    size = state.frame_size
    with dpg.texture_registry():
        blank_data = [0.0] * (size * size * 4)
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="input_texture")
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="kernel_texture")
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="result_texture")
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="image_fft_texture")
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="kernel_fft_texture")
        dpg.add_raw_texture(size, size, blank_data, format=dpg.mvFormat_Float_rgba, tag="product_fft_texture")

    with dpg.window(label="Convolution Theorem Demo", tag="main_window"):
        # Global controls row
        def _extra_reset():
            update_mode(None, DEFAULTS["mode"])

        add_global_controls(
            DEFAULTS, state,
            cat_mode_callback=make_state_updater(state, "cat_mode"),
            pause_callback=make_state_updater(state, "pause"),
            reset_extra=_extra_reset,
            guide=GUIDE_CONV_THEOREM, guide_title="Convolution Theorem",
        )

        dpg.add_separator()

        # Parameters
        with dpg.group(horizontal=True):
            # Mode selector
            with control_panel("Mode", width=160, height=160,
                               color=(255, 220, 100)):
                dpg.add_combo(
                    label="Mode", items=["Convolution", "Deconvolution"],
                    default_value=state.mode, callback=update_mode,
                    tag="mode_combo", width=120,
                )

            dpg.add_spacer(width=10)

            # Kernel
            with control_panel("Kernel", width=420, height=160,
                               color=(150, 200, 255)):
                dpg.add_combo(tag="kernel_type_combo", items=KERNELS, default_value=state.kernel_type,
                              callback=update_kernel_type, width=150, label="Type")
                with create_parameter_table():
                    add_parameter_row(
                        "Kernel Size", "kernel_size_slider", DEFAULTS["kernel_size"],
                        3, 51, update_kernel_size,
                        make_reset_callback(state, "kernel_size", "kernel_size_slider", DEFAULTS["kernel_size"]),
                        slider_type="int")
                    add_parameter_row(
                        "Gaussian Sigma", "gaussian_sigma_slider", DEFAULTS["gaussian_sigma"],
                        0.1, 15.0, make_state_updater(state, "gaussian_sigma"),
                        make_reset_callback(state, "gaussian_sigma", "gaussian_sigma_slider", DEFAULTS["gaussian_sigma"]),
                        format_str="%.1f")
                dpg.add_button(label="Regenerate Random Kernel", callback=regenerate_random_kernel,
                               tag="randomize_button", show=(state.kernel_type == "Random"))

            dpg.add_spacer(width=10)

            # Regularization (only visible in Deconvolution mode)
            with control_panel("Regularization", width=420, height=160,
                               tag="regularization_panel", color=(220, 180, 100)):
                dpg.add_checkbox(label="Enable", tag="use_regularization_checkbox",
                                 default_value=state.use_regularization,
                                 callback=make_state_updater(state, "use_regularization"))
                with create_parameter_table():
                    add_parameter_row(
                        "Reg. Value", "regularization_slider", DEFAULTS["regularization"],
                        0.0001, 0.5, make_state_updater(state, "regularization"),
                        make_reset_callback(state, "regularization", "regularization_slider", DEFAULTS["regularization"]),
                        format_str="%.4f")
            dpg.configure_item("regularization_panel", show=False)

        dpg.add_separator()

        # Spatial domain row
        dpg.add_text("Spatial Domain:", color=(100, 200, 255))
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Input Image", tag="input_label")
                dpg.add_image("input_texture", tag="input_image")
            with dpg.group():
                dpg.add_spacer(height=60)
                with dpg.drawlist(width=60, height=60, tag="spatial_operator_drawing"):
                    draw_convolution_symbol("spatial_operator_drawing")
            with dpg.group():
                dpg.add_text("Kernel")
                dpg.add_image("kernel_texture", tag="kernel_image")
            with dpg.group():
                dpg.add_spacer(height=70)
                with dpg.group():
                    eq_text = dpg.add_text("=", tag="spatial_equals", color=(255, 255, 100))
                    if large_font:
                        dpg.bind_item_font(eq_text, large_font)
            with dpg.group():
                dpg.add_text("Convolved Result", tag="result_label")
                dpg.add_image("result_texture", tag="result_image")

        dpg.add_spacer(height=10)
        dpg.add_separator()

        # Frequency domain row
        dpg.add_text("Frequency Domain:", color=(100, 200, 255))
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("F(Image)", tag="image_fft_label")
                dpg.add_image("image_fft_texture", tag="image_fft")
            with dpg.group():
                dpg.add_spacer(height=70)
                with dpg.group():
                    op_text2 = dpg.add_text("\u00d7", tag="operator_label", color=(255, 255, 100))
                    if large_font:
                        dpg.bind_item_font(op_text2, large_font)
            with dpg.group():
                dpg.add_text("F(Kernel)")
                dpg.add_image("kernel_fft_texture", tag="kernel_fft")
            with dpg.group():
                dpg.add_spacer(height=70)
                with dpg.group():
                    eq_text2 = dpg.add_text("=", tag="freq_equals", color=(255, 255, 100))
                    if large_font:
                        dpg.bind_item_font(eq_text2, large_font)
            with dpg.group():
                dpg.add_text("F(Image) \u00d7 F(Kernel)", tag="product_fft_label")
                dpg.add_image("product_fft_texture", tag="product_fft")

    # Setup viewport
    setup_viewport("CSCI 1430 - Convolution Theorem",
                   size * 3 + 350, size * 2 + 500,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])
    dpg.maximize_viewport()

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        if not state.pause:
            frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)
            if frame is not None:
                im = img_as_float(rgb2gray(frame))
                im = crop_to_square(im, state.frame_size)
            else:
                # Camera read failed - use random noise
                im = np.random.rand(state.frame_size, state.frame_size)

            # Create kernel (use cached random kernel for "Random" type)
            if state.kernel_type == "Random":
                kernel = create_random_kernel(state.kernel_size)
                kernel_padded = pad_kernel_to_image_size(kernel, im.shape)
            elif state.kernel_type == "Gaussian":
                kernel_padded = create_gaussian_kernel_fft(im.shape, state.gaussian_sigma)
                kernel = create_kernel(state.kernel_type, state.kernel_size, state.gaussian_sigma)
            else:
                kernel = create_kernel(state.kernel_type, state.kernel_size, state.gaussian_sigma)
                kernel_padded = pad_kernel_to_image_size(kernel, im.shape)

            if state.mode == "Convolution":
                result, image_fft, kernel_fft, product_fft = process_convolution(im, kernel_padded)

                state.blurred_image = result.copy()
                state.original_for_deconv = im.copy()

                input_display = im

            else:  # Deconvolution
                blurred, _, _, _ = process_convolution(im, kernel_padded)

                reg_value = state.regularization if state.use_regularization else 1e-14
                result, blurred_fft, kernel_fft, product_fft = process_deconvolution(
                    blurred, kernel_padded, reg_value
                )
                image_fft = blurred_fft
                input_display = blurred

            # Visualize FFTs
            image_fft_vis = visualize_fft_amplitude(image_fft, im.shape, use_log=True)
            kernel_fft_vis = visualize_fft_amplitude(np.fft.fft2(kernel_padded), im.shape, use_log=False)
            product_fft_vis = visualize_fft_amplitude(product_fft, im.shape, use_log=True)

            kernel_vis = visualize_kernel(kernel, state.frame_size)

            result_display = np.clip(result, 0, 1)

            # Update textures
            dpg.set_value("input_texture", convert_cv_to_dpg_float(input_display))
            dpg.set_value("kernel_texture", convert_cv_to_dpg_float(kernel_vis))
            dpg.set_value("result_texture", convert_cv_to_dpg_float(result_display))
            dpg.set_value("image_fft_texture", convert_cv_to_dpg_float(image_fft_vis))
            dpg.set_value("kernel_fft_texture", convert_cv_to_dpg_float(kernel_fft_vis))
            dpg.set_value("product_fft_texture", convert_cv_to_dpg_float(product_fft_vis))

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
