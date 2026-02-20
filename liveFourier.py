"""
Interactive Fourier transform demo with webcam
CSCI 1430 - Brown University
"""

import cv2
import os
import math
import numpy as np
import dearpygui.dearpygui as dpg
from skimage import img_as_float
from skimage.color import rgb2gray

from utils.demo_utils import init_camera, load_fallback_image, convert_cv_to_dpg_float, crop_to_square, get_frame
from utils.demo_ui import load_fonts, setup_viewport, make_state_updater, make_reset_callback

# Default values
DEFAULTS = {
    "amplitude_scalar": 1.0,
    "phase_offset": 0.0,
    "image_rotation": 0.0,
    "image_translate_x": 0.0,
    "image_translate_y": 0.0,
    "image_scale": 1.0,
    "intensity_scale": 1.0,
    "intensity_shift": 0.0,
    "dc_shift": 0.0,
    "dc_zero": False,
    "pause": False,
    "animate_magnitude": True,
    "animate_orientation": True,
    "ui_scale": 1.5,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class State:
    cap = None
    use_camera = True
    cat_mode = False
    frame_size = 240  # Square frame size for FFT processing
    display_size = 480  # Upscaled size for display (reduces aliasing)
    fallback_image = None

    # Display mode
    mode = 0  # 0=normal FFT, 1=DC only, 2=rotating dot, 3=frequency reconstruction

    # Controls
    amplitude_scalar = DEFAULTS["amplitude_scalar"]
    phase_offset = DEFAULTS["phase_offset"]
    image_rotation = DEFAULTS["image_rotation"]
    image_translate_x = DEFAULTS["image_translate_x"]
    image_translate_y = DEFAULTS["image_translate_y"]
    image_scale = DEFAULTS["image_scale"]
    intensity_scale = DEFAULTS["intensity_scale"]
    intensity_shift = DEFAULTS["intensity_shift"]
    dc_shift = DEFAULTS["dc_shift"]
    dc_zero = DEFAULTS["dc_zero"]
    pause = DEFAULTS["pause"]
    show_text = True

    # Animation for mode 2 (rotating dot)
    magnitude = 1
    orientation = 0.0
    animate_magnitude = DEFAULTS["animate_magnitude"]
    animate_orientation = DEFAULTS["animate_orientation"]
    magnitude_frame_counter = 0

    # Animation for mode 3 (frequency reconstruction)
    amplitude_cutoff = 1
    cutoff_direction = 1
    cutoff_frame_counter = 0


state = State()


def update_image_sizes():
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - 50
    available_height = vp_height - 230

    size = min(available_width // 2, available_height // 2)

    for tag in ["input_image", "inverse_image", "amplitude_image", "phase_image"]:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=size, height=size)


def on_viewport_resize():
    update_image_sizes()


def update_mode(sender, value):
    modes = {"Normal FFT": 0, "DC Only": 1, "Rotating Dot": 2, "Frequency Reconstruction": 3}
    state.mode = modes.get(value, 0)
    if dpg.does_item_exist("animation_panel"):
        dpg.configure_item("animation_panel", show=(state.mode == 2))


def transform_image(image, angle_degrees, translate_x, translate_y, scale):
    """Apply rotation, translation, and scale to image using affine transform"""
    h, w = image.shape[:2]

    # Pure translation: use np.roll for exact pixel shifts
    if angle_degrees == 0 and scale == 1.0:
        if translate_x != 0 or translate_y != 0:
            shift_x = int(translate_x * w / 100.0)
            shift_y = int(translate_y * h / 100.0)
            image = np.roll(image, shift_x, axis=1)
            image = np.roll(image, shift_y, axis=0)
        return image

    # Combined transform: use warpAffine with INTER_LINEAR
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    rotation_matrix[0, 2] += translate_x * w / 100.0
    rotation_matrix[1, 2] += translate_y * h / 100.0
    return cv2.warpAffine(image, rotation_matrix, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def process_fft(im):
    """Process FFT based on current mode"""
    height, width = im.shape
    amplitude = np.zeros(im.shape)
    phase = np.zeros(im.shape)

    if state.mode == 0:  # Normal FFT
        imFFT = np.fft.fft2(im)
        amplitude = np.sqrt(np.power(imFFT.real, 2) + np.power(imFFT.imag, 2))
        phase = np.arctan2(imFFT.imag, imFFT.real)

    elif state.mode == 1:  # DC Only
        amplitude[0, 0] = im.shape[0] * im.shape[1] / 2.0

    elif state.mode == 2:  # Rotating dot
        if state.animate_orientation:
            state.orientation += math.pi / 600.0
            if state.orientation > math.pi * 2:
                state.orientation = 0

        if state.animate_magnitude:
            state.magnitude_frame_counter += 1
            if state.magnitude_frame_counter >= 10:
                state.magnitude_frame_counter = 0
                state.magnitude += 1
                if state.magnitude >= 50:
                    state.magnitude = 1

        cx, cy = width // 2, height // 2
        xd = state.magnitude * math.cos(state.orientation)
        yd = state.magnitude * math.sin(state.orientation)

        a = np.zeros(im.shape)
        a[int(cy + yd), int(cx + xd)] = (im.shape[0] * im.shape[1] / 2.0) * state.amplitude_scalar
        amplitude = np.fft.fftshift(a)
        amplitude[0, 0] = im.shape[0] * im.shape[1] / 2.0

        p = np.zeros(im.shape)
        p[int(cy + yd), int(cx + xd)] = state.phase_offset
        phase = np.fft.fftshift(p)

    elif state.mode == 3:  # Frequency reconstruction
        imFFT = np.fft.fft2(im)
        amplitude = np.sqrt(np.power(imFFT.real, 2) + np.power(imFFT.imag, 2))
        phase = np.arctan2(imFFT.imag, imFFT.real)

        Y, X = np.ogrid[:height, :width]
        mask = np.logical_or(np.abs(X - width/2) >= state.amplitude_cutoff,
                           np.abs(Y - height/2) >= state.amplitude_cutoff)
        a = np.fft.fftshift(amplitude)
        a[mask] = 0
        amplitude = np.fft.fftshift(a)

        state.cutoff_frame_counter += 1
        if state.cutoff_frame_counter >= 10:
            state.cutoff_frame_counter = 0
            if state.amplitude_cutoff <= 1 and state.cutoff_direction < 0:
                state.cutoff_direction *= -1
            if state.amplitude_cutoff > width/3 and state.cutoff_direction > 0:
                state.cutoff_direction *= -1
            state.amplitude_cutoff += state.cutoff_direction

    if state.dc_zero:
        amplitude[0, 0] = 0

    amplitude[0, 0] += state.dc_shift * height * width
    amplitude[0, 0] = max(amplitude[0, 0], 0)

    dc_value = amplitude[0, 0]
    amplitude = amplitude * state.amplitude_scalar
    amplitude[0, 0] = dc_value
    dc_phase = phase[0, 0]
    phase = phase + state.phase_offset
    phase[0, 0] = dc_phase

    recReal = np.cos(phase) * amplitude
    recImag = np.sin(phase) * amplitude
    rec = recReal + 1j * recImag
    newImage = np.fft.ifft2(rec).real

    amplitude[amplitude == 0] = np.finfo(float).eps
    amplitude_vis = np.log(np.fft.fftshift(amplitude))
    phase_vis = np.fft.fftshift(phase)

    min_log_amplitude = 0.0
    max_log_amplitude = 0.9 * np.log(height * width)
    amplitude_vis = (amplitude_vis - min_log_amplitude) / (max_log_amplitude - min_log_amplitude)
    amplitude_vis = np.clip(amplitude_vis, 0, 1)

    phase_vis = np.angle(np.exp(1j * phase_vis))
    phase_vis = (phase_vis + np.pi) / (2 * np.pi)

    return newImage, amplitude_vis, phase_vis


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fourier Transform Demo')
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

    size = state.frame_size
    dsize = state.display_size

    dpg.create_context()

    load_fonts()

    with dpg.texture_registry():
        blank_data = [0.0] * (dsize * dsize * 4)
        dpg.add_raw_texture(dsize, dsize, blank_data, format=dpg.mvFormat_Float_rgba, tag="input_texture")
        dpg.add_raw_texture(dsize, dsize, blank_data, format=dpg.mvFormat_Float_rgba, tag="inverse_texture")
        dpg.add_raw_texture(dsize, dsize, blank_data, format=dpg.mvFormat_Float_rgba, tag="amplitude_texture")
        dpg.add_raw_texture(dsize, dsize, blank_data, format=dpg.mvFormat_Float_rgba, tag="phase_texture")

    with dpg.window(label="Fourier Transform Demo", tag="main_window"):
        # Global controls row
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="Mode",
                items=["Normal FFT", "DC Only", "Rotating Dot", "Frequency Reconstruction"],
                default_value="Normal FFT",
                callback=update_mode,
                tag="mode_combo",
                width=160
            )
            dpg.add_combo(
                label="UI Scale",
                items=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"],
                default_value=str(DEFAULTS["ui_scale"]),
                callback=lambda s, v: dpg.set_global_font_scale(float(v)),
                width=80
            )
            dpg.add_spacer(width=20)
            dpg.add_checkbox(label="Pause", default_value=state.pause,
                           callback=make_state_updater(state, "pause"))
            dpg.add_checkbox(
                label="Cat Mode",
                default_value=state.cat_mode,
                callback=make_state_updater(state, "cat_mode"),
                tag="cat_mode_checkbox",
                enabled=state.use_camera
            )
            if not state.use_camera:
                dpg.add_text("(no webcam)", color=(255, 100, 100))

        dpg.add_separator()

        # Parameters using child_window containers
        with dpg.group(horizontal=True):
            # Column 1: Input Intensity
            with dpg.child_window(width=300, height=130, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Input Intensity [0,1]", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=70)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("Shift I")
                            dpg.add_slider_float(tag="intensity_shift_slider", default_value=state.intensity_shift,
                                                 min_value=-1.0, max_value=1.0,
                                                 callback=make_state_updater(state, "intensity_shift"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "intensity_shift", "intensity_shift_slider", DEFAULTS["intensity_shift"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Scale I")
                            dpg.add_slider_float(tag="intensity_scale_slider", default_value=state.intensity_scale,
                                                 min_value=0.0, max_value=3.0,
                                                 callback=make_state_updater(state, "intensity_scale"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "intensity_scale", "intensity_scale_slider", DEFAULTS["intensity_scale"]),
                                           width=25)

            dpg.add_spacer(width=10)

            # Column 2: Input Transforms
            with dpg.child_window(width=320, height=160, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Input 2D Transforms", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=130)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("Rotate x,y")
                            dpg.add_slider_float(tag="rotation_slider", default_value=state.image_rotation,
                                                 min_value=-180.0, max_value=180.0,
                                                 callback=make_state_updater(state, "image_rotation"),
                                                 width=60, format="%.1f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "image_rotation", "rotation_slider", DEFAULTS["image_rotation"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Scale x,y")
                            dpg.add_slider_float(tag="scale_slider", default_value=state.image_scale,
                                                 min_value=0.25, max_value=4.0,
                                                 callback=make_state_updater(state, "image_scale"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "image_scale", "scale_slider", DEFAULTS["image_scale"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Translate x")
                            dpg.add_slider_float(tag="translate_x_slider", default_value=state.image_translate_x,
                                                 min_value=-100.0, max_value=100.0,
                                                 callback=make_state_updater(state, "image_translate_x"),
                                                 width=60, format="%.1f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "image_translate_x", "translate_x_slider", DEFAULTS["image_translate_x"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Translate y")
                            dpg.add_slider_float(tag="translate_y_slider", default_value=state.image_translate_y,
                                                 min_value=-100.0, max_value=100.0,
                                                 callback=make_state_updater(state, "image_translate_y"),
                                                 width=60, format="%.1f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "image_translate_y", "translate_y_slider", DEFAULTS["image_translate_y"]),
                                           width=25)

            dpg.add_spacer(width=10)

            # Column 3: Fourier Parameters
            with dpg.child_window(width=460, height=160, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Fourier Parameters", default_open=True):
                    with dpg.table(header_row=False,
                                   borders_innerV=False, borders_outerV=False,
                                   borders_innerH=False, borders_outerH=False,
                                   policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=240)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

                        with dpg.table_row():
                            dpg.add_text("DC Shift")
                            dpg.add_slider_float(tag="dc_shift_slider", default_value=state.dc_shift,
                                                 min_value=-1.0, max_value=1.0,
                                                 callback=make_state_updater(state, "dc_shift"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "dc_shift", "dc_shift_slider", DEFAULTS["dc_shift"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Amplitude Scale (no DC)")
                            dpg.add_slider_float(tag="amplitude_slider", default_value=state.amplitude_scalar,
                                                 min_value=0.1, max_value=5.0,
                                                 callback=make_state_updater(state, "amplitude_scalar"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "amplitude_scalar", "amplitude_slider", DEFAULTS["amplitude_scalar"]),
                                           width=25)

                        with dpg.table_row():
                            dpg.add_text("Phase Offset")
                            dpg.add_slider_float(tag="phase_slider", default_value=state.phase_offset,
                                                 min_value=-3.14, max_value=3.14,
                                                 callback=make_state_updater(state, "phase_offset"),
                                                 width=60, format="%.2f")
                            dpg.add_button(label="R",
                                           callback=make_reset_callback(state, "phase_offset", "phase_slider", DEFAULTS["phase_offset"]),
                                           width=25)

                    dpg.add_checkbox(label="Zero DC", default_value=state.dc_zero,
                                   callback=make_state_updater(state, "dc_zero"))

            dpg.add_spacer(width=10)

            # Column 4: Animation (only visible in Rotating Dot mode)
            with dpg.child_window(width=200, height=130, border=False, no_scrollbar=True,
                                  tag="animation_panel", show=False):
                with dpg.collapsing_header(label="Animation", default_open=True):
                    dpg.add_checkbox(label="Animate Magnitude", default_value=state.animate_magnitude,
                                   callback=make_state_updater(state, "animate_magnitude"))
                    dpg.add_checkbox(label="Animate Orientation", default_value=state.animate_orientation,
                                   callback=make_state_updater(state, "animate_orientation"))

        dpg.add_separator()

        # 2x2 grid of images
        with dpg.group():
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Input")
                    dpg.add_image("input_texture", tag="input_image")
                with dpg.group():
                    dpg.add_text("Inverse FT")
                    dpg.add_image("inverse_texture", tag="inverse_image")

            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Fourier Amplitude")
                    dpg.add_image("amplitude_texture", tag="amplitude_image")
                with dpg.group():
                    dpg.add_text("Fourier Phase")
                    dpg.add_image("phase_texture", tag="phase_image")

    # Setup viewport
    setup_viewport("CSCI 1430 - Fourier Transform",
                   max(size * 2 + 100, 650), size * 2 + 280,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])
    dpg.maximize_viewport()

    update_image_sizes()

    # Main loop
    while dpg.is_dearpygui_running():
        if not state.pause:
            frame = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)
            if frame is not None:
                im = img_as_float(rgb2gray(frame))
                im = crop_to_square(im, state.frame_size)
            else:
                # Camera read failed - use random noise
                im = np.random.rand(state.frame_size, state.frame_size)

            if (state.image_rotation != 0 or state.image_translate_x != 0 or
                state.image_translate_y != 0 or state.image_scale != 1.0):
                im = transform_image(im.astype(np.float32), state.image_rotation,
                                    state.image_translate_x, state.image_translate_y,
                                    state.image_scale)

            if state.intensity_scale != 1.0 or state.intensity_shift != 0.0:
                im = np.clip(im * state.intensity_scale + state.intensity_shift, 0, 1)

            inverse, amplitude, phase = process_fft(im)

            blank_input = state.mode in [1, 2]
            ds = (state.display_size, state.display_size)
            up = lambda img: cv2.resize(img.astype(np.float32), ds, interpolation=cv2.INTER_CUBIC)
            dpg.set_value("input_texture", convert_cv_to_dpg_float(up(np.zeros_like(im) if blank_input else im)))
            dpg.set_value("inverse_texture", convert_cv_to_dpg_float(up(np.clip(inverse, 0, 1))))
            dpg.set_value("amplitude_texture", convert_cv_to_dpg_float(up(amplitude)))
            dpg.set_value("phase_texture", convert_cv_to_dpg_float(up(phase)))

        dpg.render_dearpygui_frame()

    if state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
