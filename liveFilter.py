"""
Live Filter Demo with Interactive Kernel Editor
CSCI 1430 - Brown University

Features:
- Pre-defined kernel presets (Box, Gaussian, Sharpen, Sobel, Laplacian, Emboss)
- Visual kernel editor with colored cells and value overlays
- Direct click modification: left-click = increase, right-click = decrease
"""

import cv2
import os
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_utils import convert_cv_to_dpg, init_camera, load_fallback_image, get_frame
from utils.demo_ui import load_fonts, setup_viewport, make_state_updater
from utils.demo_kernels import KERNEL_PRESETS as _KERNEL_PRESETS, SIGMA_KERNELS, ZERO_DC_KERNELS, create_kernel, resize_kernel

# Default values
DEFAULTS = {
    "kernel_size": 3,
    "kernel_type": "Box Blur",
    "normalize_kernel": True,
    "gaussian_sigma": 1.0,
    "show_original": True,
    "ui_scale": 1.5,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Available kernel presets (add "Custom" for interactive editor)
KERNEL_PRESETS = _KERNEL_PRESETS + ["Custom"]


class State:
    cap = None
    frame_width = 0
    frame_height = 0
    input_ratio = 0.4
    use_camera = True
    cat_mode = False
    fallback_image = None
    show_original = DEFAULTS["show_original"]

    # Kernel state
    kernel_size = DEFAULTS["kernel_size"]
    kernel_type = DEFAULTS["kernel_type"]
    kernel_values = None  # np.array of actual kernel values
    normalize_kernel = DEFAULTS["normalize_kernel"]
    gaussian_sigma = DEFAULTS["gaussian_sigma"]

    # Kernel editor state
    kernel_editor_size = 300  # pixels for the drawlist
    hovered_cell = None  # (row, col) or None


state = State()


def create_kernel_preset(preset_name, size, sigma=1.0):
    """Create a kernel based on preset name.

    Wraps the shared create_kernel function but handles "Custom" specially
    to preserve user edits in the interactive editor.
    """
    size = size if size % 2 == 1 else size + 1

    if preset_name == "Custom":
        # Keep current kernel if it exists and is the right size
        if state.kernel_values is not None and state.kernel_values.shape[0] == size:
            return state.kernel_values.copy()
        # Otherwise create identity
        kernel = np.zeros((size, size), dtype=np.float64)
        kernel[size // 2, size // 2] = 1.0
        return kernel

    # Use the shared kernel creation function
    return create_kernel(preset_name, size, sigma)


def value_to_color(value, min_val, max_val):
    """Convert kernel value to grayscale intensity.

    If all values are non-negative: black (0) to white (max)
    If there are negative values: mid-gray = 0, black = min, white = max
    """
    if max_val == min_val:
        return (128, 128, 128)

    has_negative = min_val < 0

    if has_negative:
        # Scale so that 0 maps to mid-gray (128)
        # Negative values go toward black, positive toward white
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0:
            intensity = 128
        else:
            # Map value from [-abs_max, abs_max] to [0, 255]
            normalized = (value + abs_max) / (2 * abs_max)
            intensity = int(normalized * 255)
    else:
        # All non-negative: simple 0=black to max=white
        if max_val == 0:
            intensity = 0
        else:
            intensity = int((value / max_val) * 255)

    intensity = max(0, min(255, intensity))
    return (intensity, intensity, intensity)


def draw_kernel_editor():
    """Draw the kernel grid with colored cells and value text."""
    if state.kernel_values is None:
        return

    dpg.delete_item("kernel_drawlist", children_only=True)

    size = state.kernel_values.shape[0]
    editor_size = state.kernel_editor_size
    cell_size = editor_size / size
    padding = 2

    min_val = state.kernel_values.min()
    max_val = state.kernel_values.max()

    for row in range(size):
        for col in range(size):
            value = state.kernel_values[row, col]
            color = value_to_color(value, min_val, max_val)

            x1 = col * cell_size + padding
            y1 = row * cell_size + padding
            x2 = (col + 1) * cell_size - padding
            y2 = (row + 1) * cell_size - padding

            # Draw cell background
            dpg.draw_rectangle(
                (x1, y1), (x2, y2),
                color=(100, 100, 100, 255),
                fill=color + (255,),
                parent="kernel_drawlist"
            )

            # Highlight hovered cell
            if state.hovered_cell == (row, col):
                dpg.draw_rectangle(
                    (x1, y1), (x2, y2),
                    color=(255, 255, 0, 255),
                    thickness=3,
                    parent="kernel_drawlist"
                )

            # Draw value text
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Format value - always use fixed decimal, never scientific notation
            text = f"{value:.3f}"

            # Text color based on background brightness
            brightness = (color[0] + color[1] + color[2]) / 3
            text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)

            # Scale font size to fill cell (leave some padding)
            # Text width is approximately 0.6 * font_size * num_chars
            # Text height is approximately font_size
            text_len = len(text)
            cell_inner = cell_size - 2 * padding

            # Calculate max font size that fits width and height
            max_font_by_width = cell_inner / (0.6 * text_len) if text_len > 0 else cell_inner
            max_font_by_height = cell_inner * 0.7  # Leave vertical padding
            font_size = min(max_font_by_width, max_font_by_height)

            # Clamp to reasonable range
            font_size = max(6, min(24, int(font_size)))

            if font_size >= 6:
                # Center text in cell
                text_width = 0.6 * font_size * text_len
                text_height = font_size
                tx = cx - text_width / 2
                ty = cy - text_height / 2
                dpg.draw_text((tx, ty), text,
                              color=text_color, size=font_size, parent="kernel_drawlist")


def get_cell_from_mouse(mouse_pos, drawlist_pos):
    """Convert mouse position to cell (row, col)."""
    if state.kernel_values is None:
        return None

    size = state.kernel_values.shape[0]
    cell_size = state.kernel_editor_size / size

    # Get relative position within drawlist
    rel_x = mouse_pos[0] - drawlist_pos[0]
    rel_y = mouse_pos[1] - drawlist_pos[1]

    if rel_x < 0 or rel_y < 0:
        return None
    if rel_x >= state.kernel_editor_size or rel_y >= state.kernel_editor_size:
        return None

    col = int(rel_x / cell_size)
    row = int(rel_y / cell_size)

    if 0 <= row < size and 0 <= col < size:
        return (row, col)
    return None


def modify_cell(row, col, delta):
    """Modify a cell value by delta."""
    if state.kernel_values is None:
        return

    state.kernel_values[row, col] += delta

    # Switch to Custom mode when editing
    if state.kernel_type != "Custom":
        state.kernel_type = "Custom"
        if dpg.does_item_exist("kernel_combo"):
            dpg.set_value("kernel_combo", "Custom")

    draw_kernel_editor()


def update_kernel_size(sender, value):
    """Handle kernel size change."""
    new_size = value if value % 2 == 1 else value + 1
    state.kernel_size = new_size
    dpg.set_value(sender, new_size)

    # For Custom kernels, preserve existing values by resizing
    if state.kernel_type == "Custom" and state.kernel_values is not None:
        state.kernel_values = resize_kernel(state.kernel_values, new_size)
    else:
        # Recreate kernel with new size
        state.kernel_values = create_kernel_preset(state.kernel_type, new_size, state.gaussian_sigma)
    draw_kernel_editor()


def update_kernel_type(sender, value):
    """Handle kernel type change."""
    state.kernel_type = value
    state.kernel_values = create_kernel_preset(value, state.kernel_size, state.gaussian_sigma)
    draw_kernel_editor()

    # Show/hide sigma slider based on kernel type
    if dpg.does_item_exist("sigma_slider"):
        dpg.configure_item("sigma_slider", show=(value in SIGMA_KERNELS))


def update_gaussian_sigma(sender, value):
    """Handle sigma change for Gaussian/LoG kernels."""
    state.gaussian_sigma = value
    if state.kernel_type in SIGMA_KERNELS:
        state.kernel_values = create_kernel_preset(state.kernel_type, state.kernel_size, value)
        draw_kernel_editor()


def update_image_sizes():
    """Update image display sizes based on viewport dimensions."""
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    available_width = vp_width - state.kernel_editor_size - 100
    available_height = vp_height - 280

    aspect_ratio = state.frame_width / state.frame_height if state.frame_height > 0 else 1.33

    filtered_width = int(available_width / (1 + state.input_ratio))
    filtered_height = int(filtered_width / aspect_ratio)

    if filtered_height > available_height:
        filtered_height = available_height
        filtered_width = int(filtered_height * aspect_ratio)

    input_width = int(filtered_width * state.input_ratio)
    input_height = int(filtered_height * state.input_ratio)

    if dpg.does_item_exist("input_image"):
        dpg.configure_item("input_image", width=input_width, height=input_height)
    if dpg.does_item_exist("filtered_image"):
        dpg.configure_item("filtered_image", width=filtered_width, height=filtered_height)


def on_viewport_resize():
    update_image_sizes()


def process_frame():
    """Capture and process a single frame."""
    img = get_frame(state.cap, state.fallback_image, state.use_camera, state.cat_mode)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the current kernel
    if state.kernel_values is not None:
        kernel = state.kernel_values.copy()
        # Skip normalization for zero-DC kernels (edge detectors, LoG, Laplacian)
        if state.normalize_kernel and state.kernel_type not in ZERO_DC_KERNELS and kernel.sum() != 0:
            kernel = kernel / kernel.sum()
        kernel = kernel.astype(np.float32)
        filtered = cv2.filter2D(gray, -1, kernel)
    else:
        filtered = gray

    return gray, filtered


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Image Filtering Demo')
    parser.add_argument('--width', type=int, default=None, help='Camera width')
    parser.add_argument('--height', type=int, default=None, help='Camera height')
    args = parser.parse_args()

    # Initialize camera with optional resolution
    state.cap, state.frame_width, state.frame_height, state.use_camera = \
        init_camera(width=args.width, height=args.height)

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    # Load fallback image
    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.frame_height, state.frame_width = state.fallback_image.shape[:2]
        state.cat_mode = True

    # Initialize kernel
    state.kernel_values = create_kernel_preset(state.kernel_type, state.kernel_size, state.gaussian_sigma)

    frame_width, frame_height = state.frame_width, state.frame_height

    dpg.create_context()

    load_fonts()

    with dpg.texture_registry():
        blank_data = [0.0] * (frame_width * frame_height * 4)
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                            format=dpg.mvFormat_Float_rgba, tag="input_texture")
        dpg.add_raw_texture(frame_width, frame_height, blank_data,
                            format=dpg.mvFormat_Float_rgba, tag="filtered_texture")

    with dpg.window(label="Interactive Filter Demo", tag="main_window"):
        # Top row controls
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="Kernel",
                items=KERNEL_PRESETS,
                default_value=state.kernel_type,
                callback=update_kernel_type,
                tag="kernel_combo",
                width=120
            )
            dpg.add_slider_int(
                label="Size",
                default_value=state.kernel_size,
                min_value=3, max_value=15,
                callback=update_kernel_size,
                tag="size_slider",
                width=100
            )
            dpg.add_slider_float(
                label="Sigma",
                default_value=state.gaussian_sigma,
                min_value=0.1, max_value=5.0,
                callback=update_gaussian_sigma,
                tag="sigma_slider",
                width=80,
                show=(state.kernel_type in SIGMA_KERNELS)
            )
            dpg.add_checkbox(
                label="Normalize",
                default_value=state.normalize_kernel,
                callback=make_state_updater(state, "normalize_kernel")
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

        dpg.add_separator()
        dpg.add_text("Left-click: +0.1  |  Right-click: -0.1  |  Shift+click: +/-1.0  |  Ctrl+click: set to 0",
                     color=(150, 150, 150))
        dpg.add_text("", tag="status_text")
        dpg.add_separator()

        # Main content: kernel editor + images
        with dpg.group(horizontal=True):
            # Kernel editor panel
            with dpg.group():
                dpg.add_text("Kernel Editor")
                with dpg.drawlist(width=state.kernel_editor_size,
                                  height=state.kernel_editor_size,
                                  tag="kernel_drawlist"):
                    pass  # Will be populated by draw_kernel_editor()

                # Kernel sum display
                dpg.add_text("", tag="kernel_sum_text")

            dpg.add_spacer(width=20)

            # Images panel
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Input Stream")
                        dpg.add_image("input_texture", tag="input_image")

                    with dpg.group():
                        dpg.add_text("Filtered")
                        dpg.add_image("filtered_texture", tag="filtered_image")

    # Setup viewport
    initial_width = int(frame_width * (1 + state.input_ratio) + state.kernel_editor_size + 150)
    setup_viewport("CSCI 1430 - Interactive Filter Demo",
                   initial_width, frame_height + 350,
                   "main_window", on_viewport_resize, DEFAULTS["ui_scale"])

    update_image_sizes()
    draw_kernel_editor()

    # Main loop
    while dpg.is_dearpygui_running():
        # Handle mouse interaction with kernel editor
        if dpg.does_item_exist("kernel_drawlist"):
            mouse_pos = dpg.get_mouse_pos(local=False)
            # Use get_item_rect_min for actual screen coordinates
            drawlist_pos = dpg.get_item_rect_min("kernel_drawlist")

            if drawlist_pos[0] >= 0 and drawlist_pos[1] >= 0:
                cell = get_cell_from_mouse(mouse_pos, drawlist_pos)

                # Update hover state
                if cell != state.hovered_cell:
                    state.hovered_cell = cell
                    draw_kernel_editor()

                # Handle clicks
                if cell is not None:
                    shift_held = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
                    ctrl_held = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

                    if dpg.is_mouse_button_clicked(dpg.mvMouseButton_Left):
                        if ctrl_held:
                            state.kernel_values[cell[0], cell[1]] = 0
                            if state.kernel_type != "Custom":
                                state.kernel_type = "Custom"
                                dpg.set_value("kernel_combo", "Custom")
                            draw_kernel_editor()
                        else:
                            delta = 1.0 if shift_held else 0.1
                            modify_cell(cell[0], cell[1], delta)

                    elif dpg.is_mouse_button_clicked(dpg.mvMouseButton_Right):
                        if ctrl_held:
                            state.kernel_values[cell[0], cell[1]] = 0
                            if state.kernel_type != "Custom":
                                state.kernel_type = "Custom"
                                dpg.set_value("kernel_combo", "Custom")
                            draw_kernel_editor()
                        else:
                            delta = -1.0 if shift_held else -0.1
                            modify_cell(cell[0], cell[1], delta)

        # Process and display frame
        gray, filtered = process_frame()

        if gray is not None and filtered is not None:
            dpg.set_value("input_texture", convert_cv_to_dpg(gray))
            dpg.set_value("filtered_texture", convert_cv_to_dpg(filtered))

            kernel_sum = state.kernel_values.sum() if state.kernel_values is not None else 0
            status = f"Kernel: {state.kernel_type} ({state.kernel_size}x{state.kernel_size})"
            if state.hovered_cell:
                row, col = state.hovered_cell
                val = state.kernel_values[row, col]
                status += f"  |  Cell ({row},{col}): {val:.6f}"
            dpg.set_value("status_text", status)

            sum_text = f"Sum: {kernel_sum:.4f}"
            if state.normalize_kernel:
                sum_text += "\n(normalized to 1.0)"
            dpg.set_value("kernel_sum_text", sum_text)

        dpg.render_dearpygui_frame()

    if state.use_camera and state.cap is not None:
        state.cap.release()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
