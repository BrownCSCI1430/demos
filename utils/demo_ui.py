"""
Shared Dear PyGui UI utilities for CSCI 1430 computer vision demos.
Contains callback factories, UI component builders, and viewport setup.
"""

import dearpygui.dearpygui as dpg


# =============================================================================
# Callback Factories
# =============================================================================

def make_ui_scale_callback():
    """Create UI scale callback.

    Returns:
        Callback function that updates global font scale
    """
    def callback(sender, value):
        dpg.set_global_font_scale(value)
    return callback


def make_state_updater(state, attr):
    """Factory for simple state update callbacks.

    Args:
        state: State object to update
        attr: Attribute name to update on state

    Returns:
        Callback function that sets state.attr = value
    """
    def callback(sender, value):
        setattr(state, attr, value)
    return callback


def make_reset_callback(state, attr, slider_tag, default_value):
    """Factory for reset button callbacks.

    Args:
        state: State object to update
        attr: Attribute name to reset
        slider_tag: Tag of slider widget to update
        default_value: Value to reset to

    Returns:
        Callback function that resets state and slider
    """
    def callback():
        setattr(state, attr, default_value)
        dpg.set_value(slider_tag, default_value)
    return callback


# =============================================================================
# UI Component Builders
# =============================================================================

def add_global_controls(defaults, state, cat_mode_callback, extra_controls_before=None, extra_controls_after=None):
    """Add standard global controls row (UI Scale, Cat Mode, etc.).

    Args:
        defaults: Dictionary containing default values (must have "ui_scale")
        state: State object with use_camera and cat_mode attributes
        cat_mode_callback: Callback for cat mode checkbox
        extra_controls_before: Optional callable to add controls before Cat Mode
        extra_controls_after: Optional callable to add controls after Cat Mode

    Creates a horizontal group with:
    - UI Scale slider
    - Optional extra controls (before)
    - Cat Mode checkbox
    - Optional extra controls (after)
    - "(no webcam)" text if camera unavailable
    """
    with dpg.group(horizontal=True):
        dpg.add_slider_float(
            label="UI Scale",
            default_value=defaults.get("ui_scale", 1.5),
            min_value=1.0,
            max_value=3.0,
            callback=make_ui_scale_callback(),
            width=100
        )
        dpg.add_spacer(width=20)

        if extra_controls_before:
            extra_controls_before()

        dpg.add_checkbox(
            label="Cat Mode",
            default_value=state.cat_mode,
            callback=cat_mode_callback,
            tag="cat_mode_checkbox",
            enabled=state.use_camera
        )

        if extra_controls_after:
            extra_controls_after()

        if not state.use_camera:
            dpg.add_text("(no webcam)", color=(255, 100, 100))


def add_parameter_row(label, tag, default, min_val, max_val, callback,
                      reset_callback, slider_type="float", width=80, format_str=None):
    """Add a parameter row with label, slider, and reset button.

    Must be called within a dpg.table context.

    Args:
        label: Text label for the parameter
        tag: Unique tag for the slider widget
        default: Default value for the slider
        min_val: Minimum slider value
        max_val: Maximum slider value
        callback: Callback function when slider changes
        reset_callback: Callback function for reset button
        slider_type: "float" or "int"
        width: Slider width in pixels
        format_str: Optional format string for float sliders (e.g., "%.2f")
    """
    with dpg.table_row():
        dpg.add_text(label)
        if slider_type == "float":
            kwargs = {"format": format_str} if format_str else {}
            dpg.add_slider_float(
                tag=tag,
                default_value=default,
                min_value=min_val,
                max_value=max_val,
                callback=callback,
                width=width,
                **kwargs
            )
        else:
            dpg.add_slider_int(
                tag=tag,
                default_value=default,
                min_value=min_val,
                max_value=max_val,
                callback=callback,
                width=width
            )
        dpg.add_button(label="R", callback=reset_callback, width=25)


def add_parameter_spacer_row():
    """Add an empty spacer row in a parameter table.

    Must be called within a dpg.table context.
    Useful for filling out 2-column layouts.
    """
    with dpg.table_row():
        dpg.add_spacer(width=80)
        dpg.add_spacer(width=100)
        dpg.add_spacer(width=30)


def create_parameter_table():
    """Create a borderless parameter table for controls.

    Returns:
        dpg.table context manager

    Usage:
        with create_parameter_table():
            dpg.add_table_column(width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=30)
            # Add rows...
    """
    return dpg.table(
        header_row=False,
        borders_innerV=False,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
        policy=dpg.mvTable_SizingFixedFit
    )


def setup_viewport(title, width, height, main_window_tag, resize_callback, ui_scale=1.5):
    """Setup viewport with resize handler.

    Args:
        title: Viewport window title
        width: Initial viewport width
        height: Initial viewport height
        main_window_tag: Tag of the main window to set as primary
        resize_callback: Callback function for viewport resize events
        ui_scale: Initial UI scale factor
    """
    with dpg.item_handler_registry(tag="viewport_handler"):
        dpg.add_item_resize_handler(callback=resize_callback)

    dpg.create_viewport(title=title, width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(main_window_tag, True)
    dpg.bind_item_handler_registry(main_window_tag, "viewport_handler")
    dpg.set_global_font_scale(ui_scale)


def create_texture(width, height, tag):
    """Create a blank RGBA texture.

    Must be called within a dpg.texture_registry context.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        tag: Unique tag for the texture
    """
    blank_data = [0.0] * (width * height * 4)
    dpg.add_raw_texture(
        width, height, blank_data,
        format=dpg.mvFormat_Float_rgba,
        tag=tag
    )


def add_status_section():
    """Add a status text section with separators.

    Returns the tag "status_text" for updating.
    """
    dpg.add_separator()
    dpg.add_text("", tag="status_text")
    dpg.add_separator()


def add_image_pair(label1, texture1, image_tag1, label2, texture2, image_tag2):
    """Add a horizontal pair of labeled images.

    Args:
        label1: Label for first image
        texture1: Texture tag for first image
        image_tag1: Tag for first image widget
        label2: Label for second image
        texture2: Texture tag for second image
        image_tag2: Tag for second image widget
    """
    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text(label1)
            dpg.add_image(texture1, tag=image_tag1)
        dpg.add_spacer(width=10)
        with dpg.group():
            dpg.add_text(label2)
            dpg.add_image(texture2, tag=image_tag2)


def create_dual_parameter_table():
    """Create a 7-column table for dual-column parameter layouts.

    Layout: Label1 | Slider1 | Reset1 | Spacer | Label2 | Slider2 | Reset2

    Returns:
        dpg.table context manager
    """
    return dpg.table(
        header_row=False,
        borders_innerV=False,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
        policy=dpg.mvTable_SizingFixedFit
    )


def add_dual_table_columns(label_width=80, slider_width=100, reset_width=30, spacer_width=20):
    """Add columns for a dual-parameter table.

    Must be called immediately after creating the table with create_dual_parameter_table().

    Args:
        label_width: Width of label columns
        slider_width: Width of slider columns
        reset_width: Width of reset button columns
        spacer_width: Width of spacer column between the two parameter sets
    """
    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=slider_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=reset_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=spacer_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=slider_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=reset_width)
