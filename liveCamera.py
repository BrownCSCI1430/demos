"""
Live 3D Camera Demo with Dear PyGui controls
CSCI 1430 - Brown University

Interactive exploration of intrinsic (K) and extrinsic [R|t] camera matrices.
Two synchronized panels: camera view and overhead overview with frustum wireframe.
Toggle between world and camera reference frame interpretations.
"""

import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, build_rotation, build_extrinsic, euler_from_rotation,
    fov_to_focal, make_lookat_Rt,
    render_scene, create_default_scene,
    make_frustum_mesh, make_axis_mesh, make_camera_axes_mesh,
    format_matrix,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
)


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS = {
    "rot_x": 0.0,
    "rot_y": 0.0,
    "rot_z": 0.0,
    "trans_x": 0.0,
    "trans_y": 0.0,
    "trans_z": 5.0,
    "fov_x": 60.0,
    "fov_y": 60.0,
    "skew": 0.0,
    "cx_offset": 0.0,
    "cy_offset": 0.0,
    "ui_scale": 1.5,
}

RENDER_W = 400
RENDER_H = 400


# =============================================================================
# State
# =============================================================================

class State:
    # Extrinsic
    rot_x = DEFAULTS["rot_x"]
    rot_y = DEFAULTS["rot_y"]
    rot_z = DEFAULTS["rot_z"]
    trans_x = DEFAULTS["trans_x"]
    trans_y = DEFAULTS["trans_y"]
    trans_z = DEFAULTS["trans_z"]

    # Intrinsic
    fov_x = DEFAULTS["fov_x"]
    fov_y = DEFAULTS["fov_y"]
    skew = DEFAULTS["skew"]
    cx_offset = DEFAULTS["cx_offset"]
    cy_offset = DEFAULTS["cy_offset"]

    # Mode
    camera_frame = False  # False = world frame, True = camera frame

    # Cached matrices
    K = None
    Rt = None
    M = None


state = State()

# Scene (created once)
scene_meshes = create_default_scene()

# Overview camera (fixed)
overview_K = None
overview_Rt = None


# =============================================================================
# Callbacks
# =============================================================================

def on_reference_frame_change(sender, value):
    """Handle world/camera frame toggle, converting slider values to keep the view stable."""
    new_camera_frame = (value == "Camera")

    if new_camera_frame == state.camera_frame:
        return

    # Get current extrinsic to preserve the view
    R_curr = state.Rt[:, :3] if state.Rt is not None else np.eye(3)
    t_curr = state.Rt[:, 3] if state.Rt is not None else np.array([0, 0, 5.0])

    if new_camera_frame:
        # Switching TO camera frame: extract camera pose from current [R|t]
        # Camera position = -R^T @ t
        cam_pos = -R_curr.T @ t_curr
        # Camera orientation = R^T
        R_cam = R_curr.T
        alpha, beta, gamma = euler_from_rotation(R_cam)

        state.rot_x = alpha
        state.rot_y = beta
        state.rot_z = gamma
        state.trans_x = cam_pos[0]
        state.trans_y = cam_pos[1]
        state.trans_z = cam_pos[2]
    else:
        # Switching TO world frame: use current [R|t] directly
        alpha, beta, gamma = euler_from_rotation(R_curr)

        state.rot_x = alpha
        state.rot_y = beta
        state.rot_z = gamma
        state.trans_x = t_curr[0]
        state.trans_y = t_curr[1]
        state.trans_z = t_curr[2]

    state.camera_frame = new_camera_frame

    # Update slider UI to match new values
    _sync_sliders_to_state()


def _sync_sliders_to_state():
    """Update all slider widgets to match current state values."""
    slider_map = {
        "rot_x_slider": state.rot_x,
        "rot_y_slider": state.rot_y,
        "rot_z_slider": state.rot_z,
        "trans_x_slider": state.trans_x,
        "trans_y_slider": state.trans_y,
        "trans_z_slider": state.trans_z,
        "fov_x_slider": state.fov_x,
        "fov_y_slider": state.fov_y,
        "skew_slider": state.skew,
        "cx_offset_slider": state.cx_offset,
        "cy_offset_slider": state.cy_offset,
    }
    for tag, val in slider_map.items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)


def reset_all():
    """Reset all parameters to defaults."""
    for attr, val in DEFAULTS.items():
        if hasattr(state, attr):
            setattr(state, attr, val)
    state.camera_frame = False
    if dpg.does_item_exist("ref_frame_combo"):
        dpg.set_value("ref_frame_combo", "World")
    _sync_sliders_to_state()


def update_image_sizes():
    """Adjust image display sizes to fit viewport."""
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    # Available space for images (below controls)
    available_width = vp_width - 60
    available_height = vp_height - 480

    img_size = min(available_width // 2, available_height)
    img_size = max(img_size, 100)

    for tag in ["camera_image", "overview_image"]:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=img_size, height=img_size)


def on_viewport_resize():
    update_image_sizes()


# =============================================================================
# Main
# =============================================================================

def main():
    global overview_K, overview_Rt

    # Setup overview camera
    overview_Rt = make_lookat_Rt(
        eye=np.array([8.0, 6.0, 8.0]),
        target=np.array([0.0, 0.5, 0.0]),
    )
    overview_K = build_intrinsic(
        fov_to_focal(50, RENDER_W), fov_to_focal(50, RENDER_H),
        0, RENDER_W / 2, RENDER_H / 2,
    )

    # Dear PyGui setup
    dpg.create_context()

    with dpg.texture_registry():
        blank = [0.0] * (RENDER_W * RENDER_H * 4)
        dpg.add_raw_texture(RENDER_W, RENDER_H, blank,
                            format=dpg.mvFormat_Float_rgba, tag="camera_texture")
        dpg.add_raw_texture(RENDER_W, RENDER_H, blank,
                            format=dpg.mvFormat_Float_rgba, tag="overview_texture")

    with dpg.window(label="3D Camera Demo", tag="main_window"):
        # --- Top controls ---
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                label="UI Scale",
                default_value=DEFAULTS["ui_scale"],
                min_value=1.0, max_value=3.0,
                callback=lambda s, v: dpg.set_global_font_scale(v),
                width=100,
            )
            dpg.add_spacer(width=20)
            dpg.add_text("Reference Frame:")
            dpg.add_combo(
                items=["World", "Camera"],
                default_value="World",
                callback=on_reference_frame_change,
                tag="ref_frame_combo",
                width=100,
            )
            dpg.add_spacer(width=20)
            dpg.add_button(label="Reset All", callback=lambda: reset_all())

        dpg.add_separator()

        # --- Parameter controls in 2 columns ---
        with dpg.group(horizontal=True):
            # Column 1: Intrinsic
            with dpg.child_window(width=280, height=230, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Intrinsic K", default_open=True):
                    with create_parameter_table():
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=60)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=130)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=25)

                        add_parameter_row("FOV X", "fov_x_slider", DEFAULTS["fov_x"],
                                          5.0, 170.0,
                                          make_state_updater(state, "fov_x"),
                                          make_reset_callback(state, "fov_x", "fov_x_slider", DEFAULTS["fov_x"]),
                                          format_str="%.1f")
                        add_parameter_row("FOV Y", "fov_y_slider", DEFAULTS["fov_y"],
                                          5.0, 170.0,
                                          make_state_updater(state, "fov_y"),
                                          make_reset_callback(state, "fov_y", "fov_y_slider", DEFAULTS["fov_y"]),
                                          format_str="%.1f")
                        add_parameter_row("Skew", "skew_slider", DEFAULTS["skew"],
                                          -500.0, 500.0,
                                          make_state_updater(state, "skew"),
                                          make_reset_callback(state, "skew", "skew_slider", DEFAULTS["skew"]),
                                          format_str="%.1f")
                        add_parameter_row("CX off", "cx_offset_slider", DEFAULTS["cx_offset"],
                                          -200.0, 200.0,
                                          make_state_updater(state, "cx_offset"),
                                          make_reset_callback(state, "cx_offset", "cx_offset_slider", DEFAULTS["cx_offset"]),
                                          format_str="%.1f")
                        add_parameter_row("CY off", "cy_offset_slider", DEFAULTS["cy_offset"],
                                          -200.0, 200.0,
                                          make_state_updater(state, "cy_offset"),
                                          make_reset_callback(state, "cy_offset", "cy_offset_slider", DEFAULTS["cy_offset"]),
                                          format_str="%.1f")

            dpg.add_spacer(width=10)

            # Column 2: Extrinsic
            with dpg.child_window(width=280, height=230, border=False, no_scrollbar=True):
                with dpg.collapsing_header(label="Extrinsic [R|t]", default_open=True):
                    with create_parameter_table():
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=60)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=130)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=25)

                        add_parameter_row("Rot X", "rot_x_slider", DEFAULTS["rot_x"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_x"),
                                          make_reset_callback(state, "rot_x", "rot_x_slider", DEFAULTS["rot_x"]),
                                          format_str="%.2f")
                        add_parameter_row("Rot Y", "rot_y_slider", DEFAULTS["rot_y"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_y"),
                                          make_reset_callback(state, "rot_y", "rot_y_slider", DEFAULTS["rot_y"]),
                                          format_str="%.2f")
                        add_parameter_row("Rot Z", "rot_z_slider", DEFAULTS["rot_z"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_z"),
                                          make_reset_callback(state, "rot_z", "rot_z_slider", DEFAULTS["rot_z"]),
                                          format_str="%.2f")
                        add_parameter_row("Trans X", "trans_x_slider", DEFAULTS["trans_x"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_x"),
                                          make_reset_callback(state, "trans_x", "trans_x_slider", DEFAULTS["trans_x"]),
                                          format_str="%.2f")
                        add_parameter_row("Trans Y", "trans_y_slider", DEFAULTS["trans_y"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_y"),
                                          make_reset_callback(state, "trans_y", "trans_y_slider", DEFAULTS["trans_y"]),
                                          format_str="%.2f")
                        add_parameter_row("Trans Z", "trans_z_slider", DEFAULTS["trans_z"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_z"),
                                          make_reset_callback(state, "trans_z", "trans_z_slider", DEFAULTS["trans_z"]),
                                          format_str="%.2f")

        dpg.add_separator()

        # --- Matrix display: 3 panels side by side ---
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("K (Intrinsic 3x3):", color=(150, 200, 255))
                dpg.add_text("", tag="k_text")
            dpg.add_spacer(width=30)
            with dpg.group():
                dpg.add_text("[R|t] (Extrinsic 3x4):", color=(150, 255, 150))
                dpg.add_text("", tag="rt_text")
            dpg.add_spacer(width=30)
            with dpg.group():
                dpg.add_text("M = K[R|t] (Full 3x4):", color=(255, 200, 150))
                dpg.add_text("", tag="m_text")

        dpg.add_separator()

        # --- Image panels ---
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text(f"Camera View ({RENDER_W}x{RENDER_H})", color=(150, 200, 255))
                dpg.add_image("camera_texture", tag="camera_image")
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_text("Overview (with frustum)", color=(150, 255, 150))
                dpg.add_image("overview_texture", tag="overview_image")

    # Viewport setup
    setup_viewport("3D Camera Demo", 1100, 850,
                    "main_window", on_viewport_resize, DEFAULTS["ui_scale"])
    update_image_sizes()

    # === Main loop ===
    while dpg.is_dearpygui_running():
        # Build intrinsic K
        fx = fov_to_focal(state.fov_x, RENDER_W)
        fy = fov_to_focal(state.fov_y, RENDER_H)
        cx = RENDER_W / 2.0 + state.cx_offset
        cy = RENDER_H / 2.0 + state.cy_offset
        K = build_intrinsic(fx, fy, state.skew, cx, cy)

        # Build extrinsic [R|t]
        Rt, R = build_extrinsic(
            state.rot_x, state.rot_y, state.rot_z,
            state.trans_x, state.trans_y, state.trans_z,
            1.0, camera_frame=state.camera_frame,
        )

        # Full camera matrix
        M = K @ Rt
        state.K, state.Rt, state.M = K, Rt, M

        # Render camera view
        camera_img = render_scene(scene_meshes, K, Rt, RENDER_W, RENDER_H)

        # Render overview (scene + frustum + axes)
        frustum_meshes = make_frustum_mesh(K, Rt, RENDER_W, RENDER_H, near=0.3, far=5.0)
        world_axes = make_axis_mesh(origin=(0, 0, 0), length=1.5)
        cam_axes = make_camera_axes_mesh(Rt, length=0.8)
        overview_meshes = scene_meshes + frustum_meshes + world_axes + cam_axes
        overview_img = render_scene(overview_meshes, overview_K, overview_Rt,
                                     RENDER_W, RENDER_H)

        # Update textures
        dpg.set_value("camera_texture", convert_cv_to_dpg(camera_img))
        dpg.set_value("overview_texture", convert_cv_to_dpg(overview_img))

        # Update matrix display
        dpg.set_value("k_text", _fmt_mat(K))
        dpg.set_value("rt_text", _fmt_mat(Rt))
        dpg.set_value("m_text", _fmt_mat(M))

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def _fmt_mat(mat):
    """Format matrix for display as aligned text."""
    rows, cols = mat.shape
    lines = []
    for r in range(rows):
        cells = " ".join(f"{mat[r, c]:8.2f}" for c in range(cols))
        lines.append(f"[ {cells} ]")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
