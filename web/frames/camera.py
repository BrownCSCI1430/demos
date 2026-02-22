"""
Web frame module for liveCamera demo.
Interactive 3D camera with intrinsic/extrinsic matrix exploration.
"""

import numpy as np

from liveCamera import (
    compute_camera_frame, on_reference_frame_change,
    scene_meshes, overview_K, overview_Rt,
    DEFAULTS, OVERVIEW_SIZE, state as cam_state,
    _fmt_mat, _K_SYMBOLIC, _RT_SYMBOLIC, _M_SYMBOLIC,
)
from utils.demo_3d import euler_from_rotation

WEB_CONFIG = {
    "title": "3D Camera",
    "description": (
        "Interactive 3D camera demo. Explore intrinsic (K) and extrinsic [R|t] "
        "matrices with real-time software rendering. Toggle between world and "
        "camera reference frames."
    ),
    "outputs": [
        {"id": "camera_view", "label": "Camera View", "width": 400, "height": 400},
        {"id": "overview",    "label": "3D Overview",  "width": 400, "height": 400},
    ],
    "controls": {
        "reference_frame": {
            "type": "choice", "options": ["World", "Camera"],
            "default": "World", "label": "Reference Frame",
        },
        "img_w": {
            "type": "int", "min": 100, "max": 800, "step": 10,
            "default": DEFAULTS["img_w"], "label": "img_w (px)", "format": "d",
        },
        "img_h": {
            "type": "int", "min": 100, "max": 800, "step": 10,
            "default": DEFAULTS["img_h"], "label": "img_h (px)", "format": "d",
        },
        "skew": {
            "type": "float", "min": -500.0, "max": 500.0, "step": 1.0,
            "default": DEFAULTS["skew"], "label": "Skew s (px)", "format": ".1f",
        },
        "cx_offset": {
            "type": "float", "min": -200.0, "max": 200.0, "step": 1.0,
            "default": DEFAULTS["cx_offset"], "label": "cx offset (px)", "format": ".1f",
        },
        "cy_offset": {
            "type": "float", "min": -200.0, "max": 200.0, "step": 1.0,
            "default": DEFAULTS["cy_offset"], "label": "cy offset (px)", "format": ".1f",
        },
        "focal_length": {
            "type": "float", "min": 5.0, "max": 200.0, "step": 0.5,
            "default": DEFAULTS["focal_length"], "label": "Focal (mm)",
        },
        "sensor_w": {
            "type": "float", "min": 5.0, "max": 100.0, "step": 0.5,
            "default": DEFAULTS["sensor_w"], "label": "Sensor W (mm)",
        },
        "sensor_h": {
            "type": "float", "min": 5.0, "max": 100.0, "step": 0.5,
            "default": DEFAULTS["sensor_h"], "label": "Sensor H (mm)",
        },
        "rot_x": {
            "type": "float", "min": -3.14159, "max": 3.14159, "step": 0.01,
            "default": DEFAULTS["rot_x"], "label": "Rot X (rad)",
        },
        "rot_y": {
            "type": "float", "min": -3.14159, "max": 3.14159, "step": 0.01,
            "default": DEFAULTS["rot_y"], "label": "Rot Y (rad)",
        },
        "rot_z": {
            "type": "float", "min": -3.14159, "max": 3.14159, "step": 0.01,
            "default": DEFAULTS["rot_z"], "label": "Rot Z (rad)",
        },
        "trans_x": {
            "type": "float", "min": -10.0, "max": 10.0, "step": 0.1,
            "default": DEFAULTS["trans_x"], "label": "Trans X",
        },
        "trans_y": {
            "type": "float", "min": -10.0, "max": 10.0, "step": 0.1,
            "default": DEFAULTS["trans_y"], "label": "Trans Y",
        },
        "trans_z": {
            "type": "float", "min": -10.0, "max": 10.0, "step": 0.1,
            "default": DEFAULTS["trans_z"], "label": "Trans Z",
        },
        "show_symbolic": {
            "type": "bool", "default": False, "label": "Show Algebraic",
        },
        "reset_all": {
            "type": "button", "label": "Reset All",
        },
    },
    "layout": {"rows": [["camera_view", "overview"]]},
}

# Track previous reference frame to detect changes
_prev_ref_frame = "World"


def web_button(button_id):
    """Handle button clicks."""
    global _prev_ref_frame
    if button_id == "reset_all":
        _prev_ref_frame = "World"


def web_frame(state):
    global _prev_ref_frame

    # Detect reference frame change and compute new slider values
    ref_frame = state["reference_frame"]
    set_controls = None

    if ref_frame != _prev_ref_frame:
        # Sync cam_state with current slider values before frame conversion
        cam_state.rot_x = state["rot_x"]
        cam_state.rot_y = state["rot_y"]
        cam_state.rot_z = state["rot_z"]
        cam_state.trans_x = state["trans_x"]
        cam_state.trans_y = state["trans_y"]
        cam_state.trans_z = state["trans_z"]
        cam_state.camera_frame = (_prev_ref_frame == "Camera")

        # Use the desktop demo's reference frame conversion
        on_reference_frame_change(None, ref_frame)
        _prev_ref_frame = ref_frame

        # Return the converted values to update sliders
        set_controls = {
            "rot_x": cam_state.rot_x,
            "rot_y": cam_state.rot_y,
            "rot_z": cam_state.rot_z,
            "trans_x": cam_state.trans_x,
            "trans_y": cam_state.trans_y,
            "trans_z": cam_state.trans_z,
        }

        # Use the converted values for rendering
        state["rot_x"] = cam_state.rot_x
        state["rot_y"] = cam_state.rot_y
        state["rot_z"] = cam_state.rot_z
        state["trans_x"] = cam_state.trans_x
        state["trans_y"] = cam_state.trans_y
        state["trans_z"] = cam_state.trans_z

    # Handle reset_all
    if state.get("_from_reset"):
        _prev_ref_frame = "World"

    camera_frame = (ref_frame == "Camera")

    K, Rt, M, camera_img, overview_img = compute_camera_frame(
        state["focal_length"], state["sensor_w"], state["sensor_h"],
        state["img_w"], state["img_h"],
        state["skew"], state["cx_offset"], state["cy_offset"],
        state["rot_x"], state["rot_y"], state["rot_z"],
        state["trans_x"], state["trans_y"], state["trans_z"],
        camera_frame,
    )

    # Store for reference frame conversion
    cam_state.K = K
    cam_state.Rt = Rt
    cam_state.M = M

    # Build status with matrix display
    iw, ih = int(state["img_w"]), int(state["img_h"])
    fx = state["focal_length"] * iw / state["sensor_w"]
    fy = state["focal_length"] * ih / state["sensor_h"]

    if state["show_symbolic"]:
        k_str = _K_SYMBOLIC
        rt_str = _RT_SYMBOLIC
        m_str = _M_SYMBOLIC
    else:
        k_str = _fmt_mat(K)
        rt_str = _fmt_mat(Rt)
        m_str = _fmt_mat(M)

    status = (
        f"Camera View ({iw}x{ih})  |  f_x={fx:.1f}px  f_y={fy:.1f}px\n\n"
        f"K =\n{k_str}\n\n"
        f"[R|t] =\n{rt_str}\n\n"
        f"M = K[R|t] =\n{m_str}"
    )

    result = {
        "camera_view": camera_img,
        "overview": overview_img,
        "status": status,
    }

    if set_controls is not None:
        result["set_controls"] = set_controls

    # Handle reset button: return default values
    return result
