"""
Web frame module for liveTriangulation demo.
Click correspondences in two camera views to triangulate 3D points.
"""

import numpy as np
import cv2

from liveTriangulation import (
    triangulate_point, compute_F_from_cameras,
    check_cheirality, reprojection_error, epipolar_line_endpoints,
    draw_view,
    M1, M2, F_MATRIX, _BASE_LEFT, _BASE_RIGHT,
    K_SHARED, _Rt1, _Rt2, _OV_K, _OV_Rt, _SCENE,
    IMG_W, IMG_H, OVERVIEW_SIZE,
)
from utils.demo_3d import render_scene, make_frustum_mesh, make_axis_mesh, make_sphere

# ── Module-level state ──
_click_left = None    # (u, v) in left image
_click_right = None   # (u, v) in right image
_tri_point = None     # triangulated 3D point (x, y, z)
_tri_valid = False    # cheirality check passed

WEB_CONFIG = {
    "title": "Sparse Triangulation",
    "description": (
        "Click correspondences in two camera views to triangulate 3D points. "
        "The epipolar line guides you to the correct match. "
        "Green = valid (in front of both cameras), Red = behind camera."
    ),
    "outputs": [
        {"id": "left_img",  "label": "Left Camera",  "width": IMG_W, "height": IMG_H},
        {"id": "right_img", "label": "Right Camera", "width": IMG_W, "height": IMG_H},
        {"id": "overview",  "label": "3D Overview",   "width": OVERVIEW_SIZE, "height": OVERVIEW_SIZE},
    ],
    "controls": {
        "clear": {
            "type": "button", "label": "Clear Clicks",
        },
    },
    "mouse": ["left_img", "right_img"],
    "layout": {"rows": [["left_img", "right_img"], ["overview"]]},
}


def web_button(button_id):
    """Handle button clicks."""
    global _click_left, _click_right, _tri_point, _tri_valid
    if button_id == "clear":
        _click_left = None
        _click_right = None
        _tri_point = None
        _tri_valid = False


def web_mouse(event):
    """Handle mouse events on camera view canvases."""
    global _click_left, _click_right, _tri_point, _tri_valid

    # Right-click clears
    if event["type"] == "contextmenu" or event["button"] == 2:
        _click_left = None
        _click_right = None
        _tri_point = None
        _tri_valid = False
        return

    if event["type"] != "click" or event["button"] != 0:
        return

    x, y = event["x"], event["y"]

    # Bounds check
    if x < 0 or y < 0:
        return

    if event["canvas"] == "left_img":
        if x > IMG_W or y > IMG_H:
            return
        _click_left = (x, y)
        _click_right = None
        _tri_point = None
        _tri_valid = False

    elif event["canvas"] == "right_img":
        if x > IMG_W or y > IMG_H:
            return
        if _click_left is None:
            return  # Must click left first
        _click_right = (x, y)

        # Triangulate
        P = triangulate_point(M1, M2, _click_left, _click_right)
        if P is not None:
            _tri_point = P
            _tri_valid = check_cheirality(M1, P) and check_cheirality(M2, P)
        else:
            _tri_point = None
            _tri_valid = False


def web_frame(state):
    # Left view
    epipolar_left = None
    if _click_left is not None:
        # Compute epipolar line in right image from left click
        x1h = np.array([_click_left[0], _click_left[1], 1.0])
        l2 = F_MATRIX @ x1h
        epipolar_right = (l2, (100, 200, 255))  # Blue
    else:
        epipolar_right = None

    left_canvas = draw_view(_BASE_LEFT, _click_left, label="Left Camera")
    right_canvas = draw_view(_BASE_RIGHT, _click_right,
                              epipolar=epipolar_right, label="Right Camera")

    # 3D overview
    meshes = list(_SCENE)
    frustum1 = make_frustum_mesh(K_SHARED, _Rt1, IMG_W, IMG_H, near=0.3, far=6.0)
    frustum2 = make_frustum_mesh(K_SHARED, _Rt2, IMG_W, IMG_H, near=0.3, far=6.0)
    axes = make_axis_mesh(origin=(0, 0, 0), length=1.0)
    meshes += frustum1 + frustum2 + axes

    if _tri_point is not None:
        color = (0, 220, 0) if _tri_valid else (0, 0, 220)
        meshes.append(make_sphere(_tri_point, radius=0.15, color=color, n_lat=8))

    overview = render_scene(meshes, _OV_K, _OV_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE)

    # Status
    status_parts = []
    if _click_left is not None:
        status_parts.append(f"Left: ({_click_left[0]:.0f}, {_click_left[1]:.0f})")
    if _click_right is not None:
        status_parts.append(f"Right: ({_click_right[0]:.0f}, {_click_right[1]:.0f})")
    if _tri_point is not None:
        P = _tri_point
        validity = "VALID" if _tri_valid else "BEHIND CAMERA"
        status_parts.append(f"3D: ({P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f}) [{validity}]")
        # Reprojection errors
        repr1 = reprojection_error(M1, P)
        repr2 = reprojection_error(M2, P)
        if _click_left is not None:
            e1 = np.linalg.norm(repr1 - np.array(_click_left))
            status_parts.append(f"Reproj L: {e1:.1f}px")
        if _click_right is not None:
            e2 = np.linalg.norm(repr2 - np.array(_click_right))
            status_parts.append(f"Reproj R: {e2:.1f}px")
    if not status_parts:
        status_parts.append("Click left image first, then right image")

    status = "  |  ".join(status_parts)

    return {
        "left_img": left_canvas,
        "right_img": right_canvas,
        "overview": overview,
        "status": status,
    }
