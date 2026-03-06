"""
Live Epipolar Lines Demo — Epipolar Geometry & Triangulation
CSCI 1430 - Brown University

Interactive demonstration of epipolar lines and triangulation from two calibrated cameras.

Core math (HW3 Task 5):
  Given pixel (u1,v1) in camera 1 and (u2,v2) in camera 2, solve:
    A · P_h = 0  where  A (4×4) = [u1·M1[2]-M1[0]; v1·M1[2]-M1[1];
                                     u2·M2[2]-M2[0]; v2·M2[2]-M2[1]]
  Solution: P_h = Vt[-1] (last row of V^T from SVD of A)
  Dehomogenise: P = P_h[:3] / P_h[3]
  Cheirality check: M[2] @ [P;1] > 0  (positive depth in each camera)

Also shows the epipolar line in the right view when the user clicks in the left view:
  l2 = F @ x1  where  F is derived from M1, M2 analytically.

Instructions:
  1. Left-click in the LEFT panel  → epipolar line appears in the right panel
  2. Left-click in the RIGHT panel → triangulation performed, result shown in 3D
  3. Right-click anywhere          → clear all clicks

Colour coding:
  Cyan dot       — clicked point
  Yellow line    — epipolar line (click in left view to see)
  Green sphere   — triangulated point (passes cheirality check, in front of both cameras)
  Red sphere     — triangulated point (fails cheirality — behind at least one camera)
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt,
    fov_to_focal, render_scene,
    make_frustum_mesh, make_axis_mesh,
    make_sphere, make_default_scene,
    compute_F_from_cameras, epipolar_line_endpoints,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, bind_mono_font,
    setup_viewport,
    add_global_controls, control_panel,
    poll_collapsible_panels,
    get_image_pixel_coords,
    make_camera_callback,
    create_blank_texture,
)


# =============================================================================
# Constants
# =============================================================================

IMG_W, IMG_H   = 420, 380
OVERVIEW_SIZE  = 420
DEFAULTS = {"ui_scale": 1.5}

GUIDE_EPIPOLAR = [
    {"title": "Triangulation from two views",
     "body": "Each 2D point defines a ray from its camera center through the "
             "image plane. Two rays from different cameras intersect at the 3D "
             "point. Click in image 1, then click the corresponding point in "
             "image 2 to triangulate."},
    {"title": "The 4\u00d74 DLT system",
     "body": "Cross-product elimination turns x \u223c MX into AX = 0. "
             "Each view contributes 2 equations; together they form a 4\u00d74 system. "
             "The SVD null vector gives the 3D point in homogeneous coordinates. "
             "Dehomogenise by dividing by the 4th component."},
    {"title": "Epipolar constraint",
     "body": "Click in image 1 to see the epipolar line in image 2. "
             "The true corresponding point must lie on this line: l\u2082 = F\u00b7x\u2081. "
             "Use the epipolar line as a guide when clicking in image 2."},
    {"title": "Cheirality and reprojection",
     "body": "Valid triangulation requires positive depth in both cameras "
             "(the point must be in front of both). Green spheres pass the "
             "cheirality check; red spheres are behind at least one camera. "
             "Reprojection error measures how well the 3D point projects back "
             "to the original clicks."},
]

# Two camera matrices
K_SHARED = build_intrinsic(
    fx=350.0, fy=350.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)
_Rt1 = make_lookat_Rt(eye=np.array([-2.5, 1.0, 5.0]),
                       target=np.array([0.0, 0.0, 0.0]))
_Rt2 = make_lookat_Rt(eye=np.array([2.5,  1.0, 5.0]),
                       target=np.array([0.0, 0.0, 0.0]))
M1 = K_SHARED @ _Rt1
M2 = K_SHARED @ _Rt2

_OV_Rt = make_lookat_Rt(eye=np.array([0.0, 6.0, 10.0]),
                          target=np.array([0.0, 0.0, 0.0]))
_OV_K  = build_intrinsic(fov_to_focal(50, OVERVIEW_SIZE), fov_to_focal(50, OVERVIEW_SIZE),
                          0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2)

# Render scene once (used as the static background for both camera views)
_SCENE = make_default_scene()


# =============================================================================
# Triangulation Math
# =============================================================================

def triangulate_point(M1, M2, pt1, pt2):
    """
    Triangulate a 3D point from two 2D correspondences using DLT.

    Builds the 4×4 linear system A·P_h = 0:
      row 0:  u1·M1[2] - M1[0]
      row 1:  v1·M1[2] - M1[1]
      row 2:  u2·M2[2] - M2[0]
      row 3:  v2·M2[2] - M2[1]

    Solution via SVD null vector; dehomogenise to get 3D point.
    """
    u1, v1 = float(pt1[0]), float(pt1[1])
    u2, v2 = float(pt2[0]), float(pt2[1])
    A = np.vstack([
        u1 * M1[2] - M1[0],
        v1 * M1[2] - M1[1],
        u2 * M2[2] - M2[0],
        v2 * M2[2] - M2[1],
    ])                                 # (4, 4)
    _, _, Vt = np.linalg.svd(A)
    Ph = Vt[-1]                        # last row of Vt = null vector, shape (4,)
    if abs(Ph[3]) < 1e-10:
        return None                    # degenerate: point at infinity
    return Ph[:3] / Ph[3]             # (3,)


def check_cheirality(M, P):
    """True if 3D point P is in front of camera M (positive depth row)."""
    Ph = np.append(P, 1.0)
    return float(M[2] @ Ph) > 0


def reprojection_error(M, P):
    """Reprojection error |proj(M, P) - observed| is not shown here; just project."""
    Ph = np.append(P, 1.0)
    p = M @ Ph
    return p[:2] / p[2]


# Pre-compute F
F_MATRIX = compute_F_from_cameras(M1, M2)


# =============================================================================
# Render base camera views (static background, rendered once)
# =============================================================================

_BASE_LEFT  = render_scene(_SCENE, K_SHARED, _Rt1, IMG_W, IMG_H)
_BASE_RIGHT = render_scene(_SCENE, K_SHARED, _Rt2, IMG_W, IMG_H)


# =============================================================================
# State
# =============================================================================

class State:
    click_left  = None      # (u, v) in left image
    click_right = None      # (u, v) in right image
    tri_point   = None      # triangulated 3D point or None
    tri_valid   = False     # cheirality passed

    # Image widget screen positions (updated each frame via dpg)
    left_img_pos  = (0, 0)
    right_img_pos = (0, 0)


state = State()


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_view(base_img, click_px, epipolar=None, label=""):
    """Draw camera view with optional click dot and epipolar line."""
    canvas = base_img.copy()

    if epipolar is not None:
        l, color = epipolar
        p1, p2 = epipolar_line_endpoints(l, IMG_W, IMG_H)
        if p1 is not None:
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        # Label
        cv2.putText(canvas, "Epipolar line", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if click_px is not None:
        u, v = int(click_px[0]), int(click_px[1])
        cv2.circle(canvas, (u, v), 8, (255, 220, 0), -1)   # cyan-yellow dot
        cv2.circle(canvas, (u, v), 8, (255, 255, 255), 1)

    cv2.putText(canvas, label, (10, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# =============================================================================
# Click Detection
# =============================================================================

def handle_clicks():
    """Process mouse clicks and update state."""
    if dpg.is_mouse_button_clicked(1):   # right click -> clear
        state.click_left  = None
        state.click_right = None
        state.tri_point   = None
        state.tri_valid   = False
        return

    if dpg.is_mouse_button_clicked(0):   # left click
        coords_l = get_image_pixel_coords("left_img", IMG_W, IMG_H)
        if coords_l is not None:
            state.click_left  = coords_l
            state.click_right = None
            state.tri_point   = None
            state.tri_valid   = False
            return

        coords_r = get_image_pixel_coords("right_img", IMG_W, IMG_H)
        if coords_r is not None and state.click_left is not None:
            state.click_right = coords_r
            # Triangulate
            P = triangulate_point(M1, M2, state.click_left, state.click_right)
            if P is not None:
                state.tri_point = P
                state.tri_valid = (check_cheirality(M1, P) and
                                   check_cheirality(M2, P))
            return


# =============================================================================
# Main
# =============================================================================

def main():
    dpg.create_context()

    load_fonts()

    with dpg.texture_registry(tag="texture_registry"):
        for tag in ["left_tex", "right_tex"]:
            create_blank_texture(IMG_W, IMG_H, tag)
        create_blank_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, "overview_tex")

    with dpg.window(label="Epipolar Lines Demo", tag="main_window"):

        # ── Top bar ──────────────────────────────────────────────────────────
        def _clear_clicks():
            state.click_left = None
            state.click_right = None
            state.tri_point = None
            state.tri_valid = False

        add_global_controls(
            DEFAULTS, state,
            camera_callback=make_camera_callback(state),
            reset_extra=_clear_clicks,
            guide=GUIDE_EPIPOLAR, guide_title="Epipolar Lines & Triangulation",
        )

        dpg.add_separator()

        dpg.add_text("", tag="status_text", color=(255, 220, 100))

        # ── Main panels ───────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            # Controls
            with control_panel("Controls", width=180, height=300,
                               color=(255, 220, 100)):
                dpg.add_button(
                    label="Clear clicks", callback=_clear_clicks,
                )
                dpg.add_spacer(height=10)
                dpg.add_text(
                    "Left-click in LEFT view \u2192 epipolar line\n"
                    "Left-click in RIGHT view \u2192 triangulate\n"
                    "Right-click \u2192 clear",
                    color=(200, 200, 120), wrap=160,
                )

            dpg.add_spacer(width=10)

            # Left camera view
            with dpg.group():
                dpg.add_text("Camera 1 (Left)", color=(150, 255, 150))
                dpg.add_text("← click here first", color=(150, 150, 150))
                dpg.add_image("left_tex",  tag="left_img",
                              width=IMG_W, height=IMG_H)

            dpg.add_spacer(width=20)

            # Right camera view
            with dpg.group():
                dpg.add_text("Camera 2 (Right)", color=(150, 200, 255))
                dpg.add_text("← then click here", color=(150, 150, 150))
                dpg.add_image("right_tex", tag="right_img",
                              width=IMG_W, height=IMG_H)

            dpg.add_spacer(width=20)

            # 3D overview
            with dpg.group():
                dpg.add_text("3D Overview", color=(220, 180, 100))
                dpg.add_text("(triangulated point shown here)", color=(150, 150, 150))
                dpg.add_image("overview_tex", tag="overview_img",
                              width=OVERVIEW_SIZE, height=OVERVIEW_SIZE)

        dpg.add_separator()

        # ── Math display ──────────────────────────────────────────────────────
        with dpg.collapsing_header(label="Math: Triangulation A matrix", default_open=False):
            dpg.add_text("", tag="math_text", color=(160, 200, 255))

    bind_mono_font("math_text")

    setup_viewport("Epipolar Lines & Triangulation", 1400, 750,
                   "main_window", lambda: None, DEFAULTS["ui_scale"])

    # ── Main loop ─────────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        handle_clicks()

        # Compute epipolar line in right view if left clicked
        epi_right = None
        if state.click_left is not None:
            x1h = np.array([state.click_left[0], state.click_left[1], 1.0])
            l2 = F_MATRIX @ x1h
            epi_right = (l2, (0, 200, 255))    # yellow (BGR: 0,200,255)

        # Build camera view images
        left_img  = draw_view(_BASE_LEFT,  state.click_left,
                               label="Left-click to pick a point")
        right_img = draw_view(_BASE_RIGHT, state.click_right,
                               epipolar=epi_right,
                               label="Left-click on the epipolar line")

        # Build 3D overview with triangulated point
        pt_meshes = []
        if state.tri_point is not None:
            P = state.tri_point
            color = (50, 200, 50) if state.tri_valid else (50, 50, 200)  # BGR green/red
            pt_meshes = [make_sphere(tuple(P), radius=0.12, color=color)]

        frust1 = make_frustum_mesh(K_SHARED, _Rt1, IMG_W, IMG_H, near=0.3, far=7.0)
        frust2 = make_frustum_mesh(K_SHARED, _Rt2, IMG_W, IMG_H, near=0.3, far=7.0)
        axes   = make_axis_mesh(origin=(0, 0, 0), length=1.5)

        ov_img = render_scene(
            _SCENE + pt_meshes + frust1 + frust2 + axes,
            _OV_K, _OV_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE,
        )

        # Update textures
        dpg.set_value("left_tex",    convert_cv_to_dpg(left_img))
        dpg.set_value("right_tex",   convert_cv_to_dpg(right_img))
        dpg.set_value("overview_tex", convert_cv_to_dpg(ov_img))

        # Status bar
        if state.tri_point is None:
            if state.click_left is None:
                status = "Click a point in the LEFT view to begin."
            else:
                u1, v1 = state.click_left
                status = (f"Left click at ({u1:.1f}, {v1:.1f})  →  "
                          f"epipolar line shown in right view.  "
                          f"Now click on the line in the right view.")
        else:
            P = state.tri_point
            chk_str = "OK: in front of both cameras" if state.tri_valid \
                      else "FAIL: behind at least one camera (cheirality)"
            # Reprojection errors
            p1_est = reprojection_error(M1, P)
            p2_est = reprojection_error(M2, P)
            err1 = np.linalg.norm(p1_est - np.array(state.click_left))
            err2 = np.linalg.norm(p2_est - np.array(state.click_right))
            status = (
                f"Triangulated P = ({P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f})  |  "
                f"Cheirality: {chk_str}  |  "
                f"Reproj err: cam1={err1:.2f}px, cam2={err2:.2f}px"
            )
        dpg.set_value("status_text", status)

        # Math display
        if state.click_left is not None and state.click_right is not None:
            u1, v1 = state.click_left
            u2, v2 = state.click_right
            A = np.vstack([
                u1 * M1[2] - M1[0],
                v1 * M1[2] - M1[1],
                u2 * M2[2] - M2[0],
                v2 * M2[2] - M2[1],
            ])
            lines = ["A (4×4) = [u1·M1[2]−M1[0]; v1·M1[2]−M1[1]; u2·M2[2]−M2[0]; v2·M2[2]−M2[1]]"]
            for r in range(4):
                vals = "  ".join(f"{v:8.3f}" for v in A[r])
                lines.append(f"  row {r}: [{vals}]")
            if state.tri_point is not None:
                P = state.tri_point
                lines.append(f"\nP_h (SVD null vector dehomogenised):")
                lines.append(f"  P = [{P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f}]")
                c1_ok = check_cheirality(M1, P)
                c2_ok = check_cheirality(M2, P)
                lines.append(f"\nCheirality: M1[2]@[P;1] = {M1[2]@np.append(P,1):.3f} "
                             f"({'> 0 OK' if c1_ok else '< 0 FAIL'})   "
                             f"M2[2]@[P;1] = {M2[2]@np.append(P,1):.3f} "
                             f"({'> 0 OK' if c2_ok else '< 0 FAIL'})")
            dpg.set_value("math_text", "\n".join(lines))
        else:
            dpg.set_value("math_text",
                          "Click in both views to see the 4×4 A matrix here.")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
