"""
Live Triangulation Demo — Sparse 3D Reconstruction
CSCI 1430 - Brown University

Interactive demonstration of triangulation from two calibrated cameras.

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
    make_sphere, create_default_scene,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, bind_mono_font,
    setup_viewport,
)


# =============================================================================
# Constants
# =============================================================================

IMG_W, IMG_H   = 420, 380
OVERVIEW_SIZE  = 420
UI_SCALE       = 1.5

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
_SCENE = create_default_scene()


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


def compute_F_from_cameras(M1, M2):
    """Compute fundamental matrix F from two camera matrices.

    F = [e2]_x @ M2 @ pinv(M1)
    where e2 = M2 @ C1 is the epipole in image 2.
    """
    A1 = M1[:, :3]
    C1 = np.linalg.solve(A1, -M1[:, 3])
    C1h = np.append(C1, 1.0)

    e2 = M2 @ C1h    # epipole in image 2, shape (3,)
    e2x = np.array([
        [0,      -e2[2],  e2[1]],
        [e2[2],   0,     -e2[0]],
        [-e2[1],  e2[0],  0    ],
    ])

    F = e2x @ M2 @ np.linalg.pinv(M1)
    return F


def epipolar_line_endpoints(l, img_w, img_h):
    """Return two endpoints of line l=[a,b,c] (ax+by+c=0) clipped to image."""
    a, b, c = l
    pts = []
    if abs(b) > 1e-6:
        # Intersect x=0 and x=img_w-1
        y0 = -c / b
        y1 = -(c + a * (img_w - 1)) / b
        pts.append((0, int(y0)))
        pts.append((img_w - 1, int(y1)))
    if abs(a) > 1e-6:
        # Intersect y=0 and y=img_h-1
        x0 = -c / a
        x1 = -(c + b * (img_h - 1)) / a
        pts.append((int(x0), 0))
        pts.append((int(x1), img_h - 1))

    # Keep the pair that spans the image most widely
    if len(pts) >= 2:
        best_p1, best_p2, best_d = pts[0], pts[1], 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if d > best_d:
                    best_p1, best_p2, best_d = pts[i], pts[j], d
        return best_p1, best_p2
    return None, None


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

def _get_local_coords(img_tag, mx, my):
    """Convert global mouse coordinates to local image pixel coordinates."""
    try:
        rect_min = dpg.get_item_rect_min(img_tag)
        rect_max = dpg.get_item_rect_max(img_tag)
        rw = rect_max[0] - rect_min[0]
        rh = rect_max[1] - rect_min[1]
        if rw < 1 or rh < 1:
            return None
        lx = (mx - rect_min[0]) / rw * IMG_W
        ly = (my - rect_min[1]) / rh * IMG_H
        if 0 <= lx < IMG_W and 0 <= ly < IMG_H:
            return (lx, ly)
    except Exception:
        pass
    return None


def handle_clicks():
    """Process mouse clicks and update state."""
    if dpg.is_mouse_button_clicked(1):   # right click → clear
        state.click_left  = None
        state.click_right = None
        state.tri_point   = None
        state.tri_valid   = False
        return

    if dpg.is_mouse_button_clicked(0):   # left click
        mx, my = dpg.get_mouse_pos()

        coords_l = _get_local_coords("left_img",  mx, my)
        if coords_l is not None:
            state.click_left  = coords_l
            state.click_right = None
            state.tri_point   = None
            state.tri_valid   = False
            return

        coords_r = _get_local_coords("right_img", mx, my)
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
        blank = [0.0] * (IMG_W * IMG_H * 4)
        for tag in ["left_tex", "right_tex"]:
            dpg.add_raw_texture(IMG_W, IMG_H, list(blank),
                                format=dpg.mvFormat_Float_rgba, tag=tag)
        blank_ov = [0.0] * (OVERVIEW_SIZE * OVERVIEW_SIZE * 4)
        dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, blank_ov,
                            format=dpg.mvFormat_Float_rgba, tag="overview_tex")

    with dpg.window(label="Triangulation Demo", tag="main_window"):

        # ── Top bar ──────────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="UI Scale",
                items=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"],
                default_value=str(UI_SCALE), width=80,
                callback=lambda s, v: dpg.set_global_font_scale(float(v)),
            )
            dpg.add_spacer(width=20)
            dpg.add_button(
                label="Clear clicks",
                callback=lambda: (
                    setattr(state, "click_left",  None),
                    setattr(state, "click_right", None),
                    setattr(state, "tri_point",   None),
                    setattr(state, "tri_valid",   False),
                ),
            )
            dpg.add_spacer(width=20)
            dpg.add_text(
                "Left-click in LEFT view → epipolar line  |  "
                "Left-click in RIGHT view → triangulate  |  "
                "Right-click → clear",
                color=(200, 200, 120),
            )

        dpg.add_separator()
        dpg.add_text("", tag="status_text", color=(255, 220, 100))
        dpg.add_separator()

        # ── Main panels ───────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
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

    setup_viewport("Triangulation Demo (DLT)", 1400, 750,
                   "main_window", lambda: None, UI_SCALE)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
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
