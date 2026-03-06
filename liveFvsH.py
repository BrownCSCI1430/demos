"""
Live F vs H(λ) Demo — Fundamental Matrix vs Depth Homography
CSCI 1430 - Brown University

Interactive demonstration comparing what F and H(λ) tell you about correspondences.

F (Fundamental Matrix):
  A point x in View 1 maps to an epipolar LINE l = F·x in View 2.
  The true correspondence lies somewhere on this line, but F alone cannot say where.

H(λ) (Depth-Dependent Homography):
  The same point x maps to a specific POINT H(λ)·x in View 2, assuming depth λ.
  As λ sweeps, the projected point slides along the epipolar line.

E (Essential Matrix):
  E = K^T F K — the "calibrated" fundamental matrix. Its SVD has the special
  structure (σ, σ, 0): the two nonzero singular values are equal.

Layout:
  Top row:    View 1 (click points)  |  View 2: epipolar lines (F)
  Bottom row: View 1 (depth info)    |  View 2: projected dots on epi-lines (H(λ))
  Right side: 3D overview + SVD / matrix panel
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt,
    fov_to_focal, render_scene,
    make_frustum_mesh, make_axis_mesh,
    make_sphere, make_default_scene,
    raycast_scene,
    compute_F_from_cameras, epipolar_line_endpoints,
    decompose_H, compute_H_lam,
    OrbitCamera,
    flip_y_matrix, compute_stereo_cameras,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
    add_global_controls, bind_mono_font, control_panel,
    poll_collapsible_panels,
    get_image_pixel_coords,
    make_camera_callback,
    create_blank_texture,
)


# =============================================================================
# Constants
# =============================================================================

IMG_W, IMG_H = 350, 350
OVERVIEW_SIZE = 350
LAM_MIN, LAM_MAX = 0.5, 8.0
MAX_POINTS = 5

DEFAULTS = {
    "lam": 4.5,
    "cam_baseline": 0.8,
    "cam_convergence": 0.15,
    "cam_distance": 5.0,
    "animate": False,
    "show_matrices": True,
    "show_svd": True,
    "show_3d": True,
    "ui_scale": 1.5,
}

# Color palette for clicked points (BGR)
POINT_COLORS = [
    (0, 200, 255),    # orange
    (255, 100, 100),  # blue
    (100, 255, 100),  # green
    (100, 100, 255),  # red
    (255, 200, 100),  # cyan-ish
]

GUIDE_FvsH = [
    {"title": "F constrains to lines, H(\u03bb) resolves to points",
     "body": "The fundamental matrix F maps a point in View 1 to an epipolar "
             "LINE in View 2: l = F\u00b7x. The true correspondence lies "
             "somewhere on this line.\n\n"
             "H(\u03bb) maps the same point to a specific POINT on that line, "
             "depending on assumed depth \u03bb. Click in View 1 and compare "
             "the two rows."},
    {"title": "Sliding \u03bb: dots move along epipolar lines",
     "body": "As you sweep \u03bb, the H(\u03bb) projection slides along the "
             "epipolar line. At the true object depth, it lands on the "
             "correct correspondence.\n\n"
             "This is the core of plane sweep stereo: try every depth, "
             "keep the one that matches best."},
    {"title": "\u03bb\u2192\u221e: all dots converge to B\u00b7x",
     "body": "At infinite depth, H(\u03bb) \u2248 \u03bbB (the epipole term "
             "vanishes). B\u00b7x is the infinite-depth homography \u2014 pure "
             "rotation, no parallax.\n\n"
             "Try setting \u03bb to the maximum and notice how all projected "
             "dots cluster together."},
    {"title": "\u03bb\u21920: all dots converge to the epipole",
     "body": "At zero depth, H(\u03bb) \u2248 a\u00b7e\u2083\u1d40, and all "
             "points map to the epipole a (the projection of Camera 1's "
             "center into Camera 2). Everything collapses to one point.\n\n"
             "Set \u03bb near the minimum to see this."},
    {"title": "F vs E: calibration peels off K",
     "body": "E = K\u1d40FK. The essential matrix E is F with intrinsics "
             "removed. Its SVD has the special structure (\u03c3,\u03c3,0) \u2014 "
             "the two nonzero singular values are equal.\n\n"
             "F's singular values are unconstrained (\u03c3\u2081 \u2265 "
             "\u03c3\u2082 > 0). Toggle 'Show SVD' to compare."},
    {"title": "M matrices \u2192 everything",
     "body": "F, E, and H(\u03bb) are all derived from the camera matrices "
             "M\u2081, M\u2082. DLT calibration (HW3 Task 1) estimates M.\n\n"
             "From M you get:\n"
             "  F = [e\u2082]\u00d7 M\u2082 M\u2081\u207a  (epipolar geometry)\n"
             "  E = K\u1d40 F K  (calibrated epipolar geometry)\n"
             "  H(\u03bb) = \u03bbB + a e\u2083\u1d40  (depth-dependent warp)"},
]

K_SHARED = build_intrinsic(
    fx=350.0, fy=350.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)


# =============================================================================
# Camera State
# =============================================================================

class Cam:
    """Mutable camera state — updated when baseline/convergence/distance change."""
    Rt_ref = None
    Rt_other = None
    M_ref = None
    M_other = None
    ref_img = None
    other_img = None

cam = Cam()

_SCENE = make_default_scene()


def apply_camera_params(baseline, convergence, distance):
    """Compute cameras from (baseline, convergence, distance) and re-render."""
    cam.Rt_ref, cam.Rt_other, cam.M_ref, cam.M_other = \
        compute_stereo_cameras(K_SHARED, baseline, convergence, distance)
    cam.ref_img  = render_scene(_SCENE, K_SHARED, cam.Rt_ref,
                                IMG_W, IMG_H, flip_y=True)
    cam.other_img = render_scene(_SCENE, K_SHARED, cam.Rt_other,
                                 IMG_W, IMG_H, flip_y=True)

apply_camera_params(DEFAULTS["cam_baseline"], DEFAULTS["cam_convergence"],
                    DEFAULTS["cam_distance"])


_OV_K = build_intrinsic(fov_to_focal(50, OVERVIEW_SIZE),
                         fov_to_focal(50, OVERVIEW_SIZE),
                         0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2)

OvCam = OrbitCamera(eye0=np.array([0.0, 8.0, 12.0]),
                    target=np.array([0.0, 0.5, 0.0]))


# =============================================================================
# Math: F, E, H(λ)
# =============================================================================

def compute_E(F, K):
    """Essential matrix: E = K^T F K."""
    return K.T @ F @ K


def project_H(H, px, py, img_h):
    """Apply H(λ) to a display-coord point, return display coords.

    Camera matrices use un-flipped coords; images use flip_y.
    We conjugate: un-flip → apply H → re-flip.
    """
    # Un-flip y
    uy = img_h - 1 - py
    p_uf = np.array([px, uy, 1.0])
    q = H @ p_uf
    if abs(q[2]) < 1e-10:
        return None
    qx = q[0] / q[2]
    qy = img_h - 1 - q[1] / q[2]  # re-flip
    return (qx, qy)


def compute_F_display(M1, M2, img_h):
    """Compute F in display coordinates (with flip_y).

    F_display = Fy^T @ F_raw @ Fy  where Fy flips y.
    """
    F_raw = compute_F_from_cameras(M1, M2)
    Fy = flip_y_matrix(img_h)
    return Fy.T @ F_raw @ Fy


# =============================================================================
# State
# =============================================================================

class State:
    lam = DEFAULTS["lam"]
    cam_baseline = DEFAULTS["cam_baseline"]
    cam_convergence = DEFAULTS["cam_convergence"]
    cam_distance = DEFAULTS["cam_distance"]
    animate = DEFAULTS["animate"]
    show_matrices = DEFAULTS["show_matrices"]
    show_svd = DEFAULTS["show_svd"]
    show_3d = DEFAULTS["show_3d"]

    clicked_points = []  # list of (u, v) in display coords

    _prev_cam = None     # (baseline, convergence, distance) for dirty check
    _anim_dir = 1.0      # animation direction

state = State()


# =============================================================================
# SVD Bar Chart
# =============================================================================

def draw_svd_bars(F, E, width=350, height=180):
    """Render side-by-side SVD bar charts for F and E."""
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    half_w = width // 2

    sv_F = np.linalg.svd(F, compute_uv=False)
    sv_E = np.linalg.svd(E, compute_uv=False)

    # Normalize to max across both
    max_sv = max(sv_F[0], sv_E[0], 1e-10)

    def draw_bars(svs, x_off, label, color, label_color):
        bar_w = 30
        max_h = height - 60
        y_base = height - 25

        cv2.putText(canvas, label, (x_off + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)

        for i, sv in enumerate(svs):
            bx = x_off + 15 + i * (bar_w + 12)
            bh = int(sv / max_sv * max_h) if max_sv > 1e-10 else 0
            bh = max(bh, 1)
            cv2.rectangle(canvas, (bx, y_base - bh), (bx + bar_w, y_base),
                          color, -1)
            cv2.rectangle(canvas, (bx, y_base - bh), (bx + bar_w, y_base),
                          (200, 200, 200), 1)
            # Value label
            val_str = f"{sv / max_sv:.2f}" if max_sv > 1e-10 else "0"
            cv2.putText(canvas, val_str, (bx, y_base + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            # Sigma label
            cv2.putText(canvas, f"s{i+1}", (bx + 5, y_base - bh - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Separator
    cv2.line(canvas, (half_w, 5), (half_w, height - 5), (80, 80, 80), 1)

    draw_bars(sv_F, 0, "F  (rank 2)", (100, 200, 255), (100, 200, 255))
    draw_bars(sv_E, half_w, "E  (s1=s2)", (100, 255, 200), (100, 255, 200))

    return canvas


# =============================================================================
# Matrix Display
# =============================================================================

def draw_matrix_panel(M1, M2, F, E, B, a, lam, width=350, height=280):
    """Render the matrix derivation chain as a CV2 image."""
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    y = 18
    ln = 16  # line height
    fs = 0.35
    white = (200, 200, 200)
    yellow = (100, 220, 255)
    green = (100, 255, 150)
    cyan = (255, 200, 100)

    def put(text, x, y, color=white):
        cv2.putText(canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

    put("M1 = K[R1|t1]     M2 = K[R2|t2]", 8, y, yellow)
    y += ln + 4

    put("F = [e2]x  M2  M1+", 8, y, green)
    put("(rank 2, epipolar geometry)", 8, y + ln, (120, 120, 120))
    y += 2 * ln + 4

    put("E = K^T  F  K", 8, y, cyan)
    put("(calibrated F, SVD: s,s,0)", 8, y + ln, (120, 120, 120))
    y += 2 * ln + 8

    cv2.line(canvas, (8, y - 2), (width - 8, y - 2), (80, 80, 80), 1)
    y += 6

    put("B = A2  A1^-1", 8, y, green)
    put("(homography at infinity)", 8, y + ln, (120, 120, 120))
    y += 2 * ln + 4

    put("a = A2  C1  + t2", 8, y, green)
    put("(epipole-like term)", 8, y + ln, (120, 120, 120))
    y += 2 * ln + 4

    put(f"H(lam) = {lam:.2f} B + a e3^T", 8, y, yellow)
    put("(depth-dependent, rank 3)", 8, y + ln, (120, 120, 120))

    return canvas


# =============================================================================
# 3D Overview Rendering
# =============================================================================

def render_3d_overview(lam, clicked_points):
    """Render the 3D overview with cameras, depth plane, and rays."""
    ov_Rt = OvCam.make_Rt()

    frustum_ref = make_frustum_mesh(K_SHARED, cam.Rt_ref,
                                     IMG_W, IMG_H, near=0.3, far=6.0)
    frustum_other = make_frustum_mesh(K_SHARED, cam.Rt_other,
                                       IMG_W, IMG_H, near=0.3, far=6.0)
    axes = make_axis_mesh(origin=(0, 0, 0), length=1.0)

    extra = frustum_ref + frustum_other + axes

    # Add ray-plane intersections for clicked points
    if clicked_points:
        A_ref = cam.M_ref[:, :3]
        t_ref = cam.M_ref[:, 3]
        C_ref = np.linalg.solve(A_ref, -t_ref)

        for i, (px, py) in enumerate(clicked_points):
            col_bgr = POINT_COLORS[i % len(POINT_COLORS)]
            # Convert BGR to RGB for 3D renderer
            col_rgb = (col_bgr[2], col_bgr[1], col_bgr[0])

            # Ray direction: un-flip y, then back-project
            uy = IMG_H - 1 - py
            ray_dir = np.linalg.solve(A_ref, np.array([px, uy, 1.0]))
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # The depth plane is at distance lam from the reference camera
            # along its principal axis. For simplicity, compute the 3D point
            # via H(λ) back-projection: the point on the ray at depth λ.
            # P = C_ref + λ * A_ref^{-1} @ [px, uy, 1]
            P = C_ref + lam * np.linalg.solve(A_ref, np.array([px, uy, 1.0]))

            extra.append(make_sphere(center=tuple(P), radius=0.08,
                                     color=col_rgb))

    ov_img = render_scene(_SCENE + extra, _OV_K, ov_Rt,
                          OVERVIEW_SIZE, OVERVIEW_SIZE)
    return ov_img


# =============================================================================
# Drawing the 4 image panels
# =============================================================================

def draw_view1(base_img, clicked_points, label="View 1 (click here)"):
    """Draw View 1 with clicked point dots."""
    canvas = base_img.copy()
    for i, (px, py) in enumerate(clicked_points):
        col = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(canvas, (int(px), int(py)), 7, col, -1)
        cv2.circle(canvas, (int(px), int(py)), 7, (255, 255, 255), 1)
    cv2.putText(canvas, label, (8, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_view2_F(base_img, clicked_points, F_disp, label="F: line constraint"):
    """Draw View 2 F row — epipolar lines for each clicked point."""
    canvas = base_img.copy()
    for i, (px, py) in enumerate(clicked_points):
        col = POINT_COLORS[i % len(POINT_COLORS)]
        x_h = np.array([px, py, 1.0])
        l = F_disp @ x_h
        p1, p2 = epipolar_line_endpoints(l, IMG_W, IMG_H)
        if p1 is not None:
            cv2.line(canvas, p1, p2, col, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_view2_H(base_img, clicked_points, F_disp, B, a, lam,
                 label="H(lam): point at depth"):
    """Draw View 2 H(λ) row — dimmed epipolar lines + bright dots."""
    canvas = base_img.copy()
    H = compute_H_lam(B, a, lam)

    for i, (px, py) in enumerate(clicked_points):
        col = POINT_COLORS[i % len(POINT_COLORS)]
        # Dimmed epipolar line
        dim_col = tuple(max(0, c // 3) for c in col)
        x_h = np.array([px, py, 1.0])
        l = F_disp @ x_h
        p1, p2 = epipolar_line_endpoints(l, IMG_W, IMG_H)
        if p1 is not None:
            cv2.line(canvas, p1, p2, dim_col, 1, cv2.LINE_AA)

        # Projected dot via H(λ) — sub-pixel precision (shift=4 → 1/16 px)
        proj = project_H(H, px, py, IMG_H)
        if proj is not None:
            sx = int(round(proj[0] * 16))
            sy = int(round(proj[1] * 16))
            cv2.circle(canvas, (sx, sy), 8 << 4, col, -1, cv2.LINE_AA, shift=4)
            cv2.circle(canvas, (sx, sy), 8 << 4, (255, 255, 255), 2,
                       cv2.LINE_AA, shift=4)

    lbl = f"H(lam={lam:.1f}): point at depth"
    cv2.putText(canvas, lbl, (8, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_view1_depth(base_img, clicked_points, B, a, lam,
                     label="View 1 (depth info)"):
    """Draw View 1 bottom-left — show clicked points with depth annotations."""
    canvas = base_img.copy()
    H = compute_H_lam(B, a, lam)

    for i, (px, py) in enumerate(clicked_points):
        col = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(canvas, (int(px), int(py)), 7, col, -1)
        cv2.circle(canvas, (int(px), int(py)), 7, (255, 255, 255), 1)

        # Raycast to get true depth at this pixel
        hit = raycast_scene(_SCENE, K_SHARED, cam.Rt_ref,
                            int(px), int(py), IMG_H)
        if hit is not None:
            depth_str = f"z={hit:.1f}"
            match_str = "<<" if abs(lam - hit) < 0.3 else ""
            cv2.putText(canvas, depth_str + match_str,
                        (int(px) + 12, int(py) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    cv2.putText(canvas, label, (8, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# =============================================================================
# Mouse Handlers
# =============================================================================

def on_mouse_wheel(sender, app_data):
    """Zoom orbit camera on scroll wheel."""
    if state.show_3d and dpg.is_item_hovered("overview_img"):
        OvCam.handle_scroll(app_data)


# =============================================================================
# Main
# =============================================================================

def main():
    dpg.create_context()
    load_fonts()

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)

    # ── Textures ──────────────────────────────────────────────────────────────
    with dpg.texture_registry():
        for t in ["v1_top_tex", "v2_f_tex", "v1_bot_tex", "v2_h_tex"]:
            create_blank_texture(IMG_W, IMG_H, t)
        create_blank_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, "overview_tex")
        create_blank_texture(350, 180, "svd_tex")
        create_blank_texture(350, 280, "matrix_tex")

    # ── Window ────────────────────────────────────────────────────────────────
    with dpg.window(label="F vs H(\u03bb) Demo", tag="main_window"):

        add_global_controls(DEFAULTS, state,
                            camera_callback=make_camera_callback(state),
                            guide=GUIDE_FvsH,
                            guide_title="F vs H(\u03bb)")
        dpg.add_separator()

        # ── Control panels ────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # Camera panel
            with control_panel("Camera", width=340, height=160,
                               color=(150, 200, 255)):
                with create_parameter_table():
                    add_parameter_row(
                        "Baseline", "cam_baseline_slider",
                        DEFAULTS["cam_baseline"], 0.0, 5.0,
                        make_state_updater(state, "cam_baseline"),
                        make_reset_callback(state, "cam_baseline",
                                            "cam_baseline_slider",
                                            DEFAULTS["cam_baseline"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Convergence", "cam_convergence_slider",
                        DEFAULTS["cam_convergence"], 0.0, 1.0,
                        make_state_updater(state, "cam_convergence"),
                        make_reset_callback(state, "cam_convergence",
                                            "cam_convergence_slider",
                                            DEFAULTS["cam_convergence"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Other cam Z", "cam_distance_slider",
                        DEFAULTS["cam_distance"], 1.0, 8.0,
                        make_state_updater(state, "cam_distance"),
                        make_reset_callback(state, "cam_distance",
                                            "cam_distance_slider",
                                            DEFAULTS["cam_distance"]),
                        format_str="%.1f")

            dpg.add_spacer(width=8)

            # Depth λ panel
            with control_panel("Depth \u03bb", width=300, height=160,
                               color=(220, 180, 100)):
                with create_parameter_table():
                    add_parameter_row(
                        "\u03bb", "lam_slider",
                        DEFAULTS["lam"], LAM_MIN, LAM_MAX,
                        make_state_updater(state, "lam"),
                        make_reset_callback(state, "lam", "lam_slider",
                                            DEFAULTS["lam"]),
                        format_str="%.2f")
                dpg.add_checkbox(
                    label="Animate \u03bb sweep",
                    default_value=DEFAULTS["animate"],
                    callback=lambda s, v: setattr(state, 'animate', v))

            dpg.add_spacer(width=8)

            # Display panel
            with control_panel("Display", width=240, height=160,
                               color=(150, 255, 150)):
                dpg.add_checkbox(
                    label="Show matrices",
                    default_value=DEFAULTS["show_matrices"],
                    callback=lambda s, v: setattr(state, 'show_matrices', v))
                dpg.add_checkbox(
                    label="Show SVD",
                    default_value=DEFAULTS["show_svd"],
                    callback=lambda s, v: setattr(state, 'show_svd', v))
                dpg.add_checkbox(
                    label="Show 3D overview",
                    default_value=DEFAULTS["show_3d"],
                    callback=lambda s, v: setattr(state, 'show_3d', v))

        dpg.add_separator()

        # ── Status bar ────────────────────────────────────────────────────
        dpg.add_text("Left-click View 1 to add points. Right-click to clear.",
                     tag="status_text", color=(255, 220, 100))
        dpg.add_separator()

        # ── Image panels (2 rows × 2 columns) + side panels ──────────────
        with dpg.group(horizontal=True):
            # Left: image grid
            with dpg.group():
                # Top row: F
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("View 1  (click to add points)",
                                     color=(150, 255, 150))
                        dpg.add_image("v1_top_tex", tag="v1_top_img",
                                      width=IMG_W, height=IMG_H)
                    dpg.add_spacer(width=6)
                    with dpg.group():
                        dpg.add_text("View 2: F  (epipolar lines)",
                                     color=(100, 200, 255))
                        dpg.add_image("v2_f_tex", tag="v2_f_img",
                                      width=IMG_W, height=IMG_H)
                dpg.add_spacer(height=6)
                # Bottom row: H(λ)
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("View 1  (depth info)",
                                     color=(150, 255, 150))
                        dpg.add_image("v1_bot_tex", tag="v1_bot_img",
                                      width=IMG_W, height=IMG_H)
                    dpg.add_spacer(width=6)
                    with dpg.group():
                        dpg.add_text("View 2: H(\u03bb)  (projected dots)",
                                     tag="v2_h_label",
                                     color=(220, 180, 100))
                        dpg.add_image("v2_h_tex", tag="v2_h_img",
                                      width=IMG_W, height=IMG_H)

            dpg.add_spacer(width=10)

            # Right: 3D overview + info panels
            with dpg.group():
                with dpg.group(tag="overview_group"):
                    dpg.add_text("3D Overview  (drag/scroll)",
                                 color=(150, 255, 150))
                    dpg.add_image("overview_tex", tag="overview_img",
                                  width=OVERVIEW_SIZE, height=OVERVIEW_SIZE)
                dpg.add_spacer(height=6)
                with dpg.group(tag="svd_group"):
                    dpg.add_text("Singular Values",
                                 color=(150, 255, 150))
                    dpg.add_image("svd_tex", tag="svd_img",
                                  width=350, height=180)
                dpg.add_spacer(height=6)
                with dpg.group(tag="matrix_group"):
                    dpg.add_text("Derivation Chain",
                                 color=(150, 255, 150))
                    dpg.add_image("matrix_tex", tag="matrix_img",
                                  width=350, height=280)

    setup_viewport("F vs H(\u03bb) \u2014 Fundamental Matrix vs Depth Homography",
                   1200, 1000, "main_window", lambda: None, DEFAULTS["ui_scale"])

    # ── Main loop ─────────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()

        # ── Camera param dirty check ──────────────────────────────────────
        cur_cam = (state.cam_baseline, state.cam_convergence, state.cam_distance)
        if cur_cam != state._prev_cam:
            apply_camera_params(*cur_cam)
            state._prev_cam = cur_cam

        # ── Animation ─────────────────────────────────────────────────────
        if state.animate:
            speed = 0.03
            state.lam += speed * state._anim_dir
            if state.lam >= LAM_MAX:
                state.lam = LAM_MAX
                state._anim_dir = -1.0
            elif state.lam <= LAM_MIN:
                state.lam = LAM_MIN
                state._anim_dir = 1.0
            if dpg.does_item_exist("lam_slider"):
                dpg.set_value("lam_slider", state.lam)

        lam = float(state.lam)

        # ── Click handling ────────────────────────────────────────────────
        if dpg.is_mouse_button_clicked(1):
            state.clicked_points.clear()

        if dpg.is_mouse_button_clicked(0):
            # Check if click is on View 1 top image
            for img_tag in ("v1_top_img", "v1_bot_img"):
                if dpg.is_item_hovered(img_tag):
                    coords = get_image_pixel_coords(img_tag, IMG_W, IMG_H)
                    if coords is not None:
                        x = np.clip(coords[0], 0, IMG_W - 1)
                        y = np.clip(coords[1], 0, IMG_H - 1)
                        if len(state.clicked_points) >= MAX_POINTS:
                            state.clicked_points.pop(0)
                        state.clicked_points.append((x, y))
                    break

        # ── Orbit camera mouse handling ───────────────────────────────────
        if state.show_3d and dpg.does_item_exist("overview_img"):
            OvCam.poll_drag("overview_img")

        # ── Compute math ─────────────────────────────────────────────────
        F_disp = compute_F_display(cam.M_ref, cam.M_other, IMG_H)
        F_raw = compute_F_from_cameras(cam.M_ref, cam.M_other)
        E = compute_E(F_raw, K_SHARED)
        B, a, C_ref = decompose_H(cam.M_ref, cam.M_other)

        # ── Draw image panels ─────────────────────────────────────────────
        v1_top = draw_view1(cam.ref_img, state.clicked_points)
        v2_f = draw_view2_F(cam.other_img, state.clicked_points, F_disp)
        v1_bot = draw_view1_depth(cam.ref_img, state.clicked_points, B, a, lam)
        v2_h = draw_view2_H(cam.other_img, state.clicked_points,
                            F_disp, B, a, lam)

        dpg.set_value("v1_top_tex", convert_cv_to_dpg(v1_top))
        dpg.set_value("v2_f_tex", convert_cv_to_dpg(v2_f))
        dpg.set_value("v1_bot_tex", convert_cv_to_dpg(v1_bot))
        dpg.set_value("v2_h_tex", convert_cv_to_dpg(v2_h))

        # Update H(λ) label
        dpg.set_value("v2_h_label",
                      f"View 2: H(\u03bb={lam:.1f})  (projected dots)")

        # ── Side panels ───────────────────────────────────────────────────
        dpg.configure_item("overview_group", show=state.show_3d)
        dpg.configure_item("svd_group", show=state.show_svd)
        dpg.configure_item("matrix_group", show=state.show_matrices)

        if state.show_3d:
            ov_img = render_3d_overview(lam, state.clicked_points)
            dpg.set_value("overview_tex", convert_cv_to_dpg(ov_img))

        if state.show_svd:
            svd_img = draw_svd_bars(F_raw, E)
            dpg.set_value("svd_tex", convert_cv_to_dpg(svd_img))

        if state.show_matrices:
            mat_img = draw_matrix_panel(cam.M_ref, cam.M_other,
                                        F_raw, E, B, a, lam)
            dpg.set_value("matrix_tex", convert_cv_to_dpg(mat_img))

        # ── Status text ───────────────────────────────────────────────────
        n = len(state.clicked_points)
        status = f"{n}/{MAX_POINTS} points"
        if state.animate:
            status += f"  |  Animating lam={lam:.2f}"
        dpg.set_value("status_text", status)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
