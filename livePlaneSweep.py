"""
Live Plane Sweep Stereo Demo — Depth-Dependent Homography H(λ)
CSCI 1430 - Brown University

Interactive demonstration of plane sweep stereo using two calibrated cameras.

Core math (HW3 Task 2):
  B = A_other @ inv(A_ref)          [homography at infinity]
  a = A_other @ C_ref + t_other     [epipole in other camera]
  H(λ) = λ·B + outer(a, [0,0,1])   [depth-dependent homography]

Scene: a synthetic checkerboard plane at depth λ_true is rendered from two
camera viewpoints.  H(λ) maps reference pixels → other pixels assuming scene
depth λ.  When λ = λ_true, the warp perfectly aligns the two views (NCC ≈ 1).

Panels:
  Left   — Reference image (fixed)
  Centre — Other image warped by H(λ) (changes with slider)
  Right  — NCC vs. λ curve (pre-computed at startup, current λ highlighted)

Status bar: current NCC and residual between warped and reference.
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt,
    fov_to_focal, render_scene,
    make_frustum_mesh, make_axis_mesh,
    make_sphere, make_cube,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
)


# =============================================================================
# Constants / Defaults
# =============================================================================

DEFAULTS = {
    "lam": 3.0,       # current depth hypothesis λ (slider)
    "ui_scale": 1.5,
}

IMG_W, IMG_H = 400, 400
OVERVIEW_SIZE = 400
LAM_MIN, LAM_MAX = 0.5, 8.0
LAM_TRUE = 3.0          # true scene depth parameter
N_SWEEP = 80            # number of depth hypotheses in the pre-computed curve

# Camera setup
K_SHARED = build_intrinsic(
    fx=350.0, fy=350.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)
_Rt_ref   = make_lookat_Rt(eye=np.array([0.0,  0.5, 5.0]),
                            target=np.array([0.0, 0.0, 0.0]))
_Rt_other = make_lookat_Rt(eye=np.array([2.5,  0.5, 5.0]),
                            target=np.array([0.0, 0.0, 0.0]))
M_REF   = K_SHARED @ _Rt_ref
M_OTHER = K_SHARED @ _Rt_other

_OV_Rt = make_lookat_Rt(eye=np.array([4.0, 4.0, 8.0]),
                         target=np.array([0.0, 0.0, 0.5]))
_OV_K  = build_intrinsic(fov_to_focal(50, OVERVIEW_SIZE), fov_to_focal(50, OVERVIEW_SIZE),
                          0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2)


# =============================================================================
# Image Synthesis (called once at startup)
# =============================================================================

def _backproject_to_depth(M, lam, img_w, img_h):
    """For each pixel, back-project along its ray to depth parameter lam.

    Returns pts3d (H*W, 3) — world coordinates at depth lam.
    The 'depth' lam parameterises distance along A^{-1}[u,v,1].
    """
    A = M[:, :3]
    t = M[:, 3]
    C = np.linalg.solve(A, -t)          # camera center (3,)

    u, v = np.meshgrid(np.arange(img_w, dtype=float),
                       np.arange(img_h, dtype=float))
    uv1 = np.stack([u.ravel(), v.ravel(), np.ones(img_w * img_h)], axis=1)  # N×3
    dirs = np.linalg.solve(A, uv1.T).T  # N×3 world ray directions

    pts3d = C + lam * dirs              # N×3
    return pts3d


def _render_checker_view(M, lam_true, img_w, img_h, sq_size=0.8):
    """Render a synthetic checkerboard plane viewed through camera M at depth lam_true."""
    pts3d = _backproject_to_depth(M, lam_true, img_w, img_h)

    # Checker from world X, Y position
    xi = np.floor(pts3d[:, 0] / sq_size).astype(int)
    yi = np.floor(pts3d[:, 1] / sq_size).astype(int)
    even = ((xi + yi) % 2 == 0)

    # Two distinct colours (BGR)
    img_flat = np.where(
        even[:, None],
        np.array([200, 200, 60],  dtype=np.uint8),   # warm ivory
        np.array([40,  80,  180], dtype=np.uint8),   # blue
    )

    # Depth-in-front check: only colour pixels where camera faces the plane
    in_front = (pts3d @ _Rt_ref[2, :3] + _Rt_ref[2, 3] > 0).reshape(-1, 1)
    img_flat = np.where(in_front, img_flat, 0)

    return img_flat.reshape(img_h, img_w, 3).astype(np.uint8)


# Pre-render the two views (done once)
REF_IMG   = _render_checker_view(M_REF,   LAM_TRUE, IMG_W, IMG_H)
OTHER_IMG = _render_checker_view(M_OTHER, LAM_TRUE, IMG_W, IMG_H)


# =============================================================================
# Plane Sweep Math
# =============================================================================

def compute_H_lam(M_ref, M_other, lam):
    """H(λ) = λ·B + outer(a, e₃ᵀ)  [HW3 Task 2 — compute_depth_homography]."""
    A_ref   = M_ref[:, :3]
    t_ref   = M_ref[:, 3]
    A_other = M_other[:, :3]
    t_other = M_other[:, 3]

    C_ref = np.linalg.solve(A_ref, -t_ref)      # (3,) camera center
    B     = A_other @ np.linalg.inv(A_ref)       # (3×3) homography at infinity
    a     = A_other @ C_ref + t_other            # (3,) epipole-like term
    e3    = np.array([0.0, 0.0, 1.0])

    H = lam * B + np.outer(a, e3)               # (3×3)
    return H


def warp_other_to_ref(other_img, H, img_w, img_h):
    """Warp other_img by H(λ) into reference coordinates (WARP_INVERSE_MAP)."""
    return cv2.warpPerspective(
        other_img, H, (img_w, img_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def compute_ncc_score(ref, warped):
    """Compute global normalised cross-correlation between ref and warped (valid pixels)."""
    valid = (warped.sum(axis=2) > 0)   # pixels that were filled
    if valid.sum() < 100:
        return -1.0
    r = ref[valid].astype(float)
    w = warped[valid].astype(float)
    r -= r.mean()
    w -= w.mean()
    denom = (np.linalg.norm(r) * np.linalg.norm(w))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(r.ravel(), w.ravel()) / denom)


# Pre-compute NCC curve at startup
_LAM_CURVE = np.linspace(LAM_MIN, LAM_MAX, N_SWEEP)
_NCC_CURVE = np.array([
    compute_ncc_score(REF_IMG, warp_other_to_ref(OTHER_IMG,
                      compute_H_lam(M_REF, M_OTHER, lam), IMG_W, IMG_H))
    for lam in _LAM_CURVE
])


# =============================================================================
# NCC Curve Canvas
# =============================================================================

def draw_ncc_curve(lam_current, ncc_current, width=400, height=300):
    """Draw the NCC vs. λ curve with the current λ highlighted."""
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    # Axes padding
    px, py = 50, 20
    pw = width - px - 10
    ph = height - py - 40

    # Draw axes
    cv2.line(canvas, (px, py), (px, py + ph), (150, 150, 150), 1)
    cv2.line(canvas, (px, py + ph), (px + pw, py + ph), (150, 150, 150), 1)

    # Axis labels
    cv2.putText(canvas, "NCC", (2, py + ph // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"lam (true={LAM_TRUE:.1f})", (px + pw // 2 - 40, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # Y-axis ticks: -0.5 to 1.0
    ncc_lo, ncc_hi = -0.3, 1.05
    for tick in [-0.2, 0.0, 0.2, 0.5, 0.8, 1.0]:
        ty = int(py + ph - (tick - ncc_lo) / (ncc_hi - ncc_lo) * ph)
        cv2.line(canvas, (px - 4, ty), (px, ty), (100, 100, 100), 1)
        cv2.putText(canvas, f"{tick:.1f}", (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    # X-axis ticks
    for tick in [1, 2, 3, 4, 5, 6, 7]:
        tx = int(px + (tick - LAM_MIN) / (LAM_MAX - LAM_MIN) * pw)
        cv2.line(canvas, (tx, py + ph), (tx, py + ph + 4), (100, 100, 100), 1)
        cv2.putText(canvas, str(tick), (tx - 4, height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    def to_canvas(lam, ncc):
        cx = int(px + (lam - LAM_MIN) / (LAM_MAX - LAM_MIN) * pw)
        cy = int(py + ph - (ncc - ncc_lo) / (ncc_hi - ncc_lo) * ph)
        return cx, cy

    # Draw curve (pre-computed)
    pts_curve = [to_canvas(l, n) for l, n in zip(_LAM_CURVE, _NCC_CURVE)]
    for i in range(len(pts_curve) - 1):
        cv2.line(canvas, pts_curve[i], pts_curve[i + 1], (100, 220, 100), 2)

    # True depth vertical line
    tx_true, _ = to_canvas(LAM_TRUE, 0)
    cv2.line(canvas, (tx_true, py), (tx_true, py + ph), (80, 80, 180), 1)
    cv2.putText(canvas, f"true", (tx_true + 2, py + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 220), 1)

    # Current λ vertical line + dot
    cx_cur, cy_cur = to_canvas(lam_current, max(ncc_lo, min(ncc_hi, ncc_current)))
    cv2.line(canvas, (cx_cur, py), (cx_cur, py + ph), (60, 200, 220), 1)
    cv2.circle(canvas, (cx_cur, cy_cur), 5, (60, 200, 220), -1)
    cv2.putText(canvas, f"λ={lam_current:.2f}", (cx_cur + 4, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 200, 220), 1)

    return canvas  # BGR


# =============================================================================
# State
# =============================================================================

class State:
    lam = DEFAULTS["lam"]

state = State()


# =============================================================================
# Callbacks
# =============================================================================

def reset_all():
    state.lam = DEFAULTS["lam"]
    if dpg.does_item_exist("lam_slider"):
        dpg.set_value("lam_slider", DEFAULTS["lam"])


# =============================================================================
# Main
# =============================================================================

def main():
    dpg.create_context()

    load_fonts()

    # Textures
    with dpg.texture_registry(tag="texture_registry"):
        blank_rgb = [0.0] * (IMG_W * IMG_H * 4)
        for tag in ["ref_tex", "warped_tex"]:
            dpg.add_raw_texture(IMG_W, IMG_H, list(blank_rgb),
                                format=dpg.mvFormat_Float_rgba, tag=tag)
        blank_curve = [0.0] * (400 * 300 * 4)
        dpg.add_raw_texture(400, 300, blank_curve,
                            format=dpg.mvFormat_Float_rgba, tag="curve_tex")
        blank_ov = [0.0] * (OVERVIEW_SIZE * OVERVIEW_SIZE * 4)
        dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, blank_ov,
                            format=dpg.mvFormat_Float_rgba, tag="overview_tex")

    with dpg.window(label="Plane Sweep Stereo Demo", tag="main_window"):

        # ── Top controls ──────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="UI Scale",
                items=["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"],
                default_value=str(DEFAULTS["ui_scale"]), width=80,
                callback=lambda s, v: dpg.set_global_font_scale(float(v)),
            )
            dpg.add_spacer(width=20)
            dpg.add_button(label="Reset All", callback=lambda: reset_all())
            dpg.add_spacer(width=20)
            dpg.add_text(
                f"  True depth: λ={LAM_TRUE:.1f}  |  "
                f"Scene: checkerboard plane  |  "
                f"H(λ) = λB + outer(a, e₃ᵀ)",
                color=(200, 200, 120),
            )

        dpg.add_separator()

        # ── λ slider ─────────────────────────────────────────────────────────
        with create_parameter_table():
            dpg.add_table_column(width_fixed=True, init_width_or_weight=160)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=300)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=30)

            add_parameter_row(
                "Depth λ (sweep)", "lam_slider",
                DEFAULTS["lam"], LAM_MIN, LAM_MAX,
                make_state_updater(state, "lam"),
                make_reset_callback(state, "lam", "lam_slider", DEFAULTS["lam"]),
                format_str="%.2f",
                width=300,
            )

        dpg.add_separator()

        # ── Status ───────────────────────────────────────────────────────────
        dpg.add_text("", tag="status_text", color=(255, 220, 100))

        dpg.add_separator()

        # ── Image panels ─────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Reference image  (fixed)", color=(150, 255, 150))
                dpg.add_image("ref_tex", tag="ref_img", width=IMG_W, height=IMG_H)

            dpg.add_spacer(width=15)
            with dpg.group():
                dpg.add_text("Warped other  [cv2.warpPerspective(other, H(λ), WARP_INVERSE_MAP)]",
                             color=(150, 200, 255))
                dpg.add_image("warped_tex", tag="warped_img", width=IMG_W, height=IMG_H)

            dpg.add_spacer(width=15)
            with dpg.group():
                dpg.add_text("NCC vs. λ  (pre-computed sweep)", color=(220, 180, 100))
                dpg.add_image("curve_tex", tag="curve_img", width=400, height=300)

        dpg.add_separator()

        # ── 3D overview row ───────────────────────────────────────────────────
        with dpg.group():
            dpg.add_text("3D Overview  (ref=green frustum, other=orange frustum)",
                         color=(150, 255, 150))
            dpg.add_image("overview_tex", tag="overview_img",
                          width=OVERVIEW_SIZE, height=OVERVIEW_SIZE)

    setup_viewport("Plane Sweep Stereo Demo — H(λ)", 1320, 870,
                   "main_window", lambda: None, DEFAULTS["ui_scale"])

    # Upload fixed reference image once
    dpg.set_value("ref_tex", convert_cv_to_dpg(REF_IMG))

    # ── Main loop ─────────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        lam = float(state.lam)

        # Compute H(λ) and warp
        H = compute_H_lam(M_REF, M_OTHER, lam)
        warped = warp_other_to_ref(OTHER_IMG, H, IMG_W, IMG_H)

        # NCC between ref and warped
        ncc = compute_ncc_score(REF_IMG, warped)

        # 3D overview: two frustums + world axes
        frustum_ref   = make_frustum_mesh(K_SHARED, _Rt_ref,   IMG_W, IMG_H, near=0.3, far=6.0)
        frustum_other = make_frustum_mesh(K_SHARED, _Rt_other, IMG_W, IMG_H, near=0.3, far=6.0)
        axes = make_axis_mesh(origin=(0, 0, 0), length=1.0)
        # Colour the other frustum differently: reuse render with orange spheres
        # (frustum meshes come with their own colours; we just add both)
        ov_img = render_scene(frustum_ref + frustum_other + axes,
                              _OV_K, _OV_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE)

        # NCC curve canvas
        curve = draw_ncc_curve(lam, ncc)

        # Update textures
        dpg.set_value("warped_tex",  convert_cv_to_dpg(warped))
        dpg.set_value("curve_tex",   convert_cv_to_dpg(curve))
        dpg.set_value("overview_tex", convert_cv_to_dpg(ov_img))

        # Status
        ncc_str = f"{ncc:.4f}" if ncc >= -0.5 else "invalid"
        dpg.set_value(
            "status_text",
            f"λ = {lam:.3f}  |  NCC = {ncc_str}  |  "
            f"True λ = {LAM_TRUE:.1f}  |  "
            f"{'>>> PEAK NEAR HERE <<<' if abs(lam - LAM_TRUE) < 0.3 else ''}",
        )

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
