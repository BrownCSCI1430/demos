"""
Web frame module for liveDLT demo.
DLT Explorer: Homography & Fundamental Matrix estimation via SVD.
"""

import numpy as np
import cv2

from liveDLT import (
    IMG_W, IMG_H, OVERVIEW_SIZE, LAM_TRUE,
    K_SHARED, _Rt1, _Rt2, M1, M2,
    PLANE_VIS, VOLUME_VIS,
    PLANE_PTS1, PLANE_PTS2, VOLUME_PTS1, VOLUME_PTS2,
    PLANE_NOISE1, PLANE_NOISE2, VOLUME_NOISE1, VOLUME_NOISE2,
    CHECKER_IMG1, CHECKER_IMG2, SCENE_IMG1, SCENE_IMG2,
    _B, _a, _OV_K, _SCENE,
    estimate_H_dlt, compose_H_lambda,
    estimate_F_dlt,
    warp_image,
    render_svd_panel,
    draw_points, draw_epipolar_lines_on_img, draw_epipole, draw_residuals,
    _depth_color,
    OvCam,
    GUIDE_HOMOGRAPHY, GUIDE_FUNDAMENTAL,
)
from utils.demo_3d import (
    render_scene, make_frustum_mesh, make_axis_mesh,
    make_sphere, make_ground_grid,
)

# ── Module-level state for orbit drag ──
_dragging = False
_drag_prev = None

SVD_W, SVD_H = 500, 250


WEB_CONFIG = {
    "title": "DLT Explorer",
    "description": (
        "Explore DLT estimation for homographies and fundamental matrices. "
        "See how SVD solves the linear system, and interactively decompose "
        "H(\u03bb) = \u03bbB + ae\u2083\u1d40. Drag the 3D view to orbit."
    ),
    "outputs": [
        {"id": "view1",    "label": "Camera 1",          "width": IMG_W, "height": IMG_H},
        {"id": "view2",    "label": "Camera 2 / Result",  "width": IMG_W, "height": IMG_H},
        {"id": "overview", "label": "3D Overview",         "width": OVERVIEW_SIZE, "height": OVERVIEW_SIZE},
        {"id": "svd",      "label": "SVD Decomposition",  "width": SVD_W, "height": SVD_H},
    ],
    "controls": {
        "mode": {
            "type": "choice",
            "options": ["Homography", "Fundamental Matrix"],
            "default": "Homography",
            "label": "Mode",
        },
        "noise_px": {
            "group": "Input Data",
            "type": "float", "min": 0.0, "max": 15.0, "step": 0.1,
            "default": 0.0, "label": "Noise (px)",
        },
        "n_points": {
            "group": "Input Data",
            "type": "int", "min": 4, "max": 36, "step": 1,
            "default": 12, "label": "# Points", "format": "d",
        },
        "use_hartley": {
            "group": "Solver",
            "type": "bool", "default": False, "label": "Hartley Normalize",
        },
        "show_warp": {
            "group": "Homography",
            "type": "bool", "default": True, "label": "Show Warp",
            "visible_when": {"mode": ["Homography"]},
        },
        "decompose_H": {
            "group": "Homography",
            "type": "bool", "default": False,
            "label": "Decompose H = \u03bbB + ae\u2083\u1d40",
            "visible_when": {"mode": ["Homography"]},
        },
        "lambda_val": {
            "group": "Homography",
            "type": "float", "min": 0.5, "max": 8.0, "step": 0.05,
            "default": LAM_TRUE, "label": "\u03bb (depth)",
            "visible_when": {"mode": ["Homography"], "decompose_H": [True]},
        },
        "epipole_coeff": {
            "group": "Homography",
            "type": "float", "min": 0.0, "max": 2.0, "step": 0.01,
            "default": 1.0, "label": "Epipole coeff",
            "visible_when": {"mode": ["Homography"], "decompose_H": [True]},
        },
        "show_epipolar_lines": {
            "group": "Epipolar Geometry",
            "type": "bool", "default": True, "label": "Epipolar Lines",
            "visible_when": {"mode": ["Fundamental Matrix"]},
        },
        "show_epipoles": {
            "group": "Epipolar Geometry",
            "type": "bool", "default": True, "label": "Epipoles",
            "visible_when": {"mode": ["Fundamental Matrix"]},
        },
        "highlight_idx": {
            "group": "Epipolar Geometry",
            "type": "int", "min": 0, "max": 35, "step": 1,
            "default": 0, "label": "Highlight pt #", "format": "d",
            "visible_when": {"mode": ["Fundamental Matrix"]},
        },
        "show_svd_A": {
            "group": "Inspect",
            "type": "bool", "default": True, "label": "Show SVD(A)",
        },
        "show_svd_F": {
            "group": "Inspect",
            "type": "bool", "default": False,
            "label": "Show SVD(F\u0302)",
            "visible_when": {"mode": ["Fundamental Matrix"]},
        },
    },
    "mouse": ["overview"],
    "layout": {
        "rows": [
            ["view1", "view2", "overview"],
            ["svd"],
        ],
    },
    "guide": (
        GUIDE_HOMOGRAPHY
        + [{"title": "\u2500\u2500\u2500 Fundamental Matrix \u2500\u2500\u2500", "body": ""}]
        + GUIDE_FUNDAMENTAL
    ),
}


def web_button(button_id):
    """Handle button clicks."""
    if button_id == "reset_orbit":
        OvCam.reset()


def web_mouse(event):
    """Handle mouse drag/scroll on 3D overview for orbit camera."""
    global _dragging, _drag_prev

    if event["canvas"] != "overview":
        return

    if event["type"] == "mousedown" and event["button"] == 0:
        _dragging = True
        _drag_prev = (event["x"], event["y"])

    elif event["type"] == "mouseup":
        _dragging = False
        _drag_prev = None

    elif event["type"] == "mousemove" and _dragging and _drag_prev is not None:
        dx = event["x"] - _drag_prev[0]
        dy = event["y"] - _drag_prev[1]
        _drag_prev = (event["x"], event["y"])
        OvCam.az += dx * 0.01
        OvCam.el = float(np.clip(OvCam.el - dy * 0.01, -1.4, 1.4))

    elif event["type"] == "wheel":
        OvCam.radius = float(np.clip(
            OvCam.radius + event["delta_y"] * 0.01, 3.0, 30.0))


def web_frame(state):
    """Compute one frame of the DLT demo."""
    is_homography = state["mode"] == "Homography"
    noise_px = state["noise_px"]
    n_req = state["n_points"]
    use_hartley = state["use_hartley"]

    # Select points and add noise
    if is_homography:
        n_avail = len(PLANE_VIS)
        n = min(max(4, n_req), n_avail)
        pts3d = PLANE_VIS[:n]
        pts1_true = PLANE_PTS1[:n]
        pts2_true = PLANE_PTS2[:n]
        pts1 = pts1_true + PLANE_NOISE1[:n] * noise_px
        pts2 = pts2_true + PLANE_NOISE2[:n] * noise_px
        base1, base2 = CHECKER_IMG1.copy(), CHECKER_IMG2.copy()
    else:
        n_avail = len(VOLUME_VIS)
        n = min(max(8, n_req), n_avail)
        pts3d = VOLUME_VIS[:n]
        pts1_true = VOLUME_PTS1[:n]
        pts2_true = VOLUME_PTS2[:n]
        pts1 = pts1_true + VOLUME_NOISE1[:n] * noise_px
        pts2 = pts2_true + VOLUME_NOISE2[:n] * noise_px
        base1, base2 = SCENE_IMG1.copy(), SCENE_IMG2.copy()

    # Camera 1 view
    view1 = base1.copy()
    draw_points(view1, pts1, pts3d, radius=5)

    status_lines = []
    svd_img = np.full((SVD_H, SVD_W, 3), 25, dtype=np.uint8)

    if is_homography:
        try:
            H, A, U, S, Vt, cond_A, fit_resid = estimate_H_dlt(
                pts1, pts2, use_hartley=use_hartley)

            view2 = base2.copy()

            decompose = state.get("decompose_H", False)
            show_warp = state.get("show_warp", True)

            if decompose:
                lam = state.get("lambda_val", LAM_TRUE)
                epi_coeff = state.get("epipole_coeff", 1.0)
                H_decomp = compose_H_lambda(_B, _a, lam, epi_coeff)
                warped = warp_image(base1, H_decomp, IMG_W, IMG_H)
            else:
                warped = warp_image(base1, H, IMG_W, IMG_H)

            if show_warp:
                mask = warped.any(axis=2)
                view2[mask] = cv2.addWeighted(
                    view2, 0.3, warped, 0.7, 0)[mask]

            draw_points(view2, pts2, pts3d, radius=5)

            # Reprojection error
            pts2_est = np.zeros_like(pts1)
            for i in range(n):
                p = H @ np.array([pts1[i, 0], pts1[i, 1], 1.0])
                pts2_est[i] = p[:2] / p[2]
            reproj = np.mean(np.linalg.norm(pts2_est - pts2_true, axis=1))
            draw_residuals(view2, pts2_true, pts2_est)

            status_lines.append(
                f"Homography | N={n} | Noise={noise_px:.1f}px | "
                f"Hartley: {'ON' if use_hartley else 'OFF'}")
            status_lines.append(
                f"cond(A)={cond_A:.3g} | fit={fit_resid:.4g} | "
                f"reproj={reproj:.2f}px")
            if decompose:
                status_lines.append(
                    f"H(\u03bb)={lam:.2f}\u00b7B + {epi_coeff:.2f}\u00b7ae\u2083\u1d40  "
                    f"(true \u03bb={LAM_TRUE:.1f})")

            if state.get("show_svd_A", True):
                svd_img = render_svd_panel(
                    U, S, Vt, SVD_W, SVD_H,
                    title="SVD of A (2Nx9 DLT system)",
                    highlight_null_col_V=True)

        except Exception as e:
            view2 = base2.copy()
            status_lines.append(f"Error: {e}")

    else:  # Fundamental Matrix
        try:
            result = estimate_F_dlt(pts1, pts2, use_hartley=use_hartley)
            F = result["F"]
            e1, e2 = result["e1"], result["e2"]

            view2 = base2.copy()

            show_lines = state.get("show_epipolar_lines", True)
            show_epi = state.get("show_epipoles", True)
            hi_idx = state.get("highlight_idx", 0)

            if show_lines:
                draw_epipolar_lines_on_img(
                    view2, F, pts1, highlight_idx=hi_idx)
                draw_epipolar_lines_on_img(
                    view1, F.T, pts2, color=(220, 180, 0),
                    highlight_idx=hi_idx)

            draw_points(view2, pts2, pts3d, radius=5)

            if show_epi:
                draw_epipole(view1, e1, color=(60, 200, 60), size=14)
                draw_epipole(view2, e2, color=(0, 140, 255), size=14)

            status_lines.append(
                f"Fundamental Matrix | N={n} | Noise={noise_px:.1f}px | "
                f"Hartley: {'ON' if use_hartley else 'OFF'}")
            status_lines.append(
                f"cond(A)={result['cond_A']:.3g} | "
                f"fit={result['fit_resid']:.4g}")
            status_lines.append(
                f"\u03c3\u2083(F\u0302)={result['S_F'][2]:.4g} "
                f"(\u2192 0 for rank-2)")
            status_lines.append(
                f"e\u2081=({e1[0]:.0f},{e1[1]:.0f}) "
                f"e\u2082=({e2[0]:.0f},{e2[1]:.0f})")

            if state.get("show_svd_F", False):
                sigma_labels = [
                    f"\u03c3\u2081={result['S_F'][0]:.3g}",
                    f"\u03c3\u2082={result['S_F'][1]:.3g}",
                    f"\u03c3\u2083={result['S_F'][2]:.3g} \u2192 0",
                ]
                svd_img = render_svd_panel(
                    result["U_F"], result["S_F"], result["Vt_F"],
                    SVD_W, SVD_H,
                    title="SVD of F\u0302 (rank enforcement)",
                    highlight_null_col_V=True,
                    highlight_null_col_U=True,
                    sigma_labels=sigma_labels)
            elif state.get("show_svd_A", True):
                svd_img = render_svd_panel(
                    result["U_A"], result["S_A"], result["Vt_A"],
                    SVD_W, SVD_H,
                    title="SVD of A (Nx9 DLT system)",
                    highlight_null_col_V=True)

        except Exception as e:
            view2 = base2.copy()
            status_lines.append(f"Error: {e}")

    # 3D Overview
    pt_meshes = [
        make_sphere(tuple(p), radius=0.06,
                    color=_depth_color(p[2], -2, 3),
                    n_lat=6, n_lon=8)
        for p in pts3d
    ]
    frust1 = make_frustum_mesh(K_SHARED, _Rt1, IMG_W, IMG_H,
                                near=0.3, far=6.0, color=(50, 200, 50))
    frust2 = make_frustum_mesh(K_SHARED, _Rt2, IMG_W, IMG_H,
                                near=0.3, far=6.0, color=(200, 150, 50))
    axes = make_axis_mesh(origin=(0, 0, 0), length=1.5)
    ground = make_ground_grid(y=0.0, extent=3.0, spacing=1.0)

    ov_Rt = OvCam.make_Rt()
    overview = render_scene(
        _SCENE + pt_meshes + frust1 + frust2 + axes + [ground],
        _OV_K, ov_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE,
    )

    return {
        "view1": view1,
        "view2": view2,
        "overview": overview,
        "svd": svd_img,
        "status": "\n".join(status_lines),
    }
