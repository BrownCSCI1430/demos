"""
Web frame module for liveCalibration demo.
Synthetic DLT calibration with interactive 3D orbit camera.
"""

import numpy as np
import cv2

from liveCalibration import (
    DEFAULTS, IMG_W, IMG_H, OVERVIEW_SIZE,
    TRUE_K, TRUE_M, _TRUE_Rt,
    VISIBLE_PTS3D, N_AVAILABLE, PROBE_PTS3D,
    _FIXED_NOISE, _PERM, _Z_PLANE,
    _COPLANAR_IDXS, _N_COPLANAR, _COPLANAR_PERM,
    _Z_MIN, _Z_MAX,
    build_A_matrix, estimate_M_dlt,
    draw_projection_canvas, _depth_color,
    OvCam, _OV_K,
    state as cal_state,
)
from utils.demo_3d import (
    render_scene, make_frustum_mesh, make_axis_mesh,
    make_sphere, make_octahedron,
)

# Module-level drag state
_dragging = False
_drag_prev = None


def _project(M, pts3d):
    """Project 3D points through camera M. Returns (N, 2) pixel coords."""
    N = len(pts3d)
    h = np.hstack([pts3d, np.ones((N, 1))])  # (N, 4)
    proj = (M @ h.T).T  # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


WEB_CONFIG = {
    "title": "Camera Calibration (DLT)",
    "description": (
        "Synthetic DLT calibration demo. Adjust noise, point count, and "
        "normalization to see their effect on camera matrix estimation. "
        "Drag the 3D view to orbit; scroll to zoom."
    ),
    "outputs": [
        {"id": "projection", "label": "2D Projection",  "width": IMG_W, "height": IMG_H},
        {"id": "overview",   "label": "3D Overview",     "width": OVERVIEW_SIZE, "height": OVERVIEW_SIZE},
    ],
    "controls": {
        "noise_px": {
            "type": "float", "min": 0.0, "max": 20.0, "step": 0.1,
            "default": DEFAULTS["noise_px"], "label": "Noise (px)",
        },
        "n_points": {
            "type": "int", "min": 6, "max": min(N_AVAILABLE, 80), "step": 1,
            "default": DEFAULTS["n_points"], "label": "# Points", "format": "d",
        },
        "offset_exp": {
            "type": "int", "min": 0, "max": 8, "step": 1,
            "default": DEFAULTS["offset_exp"], "label": "Origin Shift (10^n)", "format": "d",
        },
        "scale_exp": {
            "type": "int", "min": 0, "max": 6, "step": 1,
            "default": DEFAULTS["scale_exp"], "label": "World Scale (10^n)", "format": "d",
        },
        "use_coplanar": {
            "type": "bool", "default": False, "label": "Coplanar (z=const)",
        },
        "use_hartley": {
            "type": "bool", "default": False, "label": "Hartley Normalize",
        },
        "use_normal_eqs": {
            "type": "bool", "default": False, "label": "Solve via A^T A",
        },
        "show_A": {
            "type": "bool", "default": False, "label": "Show A Matrix",
        },
        "reset_all": {
            "type": "button", "label": "Reset All",
        },
    },
    "mouse": ["overview"],
    "layout": {"rows": [["projection", "overview"]]},
}


def web_button(button_id):
    """Handle button clicks."""
    if button_id == "reset_all":
        OvCam.reset()


def web_mouse(event):
    """Handle mouse drag/scroll on overview canvas for orbit camera."""
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
        # Orbit: dx -> azimuth, dy -> elevation
        OvCam.az += dx * 0.01
        OvCam.el = float(np.clip(OvCam.el - dy * 0.01, -1.4, 1.4))

    elif event["type"] == "wheel":
        OvCam.radius = float(np.clip(OvCam.radius + event["delta_y"] * 0.01, 2.0, 30.0))


def web_frame(state):
    noise_px = state["noise_px"]
    n_points = int(np.clip(state["n_points"], 6, N_AVAILABLE))
    offset_exp = state["offset_exp"]
    scale_exp = state["scale_exp"]
    use_coplanar = state["use_coplanar"]
    use_hartley = state["use_hartley"]
    use_normal_eqs = state["use_normal_eqs"]
    show_A = state["show_A"]

    origin_offset = 10.0 ** offset_exp if offset_exp > 0 else 0.0
    world_scale = 10.0 ** scale_exp if scale_exp > 0 else 1.0

    # Select points
    if use_coplanar:
        n_use = min(n_points, _N_COPLANAR)
        sel = _COPLANAR_IDXS[_COPLANAR_PERM[:n_use]]
        pts3d = VISIBLE_PTS3D[sel].copy()
        pts3d[:, 2] = _Z_PLANE  # Force coplanar
    else:
        sel = _PERM[:n_points]
        pts3d = VISIBLE_PTS3D[sel].copy()

    # True projections and noisy observations
    pts2d_true = _project(TRUE_M, pts3d)
    pts2d_noisy = pts2d_true + _FIXED_NOISE[sel] * noise_px

    # DLT estimation
    try:
        M_est, A, cond_A, fit_resid = estimate_M_dlt(
            pts2d_noisy, pts3d,
            use_hartley=use_hartley,
            use_normal_eqs=use_normal_eqs,
            origin_offset=origin_offset,
            world_scale=world_scale,
        )
        pts2d_est = _project(M_est, pts3d)
        residual = float(np.mean(np.linalg.norm(pts2d_est - pts2d_true, axis=1)))
    except Exception as e:
        M_est = None
        A = None
        cond_A = float("inf")
        fit_resid = float("inf")
        pts2d_est = pts2d_true
        residual = float("inf")

    # Probe points (coplanar mode)
    pts2d_probe_true = None
    pts2d_probe_est = None
    probe_residual = None
    if use_coplanar and len(PROBE_PTS3D) > 0 and M_est is not None:
        pts2d_probe_true = _project(TRUE_M, PROBE_PTS3D)
        pts2d_probe_est = _project(M_est, PROBE_PTS3D)
        probe_residual = float(np.mean(np.linalg.norm(
            pts2d_probe_est - pts2d_probe_true, axis=1)))

    # 2D projection canvas
    proj_canvas = draw_projection_canvas(
        pts2d_true, pts2d_noisy, pts2d_est, IMG_W, IMG_H,
        pts2d_probe_true, pts2d_probe_est)

    # 3D overview
    ov_Rt = OvCam.make_Rt()
    meshes = []

    # Marker spheres
    for i, idx in enumerate(sel):
        pt = VISIBLE_PTS3D[idx]
        color = _depth_color(pt[2])
        meshes.append(make_sphere(pt, radius=0.08, color=color, n_lat=6))

    # Probe octahedrons
    if use_coplanar:
        for pt in PROBE_PTS3D:
            meshes.append(make_octahedron(pt, radius=0.12, color=(255, 255, 255)))

    # True camera frustum (green)
    meshes += make_frustum_mesh(TRUE_K, _TRUE_Rt, IMG_W, IMG_H,
                                 near=0.3, far=5.0, color=(0, 180, 0))
    # Estimated camera frustum (cyan)
    if M_est is not None:
        try:
            # Decompose M_est to get K_est and Rt_est
            M3 = M_est[:, :3]
            K_est, R_est = np.linalg.qr(np.linalg.inv(M3))
            K_est = np.linalg.inv(K_est)
            R_est = np.linalg.inv(R_est)
            # Normalize K
            K_est = K_est / K_est[2, 2]
            t_est = np.linalg.solve(K_est, M_est[:, 3])
            Rt_est = np.hstack([R_est, t_est.reshape(3, 1)])
            meshes += make_frustum_mesh(K_est, Rt_est, IMG_W, IMG_H,
                                         near=0.3, far=3.0, color=(0, 200, 200))
        except Exception:
            pass  # Skip estimated frustum if decomposition fails

    meshes += make_axis_mesh(origin=(0, 0, 0), length=1.5)

    overview = render_scene(meshes, _OV_K, ov_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE)

    # Status text (multi-line)
    solver = "A^T A (eigenvalues)" if use_normal_eqs else "SVD"
    cond_label = "lambda_n/lambda_2" if use_normal_eqs else "sigma_1/sigma_{n-1}"
    fit_label = "sqrt(lambda_1)" if use_normal_eqs else "sigma_n"

    status_lines = [
        f"Solver: {solver}  |  Points: {len(sel)}"
        f"{'  (coplanar)' if use_coplanar else ''}",
        f"cond(A) [{cond_label}]: {cond_A:.2e}",
        f"Fit [{fit_label}]: {fit_resid:.2e}",
        f"Reprojection Error: {residual:.2f} px",
    ]

    if use_hartley:
        status_lines.append("Hartley normalization: ON")
    if origin_offset > 0:
        status_lines.append(f"Origin offset: {origin_offset:.0e}")
    if world_scale > 1:
        status_lines.append(f"World scale: {world_scale:.0e}")
    if probe_residual is not None:
        status_lines.append(f"Probe residual: {probe_residual:.2f} px")

    if show_A and A is not None:
        status_lines.append(f"\nA matrix ({A.shape[0]}x{A.shape[1]}):")
        for row in A[:min(12, A.shape[0])]:
            status_lines.append("  " + "  ".join(f"{v:8.2f}" for v in row))
        if A.shape[0] > 12:
            status_lines.append(f"  ... ({A.shape[0] - 12} more rows)")

    status = "\n".join(status_lines)

    return {
        "projection": proj_canvas,
        "overview": overview,
        "status": status,
    }
