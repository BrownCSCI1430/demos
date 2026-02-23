"""
DLT Explorer: Homography & Fundamental Matrix estimation via Direct Linear Transform.

A unified demo with two modes:
  1. Homography — estimate H from planar point correspondences, decompose H(λ) = λB + ae₃ᵀ
  2. Fundamental Matrix — 8-point algorithm, rank-2 enforcement, epipoles from SVD null spaces

Demonstrates what SVD produces when applied to each problem and connects to the
plane sweep stereo formulation.

Usage:
    python liveDLT.py
"""

import numpy as np
import cv2

# ── Constants ──────────────────────────────────────────────────────────

IMG_W, IMG_H = 420, 380
OVERVIEW_SIZE = 420

_RNG = np.random.RandomState(42)

# ── Camera setup ───────────────────────────────────────────────────────

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt, fov_to_focal,
    render_scene, make_frustum_mesh, make_axis_mesh,
    make_sphere, make_ground_grid, create_default_scene,
)

K_SHARED = build_intrinsic(
    fx=350.0, fy=350.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)

_Rt1 = make_lookat_Rt(eye=np.array([-2.5, 1.0, 5.0]),
                       target=np.array([0.0, 0.0, 0.0]))
_Rt2 = make_lookat_Rt(eye=np.array([2.5, 1.0, 5.0]),
                       target=np.array([0.0, 0.0, 0.0]))
M1 = K_SHARED @ _Rt1
M2 = K_SHARED @ _Rt2

# Overview camera
_OV_TARGET = np.array([0.0, 0.5, 0.0])
_OV_EYE0 = np.array([0.0, 8.0, 12.0])
_d0 = _OV_EYE0 - _OV_TARGET
_OV_R0 = float(np.linalg.norm(_d0))
_OV_EL0 = float(np.arcsin(np.clip(_d0[1] / _OV_R0, -1.0, 1.0)))
_OV_AZ0 = float(np.arctan2(_d0[0], _d0[2]))

_OV_K = build_intrinsic(
    fov_to_focal(50, OVERVIEW_SIZE), fov_to_focal(50, OVERVIEW_SIZE),
    0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2,
)


# ── 3D points ──────────────────────────────────────────────────────────

def _project(M, pts3d):
    """Project 3D points (N,3) through 3x4 M -> (N,2) pixel coords."""
    h = np.column_stack([pts3d, np.ones(len(pts3d))])
    proj = (M @ h.T).T
    return proj[:, :2] / proj[:, 2:3]


# Planar points for homography mode (coplanar on the ground plane y=0)
PLANE_PTS3D = np.array([
    [x, 0.0, z]
    for z in np.linspace(-2.0, 2.0, 6)
    for x in np.linspace(-2.0, 2.0, 6)
], dtype=np.float64)  # 36 points on the ground


def _sample_scene_surface_points():
    """Sample 3D points on surfaces of the default scene objects.

    Points lie on the actual geometry (cubes, sphere, cylinder, ground)
    so that their 2D projections correspond to real scene locations.
    """
    pts = []

    # Ground plane (y=0)
    for x in np.linspace(-2.5, 2.5, 5):
        for z in np.linspace(-2.0, 2.0, 5):
            pts.append([x, 0.0, z])

    # Blue cube: center=(-1.5, 0.5, 0), size=1.0, half=0.5
    cx, cy, cz, h = -1.5, 0.5, 0.0, 0.5
    for u in np.linspace(-h * 0.7, h * 0.7, 3):
        for v in np.linspace(-h * 0.7, h * 0.7, 3):
            pts.append([cx + h, cy + u, cz + v])   # +X face
            pts.append([cx - h, cy + u, cz + v])   # -X face
            pts.append([cx + u, cy + h, cz + v])   # top
            pts.append([cx + u, cy + v, cz + h])   # +Z face (front)

    # Red cube: center=(1.5, 0.5, -1), size=0.7, half=0.35
    cx, cy, cz, h = 1.5, 0.5, -1.0, 0.35
    for u in np.linspace(-h * 0.7, h * 0.7, 3):
        for v in np.linspace(-h * 0.7, h * 0.7, 3):
            pts.append([cx + h, cy + u, cz + v])   # +X face
            pts.append([cx - h, cy + u, cz + v])   # -X face
            pts.append([cx + u, cy + h, cz + v])   # top
            pts.append([cx + u, cy + v, cz + h])   # +Z face (front)

    # Green sphere: center=(0, 0.8, -2), radius=0.6
    cx, cy, cz, r = 0.0, 0.8, -2.0, 0.6
    for phi in np.linspace(np.pi / 6, 5 * np.pi / 6, 4):      # polar from +Y
        for theta in np.linspace(-np.pi / 3, np.pi / 3, 5):    # front-facing arc
            pts.append([
                cx + r * np.sin(phi) * np.sin(theta),
                cy + r * np.cos(phi),
                cz + r * np.sin(phi) * np.cos(theta),
            ])

    # Cyan cylinder: base_center=(2, 0, 1), radius=0.3, height=1.5
    bx, bz, cr, ch = 2.0, 1.0, 0.3, 1.5
    for y in np.linspace(0.15, ch - 0.15, 4):
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 5):    # front-facing arc
            pts.append([bx + cr * np.cos(theta), y, bz + cr * np.sin(theta)])

    return np.array(pts, dtype=np.float64)


# Volume points for fundamental matrix mode (on scene object surfaces)
VOLUME_PTS3D = _sample_scene_surface_points()

# Filter to visible points (project into both images)
def _filter_visible(pts3d, margin=15):
    p1 = _project(M1, pts3d)
    p2 = _project(M2, pts3d)
    mask = (
        (p1[:, 0] >= margin) & (p1[:, 0] < IMG_W - margin) &
        (p1[:, 1] >= margin) & (p1[:, 1] < IMG_H - margin) &
        (p2[:, 0] >= margin) & (p2[:, 0] < IMG_W - margin) &
        (p2[:, 1] >= margin) & (p2[:, 1] < IMG_H - margin)
    )
    return pts3d[mask]


PLANE_VIS = _filter_visible(PLANE_PTS3D, margin=20)
VOLUME_VIS = _filter_visible(VOLUME_PTS3D, margin=20)

# Pre-computed projections
PLANE_PTS1 = _project(M1, PLANE_VIS)
PLANE_PTS2 = _project(M2, PLANE_VIS)
VOLUME_PTS1 = _project(M1, VOLUME_VIS)
VOLUME_PTS2 = _project(M2, VOLUME_VIS)

# Fixed noise patterns (scaled by noise_px slider)
PLANE_NOISE1 = _RNG.standard_normal(PLANE_PTS1.shape)
PLANE_NOISE2 = _RNG.standard_normal(PLANE_PTS2.shape)
VOLUME_NOISE1 = _RNG.standard_normal(VOLUME_PTS1.shape)
VOLUME_NOISE2 = _RNG.standard_normal(VOLUME_PTS2.shape)


# ── Pre-rendered images ────────────────────────────────────────────────

_SCENE = create_default_scene()


# Scene views in standard pinhole coordinates (no flip) so that 2D projections,
# DLT estimation, and warps all operate in the same coordinate system.
# The flip to upright display happens at the final dpg.set_value step.
SCENE_IMG1 = render_scene(_SCENE, K_SHARED, _Rt1, IMG_W, IMG_H, flip_y=False)
SCENE_IMG2 = render_scene(_SCENE, K_SHARED, _Rt2, IMG_W, IMG_H, flip_y=False)


# ══════════════════════════════════════════════════════════════════════
#  MATH FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

# ── Shared ─────────────────────────────────────────────────────────────

def normalize_2d(pts):
    """Hartley normalization for 2D points.

    Translates centroid to origin, scales so std = 1.
    Returns (pts_norm (N,2), T (3,3)).
    """
    c = pts.mean(axis=0)
    s = 1.0 / (np.std(pts - c) + 1e-10)
    T = np.array([[s, 0, -s * c[0]],
                   [0, s, -s * c[1]],
                   [0, 0, 1.0]])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ pts_h.T).T[:, :2], T


def solve_dlt_svd(A):
    """SVD of A for DLT: Ah = 0.

    Returns (U, S, Vt, h, cond_A, fit_resid).
    h = last row of Vt (null vector).
    cond_A = s[0] / s[-2] (excluding null space).
    fit_resid = s[-1] (quality of fit, ~0 for perfect).
    """
    U, s, Vt = np.linalg.svd(A)
    h = Vt[-1]
    cond_A = float(s[0] / (s[-2] + 1e-12))
    fit_resid = float(s[-1])
    return U, s, Vt, h, cond_A, fit_resid


# ── Homography ─────────────────────────────────────────────────────────

def build_A_homography(pts1, pts2):
    """Build 2N×9 DLT matrix for homography estimation.

    Each correspondence (x1,y1) <-> (x2,y2) contributes two rows:
      row 2i  : [x1 y1 1   0  0  0  -x2·x1 -x2·y1 -x2]
      row 2i+1: [ 0  0  0  x1 y1 1  -y2·x1 -y2·y1 -y2]
    """
    N = len(pts1)
    A = np.zeros((2 * N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
    return A


def estimate_H_dlt(pts1, pts2, use_hartley=False):
    """Estimate 3×3 homography H via DLT such that x2 ~ H x1.

    Returns (H, A, U, S, Vt, cond_A, fit_resid).
    """
    p1, p2 = pts1.copy(), pts2.copy()
    T1, T2 = np.eye(3), np.eye(3)
    if use_hartley:
        p1, T1 = normalize_2d(p1)
        p2, T2 = normalize_2d(p2)

    A = build_A_homography(p1, p2)
    U, S, Vt, h, cond_A, fit_resid = solve_dlt_svd(A)
    H = h.reshape(3, 3)

    if use_hartley:
        H = np.linalg.inv(T2) @ H @ T1

    denom = H[2, 2]
    if abs(denom) > 1e-10:
        H = H / denom
    else:
        H = H / (np.linalg.norm(H) + 1e-10)

    return H, A, U, S, Vt, cond_A, fit_resid


# ── Homography decomposition: H(λ) = λB + a e₃ᵀ ──────────────────────

def compute_B_and_a(K, Rt1, Rt2):
    """Compute plane-sweep decomposition components.

    B = A₂ @ inv(A₁)     — homography at infinity
    a = A₂ @ C₁ + t₂     — epipole-related baseline vector

    H(λ) = λ·B + outer(a, e₃)
    """
    A1 = K @ Rt1[:, :3]
    t1 = K @ Rt1[:, 3]
    A2 = K @ Rt2[:, :3]
    t2 = K @ Rt2[:, 3]

    C1 = np.linalg.solve(A1, -t1)       # camera 1 center in world
    B = A2 @ np.linalg.inv(A1)          # (3×3)
    a = A2 @ C1 + t2                    # (3,)
    return B, a


def compose_H_lambda(B, a, lam, epipole_coeff=1.0):
    """Build H(λ) = λ·B + coeff · outer(a, e₃)."""
    e3 = np.array([0.0, 0.0, 1.0])
    return lam * B + epipole_coeff * np.outer(a, e3)


# Pre-compute B and a for our camera rig
_B, _a = compute_B_and_a(K_SHARED, _Rt1, _Rt2)

# Default depth for H(λ) = λB + ae₃ᵀ plane sweep decomposition.
# λ parameterizes depth planes perpendicular to camera 1's optical axis.
# The scene center is roughly at depth 5–6 from camera 1.
_e3 = np.array([0.0, 0.0, 1.0])
_p1_h = np.column_stack([PLANE_PTS1, np.ones(len(PLANE_PTS1))])
_best_lam, _best_err = 5.0, float("inf")
for _lam in np.linspace(0.1, 15.0, 1000):
    _H_try = _lam * _B + np.outer(_a, _e3)
    _pr = (_H_try @ _p1_h.T).T
    _pr = _pr[:, :2] / _pr[:, 2:3]
    _err = float(np.mean(np.sum((_pr - PLANE_PTS2) ** 2, axis=1)))
    if _err < _best_err:
        _best_err = _err
        _best_lam = _lam
LAM_DEFAULT = round(_best_lam, 1)


# ── Fundamental matrix ─────────────────────────────────────────────────

def build_A_fundamental(pts1, pts2):
    """Build N×9 DLT matrix for fundamental matrix estimation (8-point algorithm).

    Each correspondence (x1,y1) <-> (x2,y2) contributes one row:
      [x2·x1  x2·y1  x2  y2·x1  y2·y1  y2  x1  y1  1]
    """
    N = len(pts1)
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
    return A


def estimate_F_dlt(pts1, pts2, use_hartley=False):
    """Estimate 3×3 fundamental matrix F via the 8-point algorithm.

    Pipeline:
      1. Optionally Hartley-normalize both point sets
      2. Build N×9 A matrix
      3. SVD(A) → null vector f → reshape to 3×3 F_hat
      4. SVD(F_hat) → enforce rank-2 by zeroing σ₃
      5. Denormalize if needed
      6. Extract epipoles from null spaces

    Returns dict with:
      F, F_hat, A, U_A, S_A, Vt_A, U_F, S_F, Vt_F,
      S_F_enforced, e1, e2, cond_A, fit_resid
    """
    p1, p2 = pts1.copy(), pts2.copy()
    T1, T2 = np.eye(3), np.eye(3)
    if use_hartley:
        p1, T1 = normalize_2d(p1)
        p2, T2 = normalize_2d(p2)

    A = build_A_fundamental(p1, p2)
    U_A, S_A, Vt_A, f_vec, cond_A, fit_resid = solve_dlt_svd(A)

    # Reshape to 3×3 (pre-rank-enforcement)
    F_hat = f_vec.reshape(3, 3)

    # SVD of F_hat for rank enforcement
    U_F, S_F_raw, Vt_F = np.linalg.svd(F_hat)
    S_F = S_F_raw.copy()

    # Enforce rank 2: set σ₃ = 0
    S_F_enforced = np.array([S_F[0], S_F[1], 0.0])
    F = U_F @ np.diag(S_F_enforced) @ Vt_F

    if use_hartley:
        F = T2.T @ F @ T1

    # Normalize F so ||F|| = 1
    fn = np.linalg.norm(F)
    if fn > 1e-10:
        F = F / fn

    # Extract epipoles from final F
    Uf, Sf, Vtf = np.linalg.svd(F)
    e1 = Vtf[-1]  # right null: F @ e1 = 0 (cam2 center in image 1)
    e2 = Uf[:, -1]  # left null: Fᵀ @ e2 = 0 (cam1 center in image 2)

    # Dehomogenize
    if abs(e1[2]) > 1e-10:
        e1 = e1 / e1[2]
    if abs(e2[2]) > 1e-10:
        e2 = e2 / e2[2]

    return {
        "F": F, "F_hat": F_hat, "A": A,
        "U_A": U_A, "S_A": S_A, "Vt_A": Vt_A,
        "U_F": U_F, "S_F": S_F, "Vt_F": Vt_F,
        "S_F_enforced": S_F_enforced,
        "e1": e1, "e2": e2,
        "cond_A": cond_A, "fit_resid": fit_resid,
    }


def compute_F_from_cameras(M1, M2):
    """Ground truth F = [e₂]× · M₂ · pinv(M₁) from known camera matrices."""
    A1 = M1[:, :3]
    C1 = np.linalg.solve(A1, -M1[:, 3])
    C1h = np.append(C1, 1.0)

    e2 = M2 @ C1h
    e2x = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])
    F = e2x @ M2 @ np.linalg.pinv(M1)
    fn = np.linalg.norm(F)
    if fn > 1e-10:
        F = F / fn
    return F


# ── Epipolar line helpers ──────────────────────────────────────────────

def compute_epipolar_line(F, pt):
    """l = F @ [x, y, 1]ᵀ — epipolar line in the other image."""
    return F @ np.array([pt[0], pt[1], 1.0])


def epipolar_line_endpoints(l, img_w, img_h):
    """Clip line l = [a, b, c] (ax + by + c = 0) to image bounds.

    Returns (p1, p2) or (None, None) if line doesn't cross image.
    """
    a, b, c = l
    pts = []
    if abs(b) > 1e-6:
        y0 = -c / b
        y1 = -(c + a * (img_w - 1)) / b
        pts.append((0, int(y0)))
        pts.append((img_w - 1, int(y1)))
    if abs(a) > 1e-6:
        x0 = -c / a
        x1 = -(c + b * (img_h - 1)) / a
        pts.append((int(x0), 0))
        pts.append((int(x1), img_h - 1))

    if len(pts) >= 2:
        best_p1, best_p2, best_d = pts[0], pts[1], 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if d > best_d:
                    best_p1, best_p2, best_d = pts[i], pts[j], d
        return best_p1, best_p2
    return None, None


# ── Warp helper ────────────────────────────────────────────────────────

def warp_image(src, H, img_w, img_h):
    """Warp src by forward homography H (cam1→cam2).

    OpenCV internally applies H⁻¹ to look up source pixels,
    so warped[p] = src[H⁻¹·p] — correct for cam1→cam2 direction.
    """
    return cv2.warpPerspective(
        src, H, (img_w, img_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


# ══════════════════════════════════════════════════════════════════════
#  SVD VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def _val_to_color(v, vmin, vmax):
    """Map value to diverging blue-white-red colormap (BGR)."""
    if vmax - vmin < 1e-10:
        return (200, 200, 200)
    t = (v - vmin) / (vmax - vmin)  # 0..1
    # Blue(0) -> White(0.5) -> Red(1)
    if t < 0.5:
        r = int(2 * t * 255)
        g = int(2 * t * 255)
        b = 255
    else:
        r = 255
        g = int(2 * (1 - t) * 255)
        b = int(2 * (1 - t) * 255)
    return (b, g, r)  # BGR


def render_svd_panel(U, S, Vt, canvas_w, canvas_h,
                     title="SVD of A",
                     highlight_null_col_V=True,
                     highlight_null_col_U=False,
                     sigma_labels=None):
    """Render SVD decomposition as a color-coded BGR image.

    Shows: [U] [×] [Σ bars] [×] [Vᵀ]
    With null vectors highlighted.
    """
    img = np.full((canvas_h, canvas_w, 3), 25, dtype=np.uint8)

    m, k = U.shape[0], len(S)
    _, n = Vt.shape

    # Layout: title row, then matrices
    title_h = 25
    cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 220), 1, cv2.LINE_AA)

    # Available space
    avail_w = canvas_w - 20
    avail_h = canvas_h - title_h - 10
    y_off = title_h + 5

    # For large matrices, limit rows displayed
    max_rows_U = min(m, 12)
    max_cols = min(k, 9)
    max_rows_Vt = min(Vt.shape[0], max_cols)

    # Proportions: U takes 30%, bars 15%, Vt takes 30%, labels take rest
    u_w = int(avail_w * 0.32)
    bar_w = int(avail_w * 0.18)
    vt_w = int(avail_w * 0.32)
    gap = 8

    # --- Draw U matrix (left portion) ---
    u_x0 = 10
    cell_w_u = max(1, (u_w - 2) // max_cols)
    cell_h_u = max(1, (avail_h - 2) // max_rows_U)
    cell_sz_u = min(cell_w_u, cell_h_u, 22)

    U_flat = U[:max_rows_U, :max_cols].ravel()
    vmin_u, vmax_u = U_flat.min(), U_flat.max()

    # Label
    cv2.putText(img, "U", (u_x0, y_off - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 200), 1, cv2.LINE_AA)

    for r in range(max_rows_U):
        for c in range(max_cols):
            x1 = u_x0 + c * cell_sz_u
            y1 = y_off + r * cell_sz_u
            x2 = x1 + cell_sz_u - 1
            y2 = y1 + cell_sz_u - 1
            color = _val_to_color(U[r, c], vmin_u, vmax_u)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    # Highlight last column of U (left null vector for F)
    if highlight_null_col_U and max_cols > 0:
        c = max_cols - 1
        x1 = u_x0 + c * cell_sz_u - 1
        y1 = y_off - 1
        x2 = x1 + cell_sz_u + 1
        y2 = y_off + max_rows_U * cell_sz_u
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)  # orange

    if max_rows_U < m:
        cv2.putText(img, f"...{m - max_rows_U} more",
                     (u_x0, y_off + max_rows_U * cell_sz_u + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    # --- Draw Σ as bar chart ---
    bar_x0 = u_x0 + max_cols * cell_sz_u + gap + 10

    cv2.putText(img, "Σ", (bar_x0, y_off - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 200), 1, cv2.LINE_AA)

    s_max = S[0] if S[0] > 0 else 1.0
    bar_height = min(18, (avail_h - 4) // max_cols)

    for i in range(max_cols):
        bw = int((S[i] / s_max) * (bar_w - 50))
        bw = max(1, bw)
        y1 = y_off + i * bar_height
        y2 = y1 + bar_height - 2

        # Color: last bar is null space (red if forced to 0)
        if sigma_labels and i < len(sigma_labels) and "→ 0" in sigma_labels[i]:
            bar_color = (60, 60, 220)  # red for enforced zero
        elif i == max_cols - 1:
            bar_color = (60, 180, 60)  # green for null vector
        else:
            bar_color = (180, 150, 80)  # teal

        cv2.rectangle(img, (bar_x0, y1), (bar_x0 + bw, y2), bar_color, -1)

        # Label
        lbl = f"σ{i+1}={S[i]:.3g}"
        if sigma_labels and i < len(sigma_labels):
            lbl = sigma_labels[i]
        cv2.putText(img, lbl, (bar_x0 + bw + 4, y2 - 1),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

    # --- Draw Vᵀ matrix ---
    vt_x0 = bar_x0 + bar_w + gap
    cell_sz_v = min(cell_sz_u, 22, (avail_h - 2) // max_rows_Vt)

    Vt_sub = Vt[:max_rows_Vt, :max_cols]
    vmin_v, vmax_v = Vt_sub.min(), Vt_sub.max()

    cv2.putText(img, "V^T", (vt_x0, y_off - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 200), 1, cv2.LINE_AA)

    for r in range(max_rows_Vt):
        for c in range(max_cols):
            x1 = vt_x0 + c * cell_sz_v
            y1 = y_off + r * cell_sz_v
            x2 = x1 + cell_sz_v - 1
            y2 = y1 + cell_sz_v - 1
            color = _val_to_color(Vt_sub[r, c], vmin_v, vmax_v)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)

    # Highlight last row of Vᵀ (= last column of V = right null vector)
    if highlight_null_col_V and max_rows_Vt > 0:
        r = max_rows_Vt - 1
        x1 = vt_x0 - 1
        y1 = y_off + r * cell_sz_v - 1
        x2 = vt_x0 + max_cols * cell_sz_v
        y2 = y1 + cell_sz_v + 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 60), 2)  # green

    return img


# ══════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════

def _depth_color(z, zmin, zmax):
    """Map depth to color (orange=far, green=mid, blue=near). BGR."""
    if zmax - zmin < 1e-6:
        return (0, 200, 0)
    t = (z - zmin) / (zmax - zmin)
    r = int(255 * (1 - t))
    g = int(200 * (1 - abs(2 * t - 1)))
    b = int(255 * t)
    return (b, g, r)


def draw_points(img, pts, pts3d=None, radius=5, thickness=-1):
    """Draw 2D points on image, optionally colored by depth."""
    if pts3d is not None:
        zvals = pts3d[:, 2]
        zmin, zmax = zvals.min(), zvals.max()
    for i, (x, y) in enumerate(pts):
        ix, iy = int(round(x)), int(round(y))
        if pts3d is not None:
            color = _depth_color(pts3d[i, 2], zmin, zmax)
        else:
            color = (0, 200, 0)
        cv2.circle(img, (ix, iy), radius, color, thickness)


def draw_epipolar_lines_on_img(img, F, pts_other, color=(0, 220, 220),
                                highlight_idx=-1):
    """Draw epipolar lines l = F @ x for each point in pts_other."""
    for i, pt in enumerate(pts_other):
        l = compute_epipolar_line(F, pt)
        p1, p2 = epipolar_line_endpoints(l, img.shape[1], img.shape[0])
        if p1 is not None:
            th = 2 if i == highlight_idx else 1
            c = (0, 255, 255) if i == highlight_idx else color
            cv2.line(img, p1, p2, c, th, cv2.LINE_AA)


def draw_epipole(img, epipole, color=(0, 0, 255), size=12):
    """Draw epipole as a diamond marker if within image bounds (with margin)."""
    ex, ey = int(round(epipole[0])), int(round(epipole[1]))
    h, w = img.shape[:2]
    margin = 500  # draw even if slightly outside
    if -margin < ex < w + margin and -margin < ey < h + margin:
        pts = np.array([
            [ex, ey - size], [ex + size, ey],
            [ex, ey + size], [ex - size, ey],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (ex, ey), color, cv2.MARKER_DIAMOND, size, 2)


def draw_residuals(img, pts_true, pts_est, color=(128, 128, 128)):
    """Draw residual lines from true to estimated projections."""
    for (x1, y1), (x2, y2) in zip(pts_true, pts_est):
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)


# ══════════════════════════════════════════════════════════════════════
#  ORBIT CAMERA
# ══════════════════════════════════════════════════════════════════════

class OvCam:
    """Spherical orbit camera for the 3D overview."""
    az = _OV_AZ0
    el = _OV_EL0
    radius = _OV_R0
    target = _OV_TARGET.copy()
    _prev = None

    @classmethod
    def reset(cls):
        cls.az = _OV_AZ0
        cls.el = _OV_EL0
        cls.radius = _OV_R0
        cls._prev = None

    @classmethod
    def make_Rt(cls):
        eye = cls.target + cls.radius * np.array([
            np.cos(cls.el) * np.sin(cls.az),
            np.sin(cls.el),
            np.cos(cls.el) * np.cos(cls.az),
        ])
        return make_lookat_Rt(eye, cls.target)


# ══════════════════════════════════════════════════════════════════════
#  GUIDE TEXT
# ══════════════════════════════════════════════════════════════════════

GUIDE_HOMOGRAPHY = [
    {
        "title": "1. DLT solves Ah = 0",
        "body": (
            "Given N point correspondences on a plane, DLT constructs a 2N×9 "
            "matrix A where each correspondence contributes two rows. The "
            "homography h (vectorized 3×3) is the null vector of A — the "
            "direction where Ah ≈ 0."
        ),
    },
    {
        "title": "2. SVD produces the null vector",
        "body": (
            "SVD(A) = UΣVᵀ. The smallest singular value σ₉ measures how well "
            "the data fits Ah = 0. The last row of Vᵀ (= last column of V) is "
            "the null vector h. Reshape h into the 3×3 homography H."
        ),
    },
    {
        "title": "3. What does H do?",
        "body": (
            "H transforms points from image 1 to image 2 for a known plane. "
            "Warping image 1 by H should align it with image 2 on the plane. "
            "Try increasing noise to see the alignment degrade."
        ),
    },
    {
        "title": "4. Decomposition: H(λ) = λB + ae₃ᵀ",
        "body": (
            "Toggle 'Decompose H' to see the structural breakdown:\n"
            "• B = A₂A₁⁻¹ is the homography at infinity — it depends only on "
            "camera rotation and intrinsics (the 'known basis').\n"
            "• ae₃ᵀ encodes the epipolar baseline — the depth-dependent parallax.\n"
            "• λ is the depth parameter. At λ_true, the warp aligns perfectly.\n\n"
            "Set epipole coefficient to 0 to see B alone (no depth correction). "
            "Sweep λ to see how depth affects alignment — this is exactly what "
            "plane sweep stereo exploits."
        ),
    },
    {
        "title": "5. Conditioning and normalization",
        "body": (
            "The condition number σ₁/σ₈ indicates how well-defined the null "
            "vector is. Large condition numbers mean the solution is sensitive "
            "to noise. Toggle Hartley normalization to see conditioning improve."
        ),
    },
]

GUIDE_FUNDAMENTAL = [
    {
        "title": "1. The 8-point algorithm",
        "body": (
            "Given N ≥ 8 point correspondences in two views of a general 3D "
            "scene, DLT constructs an N×9 matrix A (one row per correspondence). "
            "SVD(A) → null vector f → reshape to 3×3 F̂."
        ),
    },
    {
        "title": "2. F̂ is not quite right",
        "body": (
            "The SVD of F̂ has three singular values σ₁, σ₂, σ₃. For a proper "
            "fundamental matrix, F must be rank 2 (σ₃ = 0). With noise, σ₃ is "
            "small but nonzero."
        ),
    },
    {
        "title": "3. Rank enforcement",
        "body": (
            "Set σ₃ = 0 to enforce rank 2: F = U·diag(σ₁,σ₂,0)·Vᵀ.\n\n"
            "Why rank 2? F maps points to lines, not points to points. "
            "Rank 2 ensures that Fe₁ = 0 — there exists a point (the epipole) "
            "that maps to the zero line, meaning every epipolar line passes "
            "through the epipole."
        ),
    },
    {
        "title": "4. Epipoles from null spaces",
        "body": (
            "From SVD(F) = UΣVᵀ with σ₃ = 0:\n"
            "• e₁ = last column of V (right null: Fe₁ = 0)\n"
            "  → camera 2's center projected into image 1\n"
            "• e₂ = last column of U (left null: Fᵀe₂ = 0)\n"
            "  → camera 1's center projected into image 2\n\n"
            "All epipolar lines in image 2 converge at e₂."
        ),
    },
    {
        "title": "5. Known basis + epipole component",
        "body": (
            "F = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ is the rank-2 'known basis' relating "
            "the two image planes. The zeroed σ₃u₃v₃ᵀ is the epipole component "
            "— setting it to zero constrains all epipolar lines to pass through "
            "the epipole.\n\n"
            "Equivalently: F = [e₂]× · H_π, where H_π is a homography induced "
            "by any plane. This is the same structural idea as the homography "
            "decomposition: a known inter-image mapping + an epipole constraint."
        ),
    },
    {
        "title": "6. F maps points to lines",
        "body": (
            "For point x₁ in image 1, the epipolar line in image 2 is "
            "l₂ = F·x₁. The matching point in image 2 must lie on this line. "
            "Use the 'Highlight point' slider to see individual epipolar lines "
            "— they all pass through e₂."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════
#  DEAR PYGUI DEMO (main)
# ══════════════════════════════════════════════════════════════════════

# This section is only used when running liveDLT.py directly.
# The web frame module (web/frames/dlt.py) imports the math functions above.

if __name__ == "__main__":
    import dearpygui.dearpygui as dpg
    from utils.demo_utils import convert_cv_to_dpg
    from utils.demo_ui import (
        setup_viewport, make_state_updater, make_reset_callback,
        create_parameter_table, add_parameter_row, add_parameter_spacer_row,
        load_fonts, bind_mono_font, add_global_controls,
    )

    # ── Defaults ───────────────────────────────────────────────────────
    DEFAULTS = {
        "ui_scale": 1.5,
        "mode": "Homography",
        "noise_px": 0.0,
        "n_points": 12,
        "use_hartley": False,
        "decompose_H": False,
        "lambda_val": LAM_DEFAULT,
        "epipole_coeff": 1.0,
        "show_warp": True,
        "show_epipolar_lines": True,
        "show_epipoles": True,
        "highlight_idx": 0,
        "show_svd_A": True,
        "show_svd_F": False,
        "show_A_matrix": False,
    }

    class State:
        mode = DEFAULTS["mode"]
        noise_px = DEFAULTS["noise_px"]
        n_points = DEFAULTS["n_points"]
        use_hartley = DEFAULTS["use_hartley"]
        decompose_H = DEFAULTS["decompose_H"]
        lambda_val = DEFAULTS["lambda_val"]
        epipole_coeff = DEFAULTS["epipole_coeff"]
        show_warp = DEFAULTS["show_warp"]
        show_epipolar_lines = DEFAULTS["show_epipolar_lines"]
        show_epipoles = DEFAULTS["show_epipoles"]
        highlight_idx = DEFAULTS["highlight_idx"]
        show_svd_A = DEFAULTS["show_svd_A"]
        show_svd_F = DEFAULTS["show_svd_F"]
        show_A_matrix = DEFAULTS["show_A_matrix"]

    state = State()

    def _on_mode_change(sender, app_data):
        state.mode = app_data

    # ── Render loop ────────────────────────────────────────────────────
    def render_frame():
        """Compute and render one frame of the DLT demo."""
        is_homography = state.mode == "Homography"

        # Select points and add noise
        if is_homography:
            n_avail = len(PLANE_VIS)
            n = min(max(4, state.n_points), n_avail)
            pts3d = PLANE_VIS[:n]
            pts1_true = PLANE_PTS1[:n]
            pts2_true = PLANE_PTS2[:n]
            pts1 = pts1_true + PLANE_NOISE1[:n] * state.noise_px
            pts2 = pts2_true + PLANE_NOISE2[:n] * state.noise_px
            base1, base2 = SCENE_IMG1.copy(), SCENE_IMG2.copy()
        else:
            n_avail = len(VOLUME_VIS)
            n = min(max(8, state.n_points), n_avail)
            pts3d = VOLUME_VIS[:n]
            pts1_true = VOLUME_PTS1[:n]
            pts2_true = VOLUME_PTS2[:n]
            pts1 = pts1_true + VOLUME_NOISE1[:n] * state.noise_px
            pts2 = pts2_true + VOLUME_NOISE2[:n] * state.noise_px
            base1, base2 = SCENE_IMG1.copy(), SCENE_IMG2.copy()

        # ── Camera 1 view ──────────────────────────────────────────────
        view1 = base1.copy()
        draw_points(view1, pts1, pts3d, radius=5)

        # ── Run DLT ────────────────────────────────────────────────────
        status_lines = []
        svd_img = np.full((250, 500, 3), 25, dtype=np.uint8)

        if is_homography:
            try:
                H, A, U, S, Vt, cond_A, fit_resid = estimate_H_dlt(
                    pts1, pts2, use_hartley=state.use_hartley)

                # Warp cam1 to cam2
                view2 = base2.copy()
                if state.decompose_H:
                    H_decomp = compose_H_lambda(
                        _B, _a, state.lambda_val, state.epipole_coeff)
                    warped = warp_image(base1, H_decomp, IMG_W, IMG_H)
                else:
                    warped = warp_image(base1, H, IMG_W, IMG_H)

                if state.show_warp:
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
                    f"Mode: Homography | N={n} pts | "
                    f"Noise={state.noise_px:.1f} px | "
                    f"Hartley: {'ON' if state.use_hartley else 'OFF'}")
                status_lines.append(
                    f"cond(A): σ₁/σ₈ = {cond_A:.3g} | "
                    f"fit residual: σ₉ = {fit_resid:.4g}")
                status_lines.append(f"Reprojection error: {reproj:.2f} px")
                if state.decompose_H:
                    status_lines.append(
                        f"H(λ) = {state.lambda_val:.2f}·B + "
                        f"{state.epipole_coeff:.2f}·ae₃ᵀ  "
                        f"(best-fit λ ≈ {LAM_DEFAULT:.1f})")

                # SVD panel
                if state.show_svd_A:
                    svd_img = render_svd_panel(
                        U, S, Vt, 500, 250,
                        title="SVD of A (2N×9 DLT system)",
                        highlight_null_col_V=True)

            except Exception as e:
                view2 = base2.copy()
                status_lines.append(f"Error: {e}")

        else:  # Fundamental Matrix mode
            try:
                result = estimate_F_dlt(
                    pts1, pts2, use_hartley=state.use_hartley)
                F = result["F"]
                e1, e2 = result["e1"], result["e2"]

                view2 = base2.copy()

                # Draw epipolar lines in view2 (l2 = F @ x1)
                if state.show_epipolar_lines:
                    draw_epipolar_lines_on_img(
                        view2, F, pts1,
                        highlight_idx=state.highlight_idx)

                # Draw epipolar lines in view1 (l1 = Fᵀ @ x2)
                if state.show_epipolar_lines:
                    draw_epipolar_lines_on_img(
                        view1, F.T, pts2, color=(220, 180, 0),
                        highlight_idx=state.highlight_idx)

                draw_points(view2, pts2, pts3d, radius=5)

                # Draw epipoles
                if state.show_epipoles:
                    draw_epipole(view1, e1, color=(60, 200, 60), size=14)
                    draw_epipole(view2, e2, color=(0, 140, 255), size=14)

                status_lines.append(
                    f"Mode: Fundamental Matrix | N={n} pts | "
                    f"Noise={state.noise_px:.1f} px | "
                    f"Hartley: {'ON' if state.use_hartley else 'OFF'}")
                status_lines.append(
                    f"cond(A): σ₁/σ₈ = {result['cond_A']:.3g} | "
                    f"fit residual: σ₉ = {result['fit_resid']:.4g}")
                status_lines.append(
                    f"σ₃(F̂) = {result['S_F'][2]:.4g} "
                    f"(forced to 0 for rank-2)")
                status_lines.append(
                    f"Epipoles: e₁=({e1[0]:.0f}, {e1[1]:.0f}) "
                    f"e₂=({e2[0]:.0f}, {e2[1]:.0f})")

                # SVD panel
                if state.show_svd_F:
                    sigma_labels = [
                        f"σ₁={result['S_F'][0]:.3g}",
                        f"σ₂={result['S_F'][1]:.3g}",
                        f"σ₃={result['S_F'][2]:.3g} → 0",
                    ]
                    svd_img = render_svd_panel(
                        result["U_F"], result["S_F"], result["Vt_F"],
                        500, 250,
                        title="SVD of F̂ (rank enforcement)",
                        highlight_null_col_V=True,
                        highlight_null_col_U=True,
                        sigma_labels=sigma_labels)
                elif state.show_svd_A:
                    svd_img = render_svd_panel(
                        result["U_A"], result["S_A"], result["Vt_A"],
                        500, 250,
                        title="SVD of A (N×9 DLT system)",
                        highlight_null_col_V=True)

            except Exception as e:
                view2 = base2.copy()
                status_lines.append(f"Error: {e}")

        # ── 3D Overview ────────────────────────────────────────────────
        frust1 = make_frustum_mesh(K_SHARED, _Rt1, IMG_W, IMG_H,
                                    near=0.3, far=6.0, color=(50, 200, 50))
        frust2 = make_frustum_mesh(K_SHARED, _Rt2, IMG_W, IMG_H,
                                    near=0.3, far=6.0, color=(200, 150, 50))
        axes = make_axis_mesh(origin=(0, 0, 0), length=1.5)
        ground = make_ground_grid(y=0.0, extent=3.0, spacing=1.0, color=(100, 100, 100))

        ov_Rt = OvCam.make_Rt()
        overview = render_scene(
            _SCENE + frust1 + frust2 + axes + [ground],
            _OV_K, ov_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE,
        )

        # ── Update textures ────────────────────────────────────────────
        # Camera views are in standard pinhole coords; flip for upright display.
        dpg.set_value("tex_view1", convert_cv_to_dpg(cv2.flip(view1, 0)))
        dpg.set_value("tex_view2", convert_cv_to_dpg(cv2.flip(view2, 0)))
        dpg.set_value("tex_overview", convert_cv_to_dpg(overview))
        dpg.set_value("tex_svd", convert_cv_to_dpg(svd_img))

        # ── Update status ──────────────────────────────────────────────
        dpg.set_value("status_text", "\n".join(status_lines))

    # ── Mouse handler for orbit ────────────────────────────────────────
    def _on_mouse(sender, app_data):
        if not dpg.is_item_hovered("img_overview"):
            OvCam._prev = None
            return
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            mx, my = dpg.get_mouse_pos()
            if OvCam._prev is not None:
                dx = mx - OvCam._prev[0]
                dy = my - OvCam._prev[1]
                OvCam.az += dx * 0.01
                OvCam.el = np.clip(OvCam.el - dy * 0.01, -1.4, 1.4)
            OvCam._prev = (mx, my)
        else:
            OvCam._prev = None

    def _on_scroll(sender, app_data):
        if dpg.is_item_hovered("img_overview"):
            OvCam.radius = np.clip(OvCam.radius - app_data * 0.5, 3.0, 30.0)

    # ── Main ───────────────────────────────────────────────────────────
    def main():
        dpg.create_context()
        fonts = load_fonts()

        # Mouse handlers
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=_on_mouse)
            dpg.add_mouse_wheel_handler(callback=_on_scroll)

        # Textures
        with dpg.texture_registry():
            dpg.add_raw_texture(IMG_W, IMG_H, [0.0] * IMG_W * IMG_H * 4,
                                format=dpg.mvFormat_Float_rgba, tag="tex_view1")
            dpg.add_raw_texture(IMG_W, IMG_H, [0.0] * IMG_W * IMG_H * 4,
                                format=dpg.mvFormat_Float_rgba, tag="tex_view2")
            dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE,
                                [0.0] * OVERVIEW_SIZE * OVERVIEW_SIZE * 4,
                                format=dpg.mvFormat_Float_rgba, tag="tex_overview")
            dpg.add_raw_texture(500, 250, [0.0] * 500 * 250 * 4,
                                format=dpg.mvFormat_Float_rgba, tag="tex_svd")

        # Window
        with dpg.window(tag="main_window"):
            def _dlt_extra_reset():
                # Sliders use raw DEFAULTS keys as tags (no _slider suffix)
                for key in ("noise_px", "n_points", "lambda_val",
                            "epipole_coeff", "highlight_idx"):
                    if dpg.does_item_exist(key):
                        dpg.set_value(key, DEFAULTS[key])
                # Checkboxes with non-standard tags
                for tag, key in [("cb_hartley", "use_hartley"),
                                 ("cb_decompose", "decompose_H"),
                                 ("cb_svd_f", "show_svd_F")]:
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, DEFAULTS[key])

            add_global_controls(DEFAULTS, state,
                                reset_extra=_dlt_extra_reset,
                                guide=GUIDE_HOMOGRAPHY
                                + [{"title": "\u2500\u2500\u2500 Fundamental Matrix \u2500\u2500\u2500", "body": ""}]
                                + GUIDE_FUNDAMENTAL,
                                guide_title="DLT Explorer")

            dpg.add_separator()

            # Controls
            with dpg.group(horizontal=True):
                # Mode selector
                with dpg.child_window(width=240, height=280, border=False,
                                       no_scrollbar=True):
                    with dpg.collapsing_header(label="Mode", default_open=True):
                        dpg.add_combo(["Homography", "Fundamental Matrix"],
                                      default_value=state.mode, label="Mode",
                                      callback=_on_mode_change, width=200)

                dpg.add_spacer(width=10)

                # Left column: parameters
                with dpg.child_window(width=450, height=280, border=False,
                                       no_scrollbar=True):
                    with dpg.collapsing_header(label="Input Data",
                                                default_open=True):
                        with create_parameter_table():
                            add_parameter_row(
                                "Noise (px)", "noise_px", DEFAULTS["noise_px"],
                                0.0, 15.0,
                                make_state_updater(state, "noise_px"),
                                make_reset_callback(state, "noise_px",
                                                    "noise_px",
                                                    DEFAULTS["noise_px"]),
                                slider_type="float", width=200, format_str="%.1f")
                            add_parameter_row(
                                "# Points", "n_points", DEFAULTS["n_points"],
                                4, 36,
                                make_state_updater(state, "n_points"),
                                make_reset_callback(state, "n_points",
                                                    "n_points",
                                                    DEFAULTS["n_points"]),
                                slider_type="int", width=200, format_str="%d")

                    with dpg.collapsing_header(label="Solver",
                                                default_open=True):
                        dpg.add_checkbox(label="Hartley Normalize",
                                          default_value=state.use_hartley,
                                          callback=make_state_updater(
                                              state, "use_hartley"),
                                          tag="cb_hartley")

                    with dpg.collapsing_header(label="Homography",
                                                default_open=True,
                                                tag="grp_homography"):
                        dpg.add_checkbox(label="Show Warp",
                                          default_value=state.show_warp,
                                          callback=make_state_updater(
                                              state, "show_warp"))
                        dpg.add_checkbox(
                            label="Decompose H = λB + ae₃ᵀ",
                            default_value=state.decompose_H,
                            callback=make_state_updater(
                                state, "decompose_H"),
                            tag="cb_decompose")
                        with create_parameter_table():
                            add_parameter_row(
                                "λ (depth)", "lambda_val",
                                DEFAULTS["lambda_val"], 0.5, 8.0,
                                make_state_updater(state, "lambda_val"),
                                make_reset_callback(state, "lambda_val",
                                                    "lambda_val",
                                                    DEFAULTS["lambda_val"]),
                                slider_type="float", width=200, format_str="%.2f")
                            add_parameter_row(
                                "Epipole coeff", "epipole_coeff",
                                DEFAULTS["epipole_coeff"], 0.0, 2.0,
                                make_state_updater(state, "epipole_coeff"),
                                make_reset_callback(state, "epipole_coeff",
                                                    "epipole_coeff",
                                                    DEFAULTS["epipole_coeff"]),
                                slider_type="float", width=200, format_str="%.2f")

                    with dpg.collapsing_header(label="Epipolar Geometry",
                                                default_open=True,
                                                tag="grp_fundamental"):
                        dpg.add_checkbox(
                            label="Show Epipolar Lines",
                            default_value=state.show_epipolar_lines,
                            callback=make_state_updater(
                                state, "show_epipolar_lines"))
                        dpg.add_checkbox(
                            label="Show Epipoles",
                            default_value=state.show_epipoles,
                            callback=make_state_updater(
                                state, "show_epipoles"))
                        with create_parameter_table():
                            add_parameter_row(
                                "Highlight pt", "highlight_idx",
                                DEFAULTS["highlight_idx"], 0, 35,
                                make_state_updater(state, "highlight_idx"),
                                make_reset_callback(state, "highlight_idx",
                                                    "highlight_idx",
                                                    DEFAULTS["highlight_idx"]),
                                slider_type="int", width=200, format_str="%d")

                    with dpg.collapsing_header(label="Inspect",
                                                default_open=False):
                        dpg.add_checkbox(
                            label="Show SVD(A)",
                            default_value=state.show_svd_A,
                            callback=make_state_updater(
                                state, "show_svd_A"))
                        dpg.add_checkbox(
                            label="Show SVD(F) rank enforcement",
                            default_value=state.show_svd_F,
                            callback=make_state_updater(
                                state, "show_svd_F"),
                            tag="cb_svd_f")

            dpg.add_separator()

            # Status
            dpg.add_text("", tag="status_text")
            if fonts and fonts[1]:
                bind_mono_font("status_text")

            dpg.add_separator()

            # Images
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Camera 1")
                    dpg.add_image("tex_view1", tag="img_view1")
                with dpg.group():
                    dpg.add_text("Camera 2 / Result")
                    dpg.add_image("tex_view2", tag="img_view2")
                with dpg.group():
                    dpg.add_text("3D Overview (drag to orbit)")
                    dpg.add_image("tex_overview", tag="img_overview")

            dpg.add_text("SVD Decomposition")
            dpg.add_image("tex_svd", tag="img_svd")

        setup_viewport("DLT Explorer", 1400, 950, "main_window",
                       lambda: None, DEFAULTS["ui_scale"])

        while dpg.is_dearpygui_running():
            # Toggle visibility based on mode
            is_h = state.mode == "Homography"
            dpg.configure_item("grp_homography", show=is_h)
            dpg.configure_item("grp_fundamental", show=not is_h)
            dpg.configure_item("cb_svd_f", show=not is_h)

            render_frame()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    main()
