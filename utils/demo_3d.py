"""
3D rendering utilities for CSCI 1430 camera demo.
Software renderer using NumPy + OpenCV with painter's algorithm.
"""

import numpy as np
import cv2


# =============================================================================
# Mesh Generation
# =============================================================================

def make_cube(center, size, color):
    """Create a cube mesh.

    Args:
        center: (x, y, z) center position
        size: edge length
        color: (B, G, R) base color

    Returns:
        Mesh dict with vertices (8,3), faces (6 quads), color
    """
    cx, cy, cz = center
    h = size / 2.0
    vertices = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ])
    # Quads with outward-facing normals (CCW when viewed from outside)
    faces = [
        [0, 3, 2, 1],  # back  (-Z)
        [4, 5, 6, 7],  # front (+Z)
        [0, 1, 5, 4],  # bottom (-Y)
        [2, 3, 7, 6],  # top (+Y)
        [0, 4, 7, 3],  # left (-X)
        [1, 2, 6, 5],  # right (+X)
    ]
    return {"vertices": vertices, "faces": faces, "color": color}


def make_sphere(center, radius, color, n_lat=10, n_lon=16):
    """Create a UV sphere mesh.

    Args:
        center: (x, y, z) center position
        radius: sphere radius
        color: (B, G, R) base color
        n_lat: number of latitude divisions
        n_lon: number of longitude divisions

    Returns:
        Mesh dict
    """
    cx, cy, cz = center
    vertices = []
    faces = []

    # Top pole
    vertices.append([cx, cy + radius, cz])

    # Latitude rings
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = cx + radius * np.sin(phi) * np.cos(theta)
            y = cy + radius * np.cos(phi)
            z = cz + radius * np.sin(phi) * np.sin(theta)
            vertices.append([x, y, z])

    # Bottom pole
    vertices.append([cx, cy - radius, cz])
    vertices = np.array(vertices)

    # Top cap triangles
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])

    # Quad strips between latitude rings
    for i in range(n_lat - 2):
        ring_start = 1 + i * n_lon
        next_ring_start = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            faces.append([
                ring_start + j,
                next_ring_start + j,
                next_ring_start + j_next,
                ring_start + j_next,
            ])

    # Bottom cap triangles
    bottom = len(vertices) - 1
    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom, last_ring + j_next, last_ring + j])

    return {"vertices": vertices, "faces": faces, "color": color}


def make_octahedron(center, radius, color):
    """Create a regular octahedron mesh — looks like a 3D diamond.

    Args:
        center: (x, y, z) center position
        radius: distance from center to each vertex
        color: (B, G, R) base color

    Returns:
        Mesh dict with 6 vertices and 8 triangular faces
    """
    cx, cy, cz = center
    r = radius
    vertices = np.array([
        [cx,   cy+r, cz  ],  # 0: top
        [cx+r, cy,   cz  ],  # 1: right
        [cx,   cy,   cz+r],  # 2: front
        [cx-r, cy,   cz  ],  # 3: left
        [cx,   cy,   cz-r],  # 4: back
        [cx,   cy-r, cz  ],  # 5: bottom
    ], dtype=np.float64)
    # CCW winding for outward normals (verified via cross-product)
    faces = [
        [0, 2, 1],  # top-front-right
        [0, 3, 2],  # top-left-front
        [0, 4, 3],  # top-back-left
        [0, 1, 4],  # top-right-back
        [5, 1, 2],  # bottom-right-front
        [5, 2, 3],  # bottom-front-left
        [5, 3, 4],  # bottom-left-back
        [5, 4, 1],  # bottom-back-right
    ]
    return {"vertices": vertices, "faces": faces, "color": color}


def make_cylinder(base_center, radius, height, color, n_seg=16):
    """Create a cylinder mesh (Y-axis aligned).

    Args:
        base_center: (x, y, z) center of the bottom circle
        radius: cylinder radius
        height: cylinder height
        color: (B, G, R) base color
        n_seg: number of circumference segments

    Returns:
        Mesh dict
    """
    cx, cy, cz = base_center
    vertices = []
    faces = []

    # Bottom circle center
    vertices.append([cx, cy, cz])
    # Top circle center
    vertices.append([cx, cy + height, cz])

    # Bottom ring vertices (indices 2 .. 2+n_seg-1)
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        x = cx + radius * np.cos(theta)
        z = cz + radius * np.sin(theta)
        vertices.append([x, cy, z])

    # Top ring vertices (indices 2+n_seg .. 2+2*n_seg-1)
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        x = cx + radius * np.cos(theta)
        z = cz + radius * np.sin(theta)
        vertices.append([x, cy + height, z])

    vertices = np.array(vertices)

    bot_start = 2
    top_start = 2 + n_seg

    # Bottom cap (facing -Y)
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([0, bot_start + i_next, bot_start + i])

    # Top cap (facing +Y)
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([1, top_start + i, top_start + i_next])

    # Side quads
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([
            bot_start + i,
            bot_start + i_next,
            top_start + i_next,
            top_start + i,
        ])

    return {"vertices": vertices, "faces": faces, "color": color}


def make_ground_grid(y, extent, spacing, color):
    """Create a wireframe ground grid.

    Args:
        y: height of the grid plane
        extent: half-extent of the grid
        spacing: distance between grid lines
        color: (B, G, R) line color

    Returns:
        Mesh dict with 2-vertex faces (line segments)
    """
    vertices = []
    faces = []
    idx = 0
    vals = np.arange(-extent, extent + spacing * 0.5, spacing)

    # Lines parallel to Z
    for x in vals:
        vertices.append([x, y, -extent])
        vertices.append([x, y, extent])
        faces.append([idx, idx + 1])
        idx += 2

    # Lines parallel to X
    for z in vals:
        vertices.append([-extent, y, z])
        vertices.append([extent, y, z])
        faces.append([idx, idx + 1])
        idx += 2

    return {"vertices": np.array(vertices), "faces": faces, "color": color}


# =============================================================================
# Camera Math
# =============================================================================

def build_intrinsic(fx, fy, skew, cx, cy):
    """Build the 3x3 intrinsic matrix K.

    Args:
        fx, fy: focal lengths (pixels)
        skew: skew parameter
        cx, cy: principal point (pixels)

    Returns:
        K as (3, 3) ndarray
    """
    return np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1],
    ], dtype=np.float64)


def build_rotation(alpha, beta, gamma):
    """Build rotation matrix from Euler angles. Order: Rz @ Ry @ Rx.

    Args:
        alpha: rotation about X axis (radians)
        beta: rotation about Y axis (radians)
        gamma: rotation about Z axis (radians)

    Returns:
        R as (3, 3) ndarray
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def euler_from_rotation(R):
    """Extract Euler angles (alpha, beta, gamma) from a rotation matrix.
    Assumes Rz @ Ry @ Rx order. Handles gimbal lock approximately.

    Returns:
        (alpha, beta, gamma) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        alpha = np.arctan2(R[2, 1], R[2, 2])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = np.arctan2(R[1, 0], R[0, 0])
    else:
        alpha = np.arctan2(-R[1, 2], R[1, 1])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = 0.0
    return alpha, beta, gamma


def build_extrinsic(rx, ry, rz, tx, ty, tz, scale, camera_frame=False):
    """Build the 3x4 extrinsic matrix [R|t].

    Args:
        rx, ry, rz: Euler angles (radians)
        tx, ty, tz: translation
        scale: scale factor
        camera_frame: if True, parameters describe camera pose in world;
                      if False, parameters are the world-to-camera transform directly

    Returns:
        Tuple of (Rt (3,4), R (3,3))
    """
    R = build_rotation(rx, ry, rz)

    if camera_frame:
        # Sliders describe camera pose in world
        # Camera orientation in world = R
        # Camera position in world = scale * [tx, ty, tz]
        cam_pos = scale * np.array([tx, ty, tz])
        R_ext = R.T
        t_ext = -R.T @ cam_pos
        Rt = np.hstack([R_ext, t_ext.reshape(3, 1)])
        return Rt, R_ext
    else:
        # Sliders directly specify the extrinsic
        R_scaled = scale * R
        t = np.array([tx, ty, tz])
        Rt = np.hstack([R_scaled, t.reshape(3, 1)])
        return Rt, R_scaled


def fov_to_focal(fov_degrees, image_dim_px):
    """Convert field of view angle to focal length in pixels.

    Args:
        fov_degrees: full angle FOV in degrees (clamped to [1, 179])
        image_dim_px: image dimension (width or height) in pixels

    Returns:
        Focal length in pixels
    """
    fov_degrees = np.clip(fov_degrees, 1.0, 179.0)
    fov_rad = np.radians(fov_degrees)
    return (image_dim_px / 2.0) / np.tan(fov_rad / 2.0)


def make_lookat_Rt(eye, target, up=None):
    """Build extrinsic matrix from look-at parameters.

    Args:
        eye: (3,) camera position in world
        target: (3,) point the camera looks at
        up: (3,) world up vector (default [0, 1, 0])

    Returns:
        Rt as (3, 4) ndarray
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    new_up = np.cross(right, forward)

    # CV convention: camera looks along +Z, Y=up, X=right
    # (matches homework where identity R + positive tz = looking down +Z)
    R = np.stack([right, new_up, forward], axis=0)
    t = -R @ eye
    return np.hstack([R, t.reshape(3, 1)])


# =============================================================================
# Software Renderer
# =============================================================================

def render_scene(meshes, K, Rt, img_w, img_h, light_dir=None, bg_color=(40, 40, 40),
                 flip_y=True):
    """Render meshes to a BGR uint8 image using painter's algorithm.

    Args:
        meshes: list of mesh dicts
        K: (3, 3) intrinsic matrix
        Rt: (3, 4) extrinsic matrix
        img_w, img_h: output image dimensions
        light_dir: (3,) light direction vector (points toward light). Default: (0.3, -0.8, 0.5)
        bg_color: (B, G, R) background color
        flip_y: if True, flip image vertically for upright display
                (compensates for pinhole inversion, matching standard camera display)

    Returns:
        BGR uint8 image of shape (img_h, img_w, 3)
    """
    if light_dir is None:
        light_dir = np.array([0.3, -0.8, 0.5])
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-12)

    M = K @ Rt
    R = Rt[:, :3]

    img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

    # Collect all renderable faces with depth and color
    face_list = []  # (mean_depth, projected_pts, color, is_line)

    for mesh in meshes:
        verts = mesh["vertices"]
        base_color = np.array(mesh["color"], dtype=np.float64)

        # Project all vertices: homogeneous world coords (N, 4)
        N = len(verts)
        verts_h = np.hstack([verts, np.ones((N, 1))])

        # Project to image: (3, N)
        projected = M @ verts_h.T  # (3, N)
        depths = projected[2]  # (N,)

        # Perspective divide (avoid division by zero)
        valid_mask = depths > 0.01
        uv = np.zeros((2, N))
        uv[:, valid_mask] = projected[:2, valid_mask] / depths[valid_mask]

        for face_idx in mesh["faces"]:
            face_idx = list(face_idx)
            n_verts = len(face_idx)

            # Check if all vertices are in front of camera
            face_depths = depths[face_idx]
            if np.any(face_depths <= 0.01):
                continue

            mean_depth = np.mean(face_depths)
            pts_2d = uv[:, face_idx].T  # (n_verts, 2)

            if n_verts == 2:
                # Line segment
                face_list.append((mean_depth, pts_2d.astype(np.int32), base_color, True))
            else:
                # Polygon face - compute normal for shading and back-face culling
                v0 = verts[face_idx[0]]
                v1 = verts[face_idx[1]]
                v2 = verts[face_idx[2]]
                normal_world = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal_world)
                if norm_len < 1e-12:
                    continue
                normal_world = normal_world / norm_len

                # Back-face culling: transform normal to camera space
                normal_cam = R @ normal_world
                if normal_cam[2] > 0:
                    # Normal points away from camera (positive Z = away in camera space)
                    continue

                # Lambertian shading
                intensity = np.clip(np.dot(normal_world, light_dir), 0.0, 1.0)
                intensity = 0.35 + 0.65 * intensity  # ambient + diffuse
                shaded = np.clip(base_color * intensity, 0, 255).astype(np.uint8)

                face_list.append((mean_depth, pts_2d.astype(np.int32), shaded, False))

    # Sort by depth: far to near (painter's algorithm)
    face_list.sort(key=lambda f: -f[0])

    # Draw
    for _, pts, color, is_line in face_list:
        color_tuple = tuple(int(c) for c in color)
        if is_line:
            if len(pts) >= 2:
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), color_tuple, 1, cv2.LINE_AA)
        else:
            # Filled polygon
            cv2.fillPoly(img, [pts], color_tuple)
            # Wireframe outline (slightly darker)
            outline_color = tuple(max(0, int(c * 0.5)) for c in color)
            cv2.polylines(img, [pts], True, outline_color, 1, cv2.LINE_AA)

    if flip_y:
        img = cv2.flip(img, 0)

    return img


# =============================================================================
# Frustum Visualization
# =============================================================================

def compute_frustum_corners(K, Rt, img_w, img_h, near=0.3, far=5.0):
    """Compute the 8 world-space corners of the camera frustum.

    Args:
        K: (3, 3) intrinsic matrix
        Rt: (3, 4) extrinsic matrix
        img_w, img_h: image dimensions
        near, far: near and far plane distances

    Returns:
        (cam_pos (3,), near_corners (4,3), far_corners (4,3))
    """
    K_inv = np.linalg.inv(K)
    R = Rt[:, :3]
    t = Rt[:, 3]

    # Camera position in world
    cam_pos = -R.T @ t

    # Image corners in pixel coordinates (homogeneous)
    corners_px = np.array([
        [0,     0,     1],  # top-left
        [img_w, 0,     1],  # top-right
        [img_w, img_h, 1],  # bottom-right
        [0,     img_h, 1],  # bottom-left
    ], dtype=np.float64).T  # (3, 4)

    # Ray directions in camera space
    rays_cam = K_inv @ corners_px  # (3, 4)

    # Scale to near and far depths
    near_pts_cam = rays_cam * (near / rays_cam[2:3])
    far_pts_cam = rays_cam * (far / rays_cam[2:3])

    # Transform to world space: p_world = R^T @ (p_cam - t)
    near_pts_world = (R.T @ (near_pts_cam - t.reshape(3, 1))).T  # (4, 3)
    far_pts_world = (R.T @ (far_pts_cam - t.reshape(3, 1))).T    # (4, 3)

    return cam_pos, near_pts_world, far_pts_world


def make_frustum_mesh(K, Rt, img_w, img_h, near=0.3, far=5.0):
    """Create a wireframe mesh for the camera frustum.

    Returns a list of mesh dicts (frustum lines + camera origin marker).
    """
    cam_pos, near_corners, far_corners = compute_frustum_corners(K, Rt, img_w, img_h, near, far)

    # Combine all vertices
    # 0: cam_pos, 1-4: near corners, 5-8: far corners
    vertices = np.vstack([
        cam_pos.reshape(1, 3),
        near_corners,
        far_corners,
    ])

    faces = []

    # Edges from camera origin to near corners
    for i in range(4):
        faces.append([0, 1 + i])

    # Near plane edges
    for i in range(4):
        faces.append([1 + i, 1 + (i + 1) % 4])

    # Far plane edges
    for i in range(4):
        faces.append([5 + i, 5 + (i + 1) % 4])

    # Connecting edges (near to far)
    for i in range(4):
        faces.append([1 + i, 5 + i])

    frustum_mesh = {
        "vertices": vertices,
        "faces": faces,
        "color": (0, 200, 255),  # yellow-orange in BGR
    }

    return [frustum_mesh]


def make_axis_mesh(origin, length=1.0):
    """Create axis indicator meshes (3 colored line segments).

    Args:
        origin: (3,) origin point
        length: axis length

    Returns:
        List of 3 mesh dicts (X=red, Y=green, Z=blue in BGR)
    """
    o = np.asarray(origin, dtype=np.float64)
    meshes = []
    # X axis (red in BGR = (0, 0, 255))
    meshes.append({
        "vertices": np.array([o, o + [length, 0, 0]]),
        "faces": [[0, 1]],
        "color": (0, 0, 255),
    })
    # Y axis (green = (0, 255, 0))
    meshes.append({
        "vertices": np.array([o, o + [0, length, 0]]),
        "faces": [[0, 1]],
        "color": (0, 255, 0),
    })
    # Z axis (blue = (255, 0, 0))
    meshes.append({
        "vertices": np.array([o, o + [0, 0, length]]),
        "faces": [[0, 1]],
        "color": (255, 0, 0),
    })
    return meshes


def make_camera_axes_mesh(Rt, length=0.8):
    """Create axis indicators at the camera position, aligned to camera axes.

    Args:
        Rt: (3, 4) extrinsic matrix
        length: axis length

    Returns:
        List of 3 mesh dicts for camera X/Y/Z axes in world space
    """
    R = Rt[:, :3]
    t = Rt[:, 3]
    cam_pos = -R.T @ t

    # Camera axes in world space
    cam_right = R.T @ np.array([1, 0, 0]) * length
    cam_up = R.T @ np.array([0, 1, 0]) * length
    cam_forward = R.T @ np.array([0, 0, 1]) * length

    meshes = []
    # X axis (red)
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_right]),
        "faces": [[0, 1]],
        "color": (0, 0, 255),
    })
    # Y axis (green)
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_up]),
        "faces": [[0, 1]],
        "color": (0, 255, 0),
    })
    # Z axis (blue) - note: camera looks down -Z, so this points backward
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_forward]),
        "faces": [[0, 1]],
        "color": (255, 0, 0),
    })
    return meshes


# =============================================================================
# Default Scene
# =============================================================================

def create_default_scene():
    """Create the default 3D scene with basic primitives.

    Returns:
        List of mesh dicts
    """
    return [
        make_cube(center=(-1.5, 0.5, 0), size=1.0, color=(80, 80, 220)),
        make_cube(center=(1.5, 0.5, -1), size=0.7, color=(220, 80, 80)),
        make_sphere(center=(0, 0.8, -2), radius=0.6, color=(80, 220, 80)),
        make_cylinder(base_center=(2, 0, 1), radius=0.3, height=1.5, color=(80, 220, 220)),
        make_ground_grid(y=0.0, extent=5.0, spacing=1.0, color=(100, 100, 100)),
    ]


# =============================================================================
# Matrix Formatting
# =============================================================================

def format_matrix(mat, label=""):
    """Format a matrix as a string for display.

    Args:
        mat: 2D ndarray
        label: optional label prefix

    Returns:
        Formatted string
    """
    rows, cols = mat.shape
    lines = []
    if label:
        lines.append(label)
    for r in range(rows):
        row_str = "  ".join(f"{mat[r, c]:7.2f}" for c in range(cols))
        bracket = "|" if 0 < r < rows - 1 else ("/" if r == 0 else "\\")
        lines.append(f"  {bracket} {row_str} {bracket}")
    return "\n".join(lines)
