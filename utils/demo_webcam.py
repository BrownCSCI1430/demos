"""
Physical webcam capture and fallback image handling for CSCI 1430 demos.

Separates hardware device concerns (USB camera, frame acquisition, fallback)
from the mathematical pinhole camera model (demo_3d.py) and image processing
helpers (demo_utils.py).
"""

import os
import sys
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_ui import load_fonts

# Data directory (sibling to utils/)
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_UTILS_DIR), "data")


# =============================================================================
# Camera Hardware
# =============================================================================

def probe_cameras(max_index=8):
    """Return list of working camera indices.

    Tries indices 0..max_index-1 and returns those that successfully open
    and read a frame. Returns [0] if none are found.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            available.append(i)
        cap.release()
    return available if available else [0]


def init_camera(camera_id=0, width=None, height=None):
    """Initialize camera with platform-specific backend.

    Args:
        camera_id: Camera device ID (default 0)
        width: Requested frame width (None for default)
        height: Requested frame height (None for default)

    Returns:
        Tuple of (cap, frame_width, frame_height, use_camera)
        - cap: cv2.VideoCapture object or None
        - frame_width: Width of camera frames (0 if no camera)
        - frame_height: Height of camera frames (0 if no camera)
        - use_camera: True if camera is available and working
    """
    if os.name == 'nt':
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        if ret:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width is not None and height is not None:
                if actual_width != width or actual_height != height:
                    print(f"Camera: Requested {width}x{height}, got {actual_width}x{actual_height}")
                else:
                    print(f"Camera: Using {actual_width}x{actual_height}")
            else:
                print(f"Camera: Default resolution {actual_width}x{actual_height}")

            return cap, actual_width, actual_height, True
        cap.release()

    return None, 0, 0, False


# =============================================================================
# Fallback Image
# =============================================================================

def load_fallback_image(data_dir=None, filename="cat.jpg"):
    """Load fallback image for cat mode.

    Args:
        data_dir: Directory containing the fallback image (defaults to DATA_DIR)
        filename: Name of the fallback image file

    Returns:
        Loaded image as numpy array

    Exits:
        If image cannot be loaded
    """
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: Fallback image not found at {path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load fallback image", file=sys.stderr)
        sys.exit(1)

    return img


# =============================================================================
# Frame Acquisition
# =============================================================================

def resize_with_letterbox(img, target_width, target_height):
    """Resize image to fit within target dimensions, maintaining aspect ratio with black bars."""
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_height, target_width, 3), dtype=img.dtype)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def get_frame(cap, fallback_image, use_camera, cat_mode, target_size=None, letterbox=True):
    """Get frame from camera or fallback image.

    Args:
        cap: cv2.VideoCapture object
        fallback_image: Fallback image to use when camera unavailable
        use_camera: Whether camera is available
        cat_mode: Whether to use fallback image instead of camera
        target_size: Optional (width, height) tuple to resize frame
        letterbox: If True, use letterboxing for fallback; if False, stretch

    Returns:
        Frame as numpy array, or None if camera read failed
    """
    if use_camera and not cat_mode:
        ret, frame = cap.read()
        if not ret:
            return None
        if target_size:
            frame = cv2.resize(frame, target_size)
    else:
        frame = fallback_image.copy()
        if target_size:
            if letterbox:
                frame = resize_with_letterbox(frame, target_size[0], target_size[1])
            else:
                frame = cv2.resize(frame, target_size)
    return frame


# =============================================================================
# Demo Startup / Teardown
# =============================================================================

def init_camera_demo(state, description="Demo", camera_id=None):
    """Parse CLI args, probe cameras, init capture + fallback, create DPG context.

    Sets state.cap, state.frame_width, state.frame_height, state.use_camera,
    state.fallback_image, state.cat_mode, state.camera_id, and state.camera_list.

    Args:
        state: State object to populate.
        description: argparse description string.
        camera_id: Override camera index (skips --camera-id arg).

    Returns:
        (frame_width, frame_height) for convenience.
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--width', type=int, default=None, help='Camera width')
    parser.add_argument('--height', type=int, default=None, help='Camera height')
    parser.add_argument('--camera-id', type=int, default=None, help='Camera device index')
    args = parser.parse_args()

    # Probe available cameras
    state.camera_list = probe_cameras()
    selected_id = camera_id if camera_id is not None else (args.camera_id or state.camera_list[0])

    state.cap, state.frame_width, state.frame_height, state.use_camera = \
        init_camera(selected_id, width=args.width, height=args.height)
    state.camera_id = selected_id

    if not state.use_camera:
        print("Warning: Could not open camera, using fallback image")

    state.fallback_image = load_fallback_image()
    if not state.use_camera:
        state.frame_height, state.frame_width = state.fallback_image.shape[:2]
        state.cat_mode = True

    dpg.create_context()
    load_fonts()

    return state.frame_width, state.frame_height


def switch_camera(state, camera_id, width=None, height=None):
    """Switch to a different camera device. Releases old capture.

    Args:
        state: State object with cap, frame_width, frame_height, etc.
        camera_id: New camera device index.
        width: Request width (defaults to current frame_width).
        height: Request height (defaults to current frame_height).
    """
    if state.cap is not None:
        state.cap.release()
    w = width or state.frame_width
    h = height or state.frame_height
    state.cap, _, _, state.use_camera = init_camera(camera_id, width=w, height=h)
    state.camera_id = camera_id
    if not state.use_camera:
        state.cat_mode = True


def cleanup_camera_demo(state):
    """Release camera capture and destroy DPG context."""
    if getattr(state, 'use_camera', False) and getattr(state, 'cap', None) is not None:
        state.cap.release()
    dpg.destroy_context()
