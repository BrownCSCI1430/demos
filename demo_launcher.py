"""
Demo Launcher - CSCI 1430 Brown University
Select and run computer vision demos with Dear PyGui controls
"""

import os
import sys
import subprocess
import dearpygui.dearpygui as dpg
from utils.demo_ui import load_fonts

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Available demos organised by course topic.
# Each section is (heading, colour, dict of demos).
SECTIONS = [
    ("Getting Started", (120, 220, 120), {
        "Starter Template": {
            "file": "liveStarter.py",
            "description": "Minimal template for students to experiment with image processing. Edit process_frame() to see your changes!",
            "default_resolution": "640x480",
        },
    }),
    ("Image Processing (HW1)", (100, 200, 255), {
        "Image Filtering": {
            "file": "liveFilter.py",
            "description": "Apply convolution filters with adjustable kernel size.",
            "default_resolution": "640x480",
        },
        "Canny Edge Detection": {
            "file": "liveCannyEdges.py",
            "description": "Real-time Canny edge detection with adjustable blur sigma and threshold parameters.",
            "default_resolution": "640x480",
        },
        "Fourier Transform": {
            "file": "liveFourier.py",
            "description": "Visualize 2D Fourier transforms with multiple modes: normal FFT, DC only, rotating dot, and frequency reconstruction.",
            "default_resolution": "320x240",
        },
        "Convolution Theorem": {
            "file": "liveConvolutionTheorem.py",
            "description": "Demonstrates convolution theorem: spatial convolution equals frequency multiplication. Includes deconvolution mode with regularization.",
            "default_resolution": "320x240",
        },
    }),
    ("Feature Detection & Matching (HW2)", (255, 200, 100), {
        "Harris Corner Detection": {
            "file": "liveHarrisCorners.py",
            "description": "Detect corners using Harris corner detector with tunable parameters.",
            "default_resolution": "640x480",
        },
        "SIFT Feature Matching": {
            "file": "liveSIFTMatching.py",
            "description": "Match SIFT features between a query image and live camera feed.",
            "default_resolution": "640x480",
        },
    }),
    ("Camera Geometry (HW3)", (200, 180, 255), {
        "3D Camera": {
            "file": "liveCamera.py",
            "description": "Interactive 3D camera demo. Explore intrinsic (K) and extrinsic [R|t] matrices with real-time software rendering. Toggle between world and camera reference frames.",
            "default_resolution": "400x400",
        },
        "Camera Calibration (DLT)": {
            "file": "liveCalibration.py",
            "description": "Synthetic DLT calibration demo. Drag the noise slider to add pixel noise to 2D\u20133D correspondences and watch the reprojection error and condition number change. Toggle Hartley normalization to see its effect on numerical stability.",
            "default_resolution": "480x480",
        },
        "Plane Sweep Stereo": {
            "file": "livePlaneSweep.py",
            "description": "Depth-dependent homography H(\u03bb) demo. Sweep the depth slider to warp one camera view into the other. NCC peaks when \u03bb matches the true scene depth, showing the plane sweep stereo principle (HW3 Task 2).",
            "default_resolution": "400x400",
        },
        "Sparse Triangulation": {
            "file": "liveTriangulation.py",
            "description": "Click correspondences in two camera views to triangulate 3D points. The epipolar line guides you to the correct match. Cheirality check shown in green (valid) or red (behind camera). Uses the 4\u00d74 DLT system (HW3 Task 5).",
            "default_resolution": "420x380",
        },
    }),
    ("Object Detection", (255, 150, 150), {
        "HOG Person Detection": {
            "file": "liveHOGPersonDetector.py",
            "description": "Detect people using Histogram of Oriented Gradients (HOG) features.",
            "default_resolution": "640x480",
        },
        "Viola-Jones Face Detection": {
            "file": "liveViolaJones.py",
            "description": "Detect faces using Haar cascade classifiers with multiple cascade options.",
            "default_resolution": "640x480",
        },
    }),
]

# Flat lookup used by callbacks
DEMOS = {}
for _heading, _color, _demos in SECTIONS:
    DEMOS.update(_demos)

DEFAULT_UI_SCALE = 1.5

class LauncherState:
    selected_demo = None
    running_process = None

state = LauncherState()

UI_SCALES = ["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"]

def update_ui_scale(sender, value):
    dpg.set_global_font_scale(float(value))

def select_demo(sender, app_data, user_data):
    """Handle demo selection"""
    state.selected_demo = user_data
    demo_info = DEMOS[user_data]
    dpg.set_value("selected_name_text", user_data)
    dpg.set_value("selected_file_text", demo_info["file"])
    dpg.set_value("description_text", demo_info["description"])

    # Set resolution combo to "Default" to use demo's default resolution
    dpg.set_value("resolution_combo", "Default")

    dpg.configure_item("run_button", enabled=True)

def run_demo():
    """Launch the selected demo"""
    if state.selected_demo is None:
        return

    demo_file = DEMOS[state.selected_demo]["file"]
    demo_path = os.path.join(SCRIPT_DIR, demo_file)

    if not os.path.exists(demo_path):
        dpg.set_value("status_text", f"Error: {demo_file} not found")
        return

    dpg.set_value("status_text", f"Launching {state.selected_demo}...")

    # Get resolution setting
    resolution = dpg.get_value("resolution_combo")
    cmd = [sys.executable, demo_path]

    # Use demo's default if "Default" is selected, otherwise use the selected resolution
    if resolution == "Default":
        resolution = DEMOS[state.selected_demo].get("default_resolution", "640x480")

    if resolution and 'x' in resolution:
        width, height = resolution.split('x')
        cmd.extend(['--width', width, '--height', height])

    # Launch demo as subprocess
    try:
        if sys.platform == 'win32':
            # On Windows, use CREATE_NEW_CONSOLE to give it its own window
            state.running_process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            state.running_process = subprocess.Popen(cmd)

        dpg.set_value("status_text", f"Running: {state.selected_demo} ({resolution})")
    except Exception as e:
        dpg.set_value("status_text", f"Error: {str(e)}")

def main():
    dpg.create_context()

    load_fonts()

    with dpg.window(label="CSCI 1430 Demo Launcher", tag="main_window"):
        # ── Header ──────────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_text("Computer Vision Demos", color=(100, 200, 255))
            dpg.add_spacer(width=20)
            dpg.add_text("Brown University - CSCI 1430", color=(150, 150, 150))
            dpg.add_spacer(width=40)
            dpg.add_combo(
                label="UI Scale",
                items=UI_SCALES,
                default_value=str(DEFAULT_UI_SCALE),
                callback=update_ui_scale,
                width=80,
            )
        dpg.add_separator()

        # ── Two-column body ─────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ── Left column: demo list ──────────────────────────────────────
            with dpg.child_window(width=310, border=False):
                for heading, color, demos in SECTIONS:
                    dpg.add_text(heading, color=color)
                    for demo_name in demos:
                        dpg.add_button(
                            label=f"  {demo_name}",
                            width=290,
                            height=28,
                            callback=select_demo,
                            user_data=demo_name,
                        )
                        dpg.add_spacer(height=1)
                    dpg.add_spacer(height=5)

            dpg.add_spacer(width=10)

            # ── Right column: details + run ─────────────────────────────────
            with dpg.child_window(border=False):
                dpg.add_text("Select a demo", tag="selected_name_text",
                             color=(255, 255, 255))
                dpg.add_text("", tag="selected_file_text",
                             color=(120, 120, 120))
                dpg.add_spacer(height=8)
                dpg.add_separator()
                dpg.add_spacer(height=8)

                dpg.add_text("Description:", color=(150, 150, 150))
                dpg.add_text("Click a demo on the left to see its description.",
                             tag="description_text", wrap=340)

                dpg.add_spacer(height=15)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                # Camera resolution
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Resolution",
                        items=["Default", "320x240", "640x480",
                               "1280x720", "1920x1080", "3840x2160"],
                        default_value="Default",
                        tag="resolution_combo",
                        width=130,
                    )
                    dpg.add_text("?", color=(150, 150, 150))
                    with dpg.tooltip(dpg.last_item()):
                        dpg.add_text(
                            "Camera resolution to request.\n"
                            "Not all cameras support all resolutions.\n"
                            "'Default' uses the demo's recommended resolution.")
                dpg.add_text("Most demos require a webcam.",
                             color=(150, 150, 150))

                dpg.add_spacer(height=15)

                # Run button
                dpg.add_button(
                    label="Run Selected Demo",
                    width=280,
                    height=40,
                    callback=run_demo,
                    tag="run_button",
                    enabled=False,
                )

                dpg.add_spacer(height=10)
                dpg.add_text("", tag="status_text", color=(200, 200, 100))
                dpg.add_spacer(height=5)
                dpg.add_text("Close the demo window to return here.",
                             color=(120, 120, 120))

    dpg.create_viewport(title="CSCI 1430 - Demo Launcher", width=1100, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    dpg.set_global_font_scale(DEFAULT_UI_SCALE)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    main()
