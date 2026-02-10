"""
Demo Launcher - CSCI 1430 Brown University
Select and run computer vision demos with Dear PyGui controls
"""

import os
import sys
import subprocess
import dearpygui.dearpygui as dpg

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Available demos with descriptions
DEMOS = {
    "Starter Template": {
        "file": "liveStarter.py",
        "description": "Minimal template for students to experiment with image processing. Edit process_frame() to see your changes!",
        "default_resolution": "640x480"
    },
    "Image Filtering": {
        "file": "liveFilter.py",
        "description": "Apply convolution filters with adjustable kernel size.",
        "default_resolution": "640x480"
    },
    "Canny Edge Detection": {
        "file": "liveCannyEdges.py",
        "description": "Real-time Canny edge detection with adjustable blur sigma and threshold parameters.",
        "default_resolution": "640x480"
    },
    "Fourier Transform": {
        "file": "liveFFT.py",
        "description": "Visualize 2D Fourier transforms with multiple modes: normal FFT, DC only, rotating dot, and frequency reconstruction.",
        "default_resolution": "320x240"
    },
    "Convolution Theorem": {
        "file": "liveConvolutionTheorem.py",
        "description": "Demonstrates convolution theorem: spatial convolution equals frequency multiplication. Includes deconvolution mode with regularization.",
        "default_resolution": "320x240"
    },
    "Harris Corner Detection": {
        "file": "liveHarrisCorners.py",
        "description": "Detect corners using Harris corner detector with tunable parameters.",
        "default_resolution": "640x480"
    },
    "SIFT Feature Matching": {
        "file": "liveSIFTMatching.py",
        "description": "Match SIFT features between a query image and live camera feed.",
        "default_resolution": "640x480"
    },
    "HOG Person Detection": {
        "file": "liveHOGPersonDetector.py",
        "description": "Detect people using Histogram of Oriented Gradients (HOG) features.",
        "default_resolution": "640x480"
    },
    "Viola-Jones Face Detection": {
        "file": "liveViolaJones.py",
        "description": "Detect faces using Haar cascade classifiers with multiple cascade options.",
        "default_resolution": "640x480"
    },
}

DEFAULT_UI_SCALE = 1.5

class LauncherState:
    selected_demo = None
    running_process = None

state = LauncherState()

def update_ui_scale(sender, value):
    dpg.set_global_font_scale(value)

def select_demo(sender, app_data, user_data):
    """Handle demo selection"""
    state.selected_demo = user_data
    demo_info = DEMOS[user_data]
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

    # Theme for selected items
    with dpg.theme() as selected_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 130, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 149, 237))

    with dpg.window(label="CSCI 1430 Demo Launcher", tag="main_window"):
        dpg.add_text("Computer Vision Demos", color=(100, 200, 255))
        dpg.add_text("Brown University - CSCI 1430", color=(150, 150, 150))
        dpg.add_separator()
        dpg.add_spacer(height=10)

        dpg.add_slider_float(
            label="UI Scale",
            default_value=DEFAULT_UI_SCALE,
            min_value=1.0,
            max_value=3.0,
            callback=update_ui_scale,
            width=200
        )
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Camera resolution control
        dpg.add_text("Camera Settings:")
        with dpg.group(horizontal=True):
            dpg.add_combo(
                label="Resolution",
                items=["Default", "320x240", "640x480", "1280x720", "1920x1080", "3840x2160"],
                default_value="Default",
                tag="resolution_combo",
                width=150
            )
            dpg.add_text("?", color=(150, 150, 150))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Camera resolution to request.\nNote: Not all cameras support all resolutions.\n'Default' uses the demo's recommended resolution.")
        dpg.add_separator()
        dpg.add_spacer(height=10)

        dpg.add_text("Select a demo to run:")
        dpg.add_spacer(height=5)

        # Demo buttons
        for demo_name in DEMOS:
            dpg.add_button(
                label=demo_name,
                width=350,
                height=35,
                callback=select_demo,
                user_data=demo_name
            )
            dpg.add_spacer(height=2)

        dpg.add_spacer(height=15)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Description area
        dpg.add_text("Description:", color=(150, 150, 150))
        dpg.add_text("Select a demo to see its description.", tag="description_text", wrap=380)

        dpg.add_spacer(height=15)

        # Run button
        dpg.add_button(
            label="Run Selected Demo",
            width=350,
            height=40,
            callback=run_demo,
            tag="run_button",
            enabled=False
        )

        dpg.add_spacer(height=10)
        dpg.add_text("", tag="status_text", color=(200, 200, 100))

        dpg.add_spacer(height=20)
        dpg.add_separator()
        dpg.add_spacer(height=5)
        dpg.add_text("Note: Each demo requires a webcam.", color=(150, 150, 150))
        dpg.add_text("Close the demo window to return here.", color=(150, 150, 150))

    dpg.create_viewport(title="CSCI 1430 - Demo Launcher", width=500, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    dpg.set_global_font_scale(DEFAULT_UI_SCALE)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    main()
