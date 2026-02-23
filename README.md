# Demos for CSCI 1430

Interactive computer vision demos for CSCI 1430 at Brown University. The desktop demos (Dear PyGui) are the primary versions, used for live demonstrations in class. The web versions are a convenience for distribution and publishing, allowing students to interact with the demos without installing anything.

## Changelog

- 2026 Spring: Web adapter system — all demos run in-browser via PyScript/Pyodide
- 2026 Spring: Automated deployment to course website via deploy key CI
- 2026 Spring: Reorganized into `data/` and `utils/` directories
- 2026 Spring: Refactored shared code into utility modules (demo_utils, demo_ui, demo_kernels, demo_fft)
- 2026 Spring: Interactive kernel editor with click-to-modify cells and visual value display
- 2026 Spring: New convolution theorem demo with deconvolution
- 2026 Spring: Significantly updated UI with Dear PyGui controls

## Running the Demos

### Option 1: Demo Launcher (Recommended)

```bash
python demo_launcher.py
```

This opens a GUI where you can select and launch any demo.

### Option 2: Run Individual Demos

```bash
python liveFilter.py
python liveCannyEdges.py
python liveFFT.py
python liveConvolutionTheorem.py
python liveHarrisCorners.py
python liveSIFTMatching.py
python liveHOGPersonDetector.py
python liveViolaJones.py
```

## Requirements

- Python environment with OpenCV, NumPy, scikit-image
- Dear PyGui (`pip install dearpygui`)
- Webcam (optional - demos fall back to `cat.jpg` if no camera is detected)

## Available Demos

| Demo                           | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| **Image Filtering**            | Interactive kernel editor with presets (Box, Gaussian, Sharpen, Sobel, Laplacian, LoG, Emboss) and click-to-modify cells |
| **Canny Edge Detection**       | Real-time edge detection with adjustable blur and threshold                 |
| **Fourier Transform**          | Visualize 2D FFT with amplitude/phase manipulation and frequency reconstruction |
| **Convolution Theorem**        | Demonstrates that spatial convolution equals frequency multiplication; includes Wiener deconvolution |
| **Harris Corner Detection**    | Detect corners with tunable parameters and geometric transforms for invariance testing |
| **SIFT Feature Matching**      | Match SIFT features between query images and live camera feed               |
| **HOG Person Detection**       | Detect people using Histogram of Oriented Gradients                         |
| **Viola-Jones Face Detection** | Detect faces using Haar cascade classifiers                                 |

## Project Structure

```text
demos/
├── data/                    # Images and data files
│   ├── cat.jpg              # Fallback image when no webcam
│   ├── macmillan115_*.jpg   # SIFT matching query images
│   ├── salomon001_*.jpg     # SIFT matching query images
│   └── ...
├── utils/                   # Shared utility modules
│   ├── __init__.py
│   ├── demo_utils.py        # Camera init, image conversion, transforms
│   ├── demo_ui.py           # Viewport setup, state updaters, callbacks
│   ├── demo_kernels.py      # Kernel presets and creation functions
│   └── demo_fft.py          # FFT visualization and processing
├── web/                     # Web adapter system (see below)
│   ├── adapter.py           # Generic PyScript adapter
│   ├── build_web.py         # Build script
│   ├── demo.html            # Single-page demo shell
│   ├── style.css            # Shared styles
│   └── frames/              # Per-demo web frame modules
├── .github/workflows/       # CI deployment
│   └── deploy-web-demos.yml
├── demo_launcher.py         # GUI launcher for all demos
├── liveFilter.py            # Image filtering demo
├── liveCannyEdges.py        # Canny edge detection demo
├── liveFFT.py               # Fourier transform demo
├── liveConvolutionTheorem.py# Convolution theorem demo
├── liveHarrisCorners.py     # Harris corner detection demo
├── liveSIFTMatching.py      # SIFT feature matching demo
├── liveHOGPersonDetector.py # HOG person detection demo
├── liveViolaJones.py        # Viola-Jones face detection demo
└── README.md
```

## Desktop Demos (Dear PyGui)

All desktop demos use [Dear PyGui](https://github.com/hoffstadt/DearPyGui) for the GUI, providing:

- Real-time sliders and controls for parameter adjustment
- Live video display from webcam or fallback image
- Responsive layouts that adapt to window resizing
- Consistent UI scaling for presentations

Each demo follows a common pattern:

1. Initialize camera (or load fallback image if unavailable)
2. Create Dear PyGui context with textures and controls
3. Main loop: capture frame, process with OpenCV/NumPy, update display
4. Cleanup on exit

Shared utilities in `utils/` reduce code duplication and ensure consistent behavior across all demos.

## Web Adapter System

The `web/` directory contains a system that runs the same demos in a browser using [PyScript](https://pyscript.net/) (Python via WebAssembly/Pyodide). No server-side Python is needed — everything runs client-side with NumPy and OpenCV compiled to WASM.

### How it works

Each desktop demo (e.g., `liveFilter.py`) has a corresponding **frame module** in `web/frames/` (e.g., `frames/filter.py`). A frame module exports:

- **`WEB_CONFIG`** — a dict declaring the demo's title, description, output canvases, UI controls, and layout
- **`web_frame(state)`** — a function called each frame with the current control values and camera image; returns a dict of output images and a status string

The generic **adapter** (`web/adapter.py`) loads whichever frame module is specified in the URL (`demo.html?demo=filter`), reads its `WEB_CONFIG`, and auto-builds the HTML controls (sliders, dropdowns, checkboxes, buttons) and output canvases. For camera demos, it runs an async loop that grabs webcam frames, calls `web_frame()`, and renders the results to `<canvas>` elements. For slider-only demos (no camera), it updates on control changes.

This means adding a new web demo only requires writing a new frame module — the adapter handles all the UI scaffolding, camera access, and rendering.

### WEB_CONFIG reference

```python
WEB_CONFIG = {
    "title": "Demo Title",
    "description": "One-line description shown at the top.",
    "camera": {"width": 320, "height": 240},   # omit for slider-only demos
    "outputs": [
        {"id": "edges", "label": "Canny Edges", "width": 320, "height": 240},
    ],
    "controls": {
        "threshold": {
            "group": "Detection",               # optional: collapsible group
            "type": "float",                     # float | int | bool | choice | button
            "min": 0.0, "max": 1.0, "step": 0.01,
            "default": 0.5,
            "label": "Threshold",
            "format": ".2f",                     # optional: display format
            "visible_when": {"mode": ["Advanced"]},  # optional: conditional visibility
        },
    },
    "mouse": ["canvas_id"],                      # optional: enable mouse events
    "layout": {"rows": [["edges"]]},             # arrange output canvases in rows
}
```

### Optional frame module hooks

- **`web_frame(state) -> dict`** — required; process one frame
- **`web_button(button_id)`** — called when a `"type": "button"` control is clicked
- **`web_mouse(event)`** — called for mouse events on canvases listed in `"mouse"`

### Control grouping

Controls with the same `"group"` key are wrapped in a collapsible `<details>` section, matching the `dpg.collapsing_header()` sections in the desktop demos. Controls without a `"group"` render ungrouped at the top level.

### Build process

The build script (`web/build_web.py`) prepares the web directory for serving:

1. Reads `demo_launcher.py`'s `SECTIONS` metadata to discover which demos exist
2. Copies `utils/*.py` and demo source files into `web/vendor/` (so PyScript can import them)
3. Copies `data/` files (images, cascade XMLs) into `web/data/`
4. Generates `pyscript.toml` with the file mappings for Pyodide's virtual filesystem
5. Generates `index.html` with a card for each available demo
6. Validates that no `.py` files are empty (empty files break on GitHub Pages)

To build and test locally:

```bash
cd demos
python web/build_web.py
python -m http.server 8000
# Open http://localhost:8000/web/demo.html?demo=filter
```

### Deployment

Deployment to the course website is fully automated via GitHub Actions.

When code is pushed to `main` on this repo, the workflow (`.github/workflows/deploy-web-demos.yml`) runs:

1. Checks out the demos repo
2. Runs `python demos/web/build_web.py` to generate all web artifacts
3. Uses a **deploy key** (SSH keypair) to clone the website repo (`browncsci1430.github.io`, branch `2026_Spring`)
4. Copies the built files into `website/resources/interactive_demos/`
5. Updates the submodule pointer at `website/resources/interactive_demos_python/`
6. Commits and pushes to the website repo (if anything changed)

The deploy key is stored as a repository secret (`WEBSITE_DEPLOY_KEY`) on this repo, with the corresponding public key added as a deploy key (with write access) on the website repo.

There is also a manual fallback workflow on the website repo (`build-web-demos.yml`) that can be triggered via `workflow_dispatch` if needed.

## Authors

James Tompkin (james_tompkin@brown.edu)
Srinath Sridhar (srinath@brown.edu)
