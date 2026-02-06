# Demos for CSCI 1430

Interactive computer vision demos for CSCI 1430 at Brown University.

## Changelog

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
Demos/
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

## How They're Built

All demos use [Dear PyGui](https://github.com/hoffstadt/DearPyGui) for the GUI, providing:

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

## Authors

James Tompkin (james_tompkin@brown.edu)
Srinath Sridhar (srinath@brown.edu)
