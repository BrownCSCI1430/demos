"""
Build script for CSCI 1430 web demos.

Reads demo_launcher.py SECTIONS metadata, copies needed demo files into
web/vendor/, and generates pyscript.toml + index.html.

Usage:
    uv run python web/build_web.py
"""

import os
import re
import shutil
import sys
from unittest.mock import MagicMock
from html import escape

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Auto-detect DEMOS_DIR: works whether web/ is sibling to demos/ or inside it.
#   Layout A: BrownCSCI1430/web/ + BrownCSCI1430/demos/  (local dev)
#   Layout B: demos/web/  (web/ inside demos repo)
#   Layout C: website_repo/demos/web/  (submodule, same as B from web/'s perspective)
_candidate = os.path.join(REPO_ROOT, "demos")
if os.path.isdir(_candidate) and os.path.isfile(os.path.join(_candidate, "demo_launcher.py")):
    DEMOS_DIR = _candidate
elif os.path.isfile(os.path.join(REPO_ROOT, "demo_launcher.py")):
    DEMOS_DIR = REPO_ROOT
else:
    sys.exit("Cannot find demos directory (expected demo_launcher.py in parent or parent/demos/)")

VENDOR_DIR = os.path.join(SCRIPT_DIR, "vendor")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ── Mock DearPyGui so we can import demo_launcher ──────────────────────
_mock = MagicMock()
sys.modules["dearpygui"] = _mock
sys.modules["dearpygui.dearpygui"] = _mock
sys.path.insert(0, DEMOS_DIR)

from demo_launcher import SECTIONS  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────

def demo_file_to_frame_name(demo_file):
    """Convert demo filename to frame module name.

    livePlaneSweep.py  -> plane_sweep
    liveCalibration.py -> calibration
    liveCamera.py      -> camera
    liveSIFTMatching.py -> sift_matching
    """
    name = demo_file.replace(".py", "")
    if name.startswith("live"):
        name = name[4:]
    # CamelCase/acronyms -> snake_case
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def discover_frames():
    """Return set of frame module names that exist in web/frames/."""
    if not os.path.isdir(FRAMES_DIR):
        return set()
    return {
        f[:-3]
        for f in os.listdir(FRAMES_DIR)
        if f.endswith(".py") and f != "__init__.py"
    }


# ── Step 1: Copy vendor files ──────────────────────────────────────────

def copy_vendor():
    """Copy demo .py files and utils/ into web/vendor/."""
    # Clean and recreate
    if os.path.exists(VENDOR_DIR):
        shutil.rmtree(VENDOR_DIR)
    os.makedirs(os.path.join(VENDOR_DIR, "utils"), exist_ok=True)

    # Copy all utils
    utils_src = os.path.join(DEMOS_DIR, "utils")
    for f in os.listdir(utils_src):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(utils_src, f),
                         os.path.join(VENDOR_DIR, "utils", f))

    # Copy demo .py files for available frames
    available = discover_frames()
    for _heading, _color, demos in SECTIONS:
        for _name, info in demos.items():
            frame_name = demo_file_to_frame_name(info["file"])
            if frame_name in available:
                src = os.path.join(DEMOS_DIR, info["file"])
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(VENDOR_DIR, info["file"]))

    print(f"  Copied vendor files to {VENDOR_DIR}")


# ── Step 1b: Copy data files ─────────────────────────────────────────

def copy_data():
    """Copy all files from demos/data/ into web/data/."""
    src_dir = os.path.join(DEMOS_DIR, "data")
    if not os.path.isdir(src_dir):
        print(f"  Warning: {src_dir} not found, skipping data copy")
        return

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

    count = 0
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(DATA_DIR, f))
            count += 1

    print(f"  Copied {count} data file(s) to {DATA_DIR}")


# ── Step 2: Generate pyscript.toml ─────────────────────────────────────

def generate_toml():
    """Generate pyscript.toml with package list and file mappings."""
    # PyScript [files] format: "source_url" = "vfs_destination"
    # Left  = URL to fetch from web server (relative to the HTML page)
    # Right = path in Pyodide virtual filesystem (for Python imports)
    lines = [
        'packages = ["numpy", "opencv-python"]',
        "",
        "[files]",
        "# --- Frame modules ---",
        '"frames/__init__.py" = "./frames/__init__.py"',
    ]

    available = discover_frames()
    for name in sorted(available):
        lines.append(f'"frames/{name}.py" = "./frames/{name}.py"')

    lines.append("")
    lines.append("# --- Vendor: utils (all needed, __init__.py re-exports) ---")

    utils_dir = os.path.join(VENDOR_DIR, "utils")
    if os.path.isdir(utils_dir):
        for f in sorted(os.listdir(utils_dir)):
            if f.endswith(".py"):
                lines.append(f'"vendor/utils/{f}" = "./utils/{f}"')

    lines.append("")
    lines.append("# --- Data files (fallback images, etc.) ---")
    if os.path.isdir(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if os.path.isfile(os.path.join(DATA_DIR, f)):
                lines.append(f'"data/{f}" = "./data/{f}"')

    lines.append("")
    lines.append("# --- Vendor: demo source files ---")

    for f in sorted(os.listdir(VENDOR_DIR)):
        if f.endswith(".py"):
            lines.append(f'"vendor/{f}" = "./{f}"')

    toml_path = os.path.join(SCRIPT_DIR, "pyscript.toml")
    with open(toml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Generated {toml_path}")


# ── Step 3: Generate index.html ────────────────────────────────────────

INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSCI 1430 Interactive Demos</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="index-container">
        <h1>CSCI 1430 Interactive Demos</h1>
        <p class="demo-description">
            Computer Vision &mdash; Brown University.
            These demos run entirely in your browser via PyScript (Python + NumPy + OpenCV in WebAssembly).
            First load takes 10&ndash;30s to download the WASM runtime; subsequent visits use the browser cache.
        </p>
{sections}
    </div>
</body>
</html>
"""


def generate_index():
    """Generate index.html with demo cards grouped by section."""
    available = discover_frames()
    sections_html = []

    for heading, color, demos in SECTIONS:
        r, g, b = color
        sections_html.append(
            f'        <h3 class="section-heading" '
            f'style="color: rgb({r},{g},{b})">{escape(heading)}</h3>'
        )
        sections_html.append('        <div class="demo-cards">')

        for name, info in demos.items():
            frame_name = demo_file_to_frame_name(info["file"])
            is_available = frame_name in available
            cls = "demo-card" if is_available else "demo-card unavailable"
            badge_cls = "badge badge-available" if is_available else "badge badge-soon"
            badge_text = "interactive" if is_available else "coming soon"

            if is_available:
                link = f'<a href="demo.html?demo={frame_name}">{escape(name)}</a>'
            else:
                link = f'<span>{escape(name)}</span>'

            desc = escape(info.get("description", ""))

            sections_html.append(f'            <div class="{cls}">')
            sections_html.append(
                f'                {link}'
                f'<span class="{badge_cls}">{badge_text}</span>'
            )
            sections_html.append(f'                <div class="desc">{desc}</div>')
            sections_html.append(f'            </div>')

        sections_html.append('        </div>')

    html = INDEX_TEMPLATE.format(sections="\n".join(sections_html))

    index_path = os.path.join(SCRIPT_DIR, "index.html")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"  Generated {index_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    available = discover_frames()
    print(f"Found {len(available)} frame module(s): {sorted(available)}")

    # Map to demo files
    for heading, _color, demos in SECTIONS:
        for name, info in demos.items():
            frame_name = demo_file_to_frame_name(info["file"])
            status = "OK" if frame_name in available else "  "
            print(f"  [{status}] {name:30s} -> {frame_name}")

    print()
    copy_vendor()
    copy_data()
    generate_toml()
    generate_index()
    print("\nDone. Serve with:  python -m http.server 8000")
    print("Open:  http://localhost:8000/web/demo.html?demo=plane_sweep")


if __name__ == "__main__":
    main()
