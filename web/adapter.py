"""
Generic PyScript adapter for CSCI 1430 DearPyGui demos.
Runs in the browser via Pyodide. Reads WEB_CONFIG from a frame module,
auto-builds HTML controls + canvases, and wires events to computation.

Supports two modes based on WEB_CONFIG:
  1. Slider-only (e.g. plane sweep): event-driven, _update() on slider input
  2. Camera (e.g. Canny edges): continuous async loop grabbing webcam frames

Generic control types: float, int, bool, choice, button
Generic features: visible_when, set_controls return, canvas mouse events,
                  pause, reset buttons, multi-line status

Usage: loaded by demo.html via <script type="py" src="adapter.py" config="pyscript.toml">
Demo selected via URL query param: demo.html?demo=plane_sweep
"""

# ── Step 1: Mock DearPyGui before any demo imports ──────────────────────
import sys
from unittest.mock import MagicMock

_mock = MagicMock()
sys.modules["dearpygui"] = _mock
sys.modules["dearpygui.dearpygui"] = _mock

# ── Step 2: Standard imports ────────────────────────────────────────────
import importlib
import time
import numpy as np
import cv2
from pyodide.ffi import create_proxy, to_js
from js import document, window, Uint8ClampedArray, ImageData

# ── Step 3: Determine which demo to load from URL ──────────────────────
_params = window.URLSearchParams.new(window.location.search)
_demo_name = _params.get("demo") or "plane_sweep"

# ── Step 4: Import the frame module ─────────────────────────────────────
_frame = importlib.import_module(f"frames.{_demo_name}")
WEB_CONFIG = _frame.WEB_CONFIG
web_frame = _frame.web_frame

# Optional frame module exports
_has_web_mouse = hasattr(_frame, "web_mouse")
_has_web_button = hasattr(_frame, "web_button")

# Build lookups
_output_map = {o["id"]: o for o in WEB_CONFIG["outputs"]}
_has_camera = "camera" in WEB_CONFIG
_controls = WEB_CONFIG.get("controls", {})
_mouse_canvases = WEB_CONFIG.get("mouse", [])

# Load fallback image for camera demos (scan data/ for any image)
_fallback_image = None
_camera_active = False   # True only when real camera loop is running
_fallback_frame = None   # Set when camera fails, used by _do_update()
if _has_camera:
    import os as _os
    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    _data_dir = "data"
    if _os.path.isdir(_data_dir):
        for _f in sorted(_os.listdir(_data_dir)):
            if _os.path.splitext(_f)[1].lower() in _IMAGE_EXTS:
                _raw = cv2.imread(_os.path.join(_data_dir, _f))
                if _raw is not None:
                    cam_cfg = WEB_CONFIG["camera"]
                    _fallback_image = cv2.resize(
                        _raw, (cam_cfg.get("width", 640),
                               cam_cfg.get("height", 480)))
                    print(f"Fallback image loaded: {_f}")
                    break


# ── Step 5: Canvas rendering helper ─────────────────────────────────────

def _push_to_canvas(canvas_id, img):
    """Push a NumPy BGR or grayscale uint8 image to an HTML canvas element."""
    if img is None:
        return
    if len(img.shape) == 2:
        rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    else:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    flat = np.ascontiguousarray(rgba, dtype=np.uint8)

    canvas = document.getElementById(f"canvas-{canvas_id}")
    if canvas is None:
        return
    canvas.width = w
    canvas.height = h
    ctx = canvas.getContext("2d")

    buf = flat.tobytes()
    js_buf = to_js(buf)
    clamped = Uint8ClampedArray.new(js_buf)
    img_data = ImageData.new(clamped, w, h)
    ctx.putImageData(img_data, 0, 0)


def _push_results(results):
    """Push all outputs from web_frame results to canvases + status + set_controls."""
    for out in WEB_CONFIG["outputs"]:
        oid = out["id"]
        if oid in results and isinstance(results[oid], np.ndarray):
            _push_to_canvas(oid, results[oid])

    if "status" in results:
        el = document.getElementById("status-text")
        if el is not None:
            el.textContent = results["status"]

    # Handle set_controls: web_frame() can request control value updates
    if "set_controls" in results:
        for ctrl_id, new_val in results["set_controls"].items():
            _set_control_value(ctrl_id, new_val)


def _set_control_value(ctrl_id, new_val):
    """Programmatically update a control's HTML element and value display."""
    el = document.getElementById(f"ctrl-{ctrl_id}")
    if el is None:
        return
    ctrl = _controls.get(ctrl_id, {})
    ctype = ctrl.get("type", "float")

    if ctype == "bool":
        el.checked = bool(new_val)
    elif ctype == "choice":
        el.value = str(new_val)
    else:
        el.value = str(new_val)
        # Update value display span
        val_el = document.getElementById(f"val-{ctrl_id}")
        if val_el is not None:
            fmt = ctrl.get("format", ".2f" if ctype == "float" else "d")
            if ctype == "int":
                val_el.textContent = f"{int(float(new_val)):{fmt}}"
            else:
                val_el.textContent = f"{float(new_val):{fmt}}"


# ── Step 6: Build HTML UI from WEB_CONFIG ───────────────────────────────

def _build_ui():
    container = document.getElementById("demo-container")

    # Title
    h2 = document.createElement("h2")
    h2.textContent = WEB_CONFIG["title"]
    container.appendChild(h2)

    # Description
    p = document.createElement("p")
    p.className = "demo-description"
    p.textContent = WEB_CONFIG["description"]
    container.appendChild(p)

    # Controls
    controls_div = document.createElement("div")
    controls_div.className = "controls"

    # Cat mode toggle (built-in for all camera demos)
    if _has_camera and _fallback_image is not None:
        row = document.createElement("div")
        row.className = "control-row"

        label = document.createElement("label")
        label.htmlFor = "ctrl-cat-mode"
        label.textContent = "Cat Mode"
        row.appendChild(label)

        inp = document.createElement("input")
        inp.type = "checkbox"
        inp.id = "ctrl-cat-mode"
        inp.checked = False
        row.appendChild(inp)

        controls_div.appendChild(row)

    for ctrl_id, ctrl in _controls.items():
        row = document.createElement("div")
        row.className = "control-row"
        row.id = f"ctrl-row-{ctrl_id}"

        ctype = ctrl.get("type", "float")

        if ctype == "button":
            # Button: no label/value, just a button element
            btn = document.createElement("button")
            btn.id = f"ctrl-{ctrl_id}"
            btn.textContent = ctrl["label"]
            btn.className = "ctrl-button"
            row.appendChild(btn)

        elif ctype == "choice":
            label = document.createElement("label")
            label.htmlFor = f"ctrl-{ctrl_id}"
            label.textContent = ctrl["label"]
            row.appendChild(label)

            sel = document.createElement("select")
            sel.id = f"ctrl-{ctrl_id}"
            sel.className = "ctrl-select"
            for opt_val in ctrl["options"]:
                opt = document.createElement("option")
                opt.value = opt_val
                opt.textContent = opt_val
                if opt_val == ctrl.get("default"):
                    opt.selected = True
                sel.appendChild(opt)
            row.appendChild(sel)

        elif ctype == "bool":
            label = document.createElement("label")
            label.htmlFor = f"ctrl-{ctrl_id}"
            label.textContent = ctrl["label"]
            row.appendChild(label)

            inp = document.createElement("input")
            inp.type = "checkbox"
            inp.id = f"ctrl-{ctrl_id}"
            inp.checked = ctrl.get("default", False)
            row.appendChild(inp)

        else:
            # float or int slider
            label = document.createElement("label")
            label.htmlFor = f"ctrl-{ctrl_id}"
            label.textContent = ctrl["label"]
            row.appendChild(label)

            inp = document.createElement("input")
            inp.type = "range"
            inp.id = f"ctrl-{ctrl_id}"
            inp.min = str(ctrl["min"])
            inp.max = str(ctrl["max"])
            inp.step = str(ctrl.get("step", 0.01 if ctype == "float" else 1))
            inp.value = str(ctrl["default"])
            row.appendChild(inp)

            val_span = document.createElement("span")
            val_span.id = f"val-{ctrl_id}"
            val_span.className = "slider-value"
            fmt = ctrl.get("format", ".2f" if ctype == "float" else "d")
            if ctype == "int":
                val_span.textContent = f"{int(ctrl['default']):{fmt}}"
            else:
                val_span.textContent = f"{ctrl['default']:{fmt}}"
            row.appendChild(val_span)

            # Reset "R" button
            rbtn = document.createElement("button")
            rbtn.className = "reset-btn"
            rbtn.textContent = "R"
            rbtn.dataset.ctrl = ctrl_id
            row.appendChild(rbtn)

        controls_div.appendChild(row)

    container.appendChild(controls_div)

    # Status text (pre for multi-line support)
    status = document.createElement("pre")
    status.id = "status-text"
    status.className = "status"
    container.appendChild(status)

    # Canvas grid
    grid = document.createElement("div")
    grid.className = "canvas-grid"

    for row_ids in WEB_CONFIG["layout"]["rows"]:
        row_div = document.createElement("div")
        row_div.className = "canvas-row"

        for out_id in row_ids:
            out = _output_map[out_id]
            cell = document.createElement("div")
            cell.className = "canvas-cell"

            lbl = document.createElement("div")
            lbl.className = "canvas-label"
            lbl.textContent = out["label"]
            cell.appendChild(lbl)

            canvas = document.createElement("canvas")
            canvas.id = f"canvas-{out_id}"
            canvas.width = out["width"]
            canvas.height = out["height"]
            cell.appendChild(canvas)

            row_div.appendChild(cell)

        grid.appendChild(row_div)

    container.appendChild(grid)


# ── Step 7: Read state from controls ────────────────────────────────────

def _read_state():
    """Read all control values into a dict."""
    state = {}
    for ctrl_id, ctrl in _controls.items():
        ctype = ctrl.get("type", "float")

        if ctype == "button":
            continue  # Buttons don't have state

        el = document.getElementById(f"ctrl-{ctrl_id}")
        if el is None:
            state[ctrl_id] = ctrl.get("default")
            continue

        if ctype == "bool":
            state[ctrl_id] = bool(el.checked)
        elif ctype == "choice":
            state[ctrl_id] = el.value
        elif ctype == "int":
            state[ctrl_id] = int(float(el.value))
        else:
            state[ctrl_id] = float(el.value)

        # Update value display for sliders
        if ctype in ("float", "int"):
            val_el = document.getElementById(f"val-{ctrl_id}")
            if val_el is not None:
                fmt = ctrl.get("format", ".2f" if ctype == "float" else "d")
                val_el.textContent = f"{state[ctrl_id]:{fmt}}"

    return state


# ── Step 8: Conditional visibility ──────────────────────────────────────

def _update_visibility():
    """Show/hide controls based on visible_when conditions."""
    state = {}
    # Quick read of all control values for visibility checks
    for ctrl_id, ctrl in _controls.items():
        ctype = ctrl.get("type", "float")
        if ctype == "button":
            continue
        el = document.getElementById(f"ctrl-{ctrl_id}")
        if el is None:
            continue
        if ctype == "bool":
            state[ctrl_id] = bool(el.checked)
        elif ctype == "choice":
            state[ctrl_id] = el.value
        elif ctype == "int":
            state[ctrl_id] = int(float(el.value))
        else:
            state[ctrl_id] = float(el.value)

    for ctrl_id, ctrl in _controls.items():
        vw = ctrl.get("visible_when")
        if vw is None:
            continue
        row = document.getElementById(f"ctrl-row-{ctrl_id}")
        if row is None:
            continue
        visible = True
        for dep_id, allowed_values in vw.items():
            if state.get(dep_id) not in allowed_values:
                visible = False
                break
        row.style.display = "" if visible else "none"


# ── Step 9: Slider-only mode ────────────────────────────────────────────

_raf_pending = False


def _update(*args):
    """Called on control input: debounce via requestAnimationFrame."""
    global _raf_pending
    _update_visibility()
    if _raf_pending:
        return
    _raf_pending = True
    window.requestAnimationFrame(create_proxy(_do_update))


def _do_update(*args):
    """Actual computation + render, called from requestAnimationFrame.
    Skipped when the real camera loop is active (it handles rendering).
    In camera-fallback mode (no camera), injects the fallback image."""
    global _raf_pending
    _raf_pending = False

    if _camera_active:
        return  # Real camera loop handles rendering

    state = _read_state()

    # Camera demo in fallback mode: inject input_image
    if _has_camera and _fallback_frame is not None:
        state["input_image"] = _fallback_frame.copy()

    results = web_frame(state)

    # Append mode label for camera fallback
    if _has_camera:
        label = "fallback" if _fallback_image is not None else "test pattern"
        status = results.get("status", "")
        results["status"] = f"{status}  |  [{label}]" if status else f"[{label}]"

    _push_results(results)


# ── Step 10: Button handlers ────────────────────────────────────────────

def _on_button_click(ctrl_id):
    """Handle button click: call frame module's web_button, then re-render."""
    if _has_web_button:
        _frame.web_button(ctrl_id)
    _update_visibility()
    if not _camera_active:
        _do_update()
    # Active camera: next frame loop iteration picks up the state change


def _on_reset_click(ctrl_id):
    """Handle reset button click: reset slider to default, then re-render."""
    ctrl = _controls.get(ctrl_id)
    if ctrl is not None:
        _set_control_value(ctrl_id, ctrl["default"])
    _update_visibility()
    if not _camera_active:
        _update()


# ── Step 11: Canvas mouse events ────────────────────────────────────────

def _make_mouse_handler(canvas_id, event_type):
    """Create a mouse event handler for a specific canvas."""
    def handler(event):
        canvas = document.getElementById(f"canvas-{canvas_id}")
        if canvas is None:
            return
        rect = canvas.getBoundingClientRect()
        x = event.clientX - rect.left
        y = event.clientY - rect.top
        evt = {
            "type": event_type,
            "canvas": canvas_id,
            "x": x,
            "y": y,
            "button": event.button,
            "shift": event.shiftKey,
            "ctrl": event.ctrlKey,
        }
        if event_type == "wheel":
            evt["delta_y"] = event.deltaY
            event.preventDefault()
        if event_type == "contextmenu":
            event.preventDefault()
        if _has_web_mouse:
            _frame.web_mouse(evt)
            if not _camera_active:
                _do_update()
    return handler


# ── Step 12: Camera support ─────────────────────────────────────────────

def _grab_frame(video, hcanvas, w, h):
    """Grab one frame from the video element as a BGR numpy array."""
    ctx = hcanvas.getContext("2d")
    ctx.drawImage(video, 0, 0, w, h)
    img_data = ctx.getImageData(0, 0, w, h)
    buf = img_data.data.to_py()
    rgba = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return bgr


def _make_test_pattern(w, h):
    """Generate a colorful test pattern for camera-unavailable fallback."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        t = x / max(w - 1, 1)
        img[:, x, 0] = int(255 * (1 - t))
        img[:, x, 2] = int(255 * t)
    for y in range(h):
        t = y / max(h - 1, 1)
        img[y, :, 1] = int(100 * t)
    for i in range(5):
        cx = int(w * (i + 1) / 6)
        cy = h // 2
        r = min(w, h) // 8
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
    cv2.putText(img, "No Camera", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


async def _camera_main():
    """Async entry point for camera-based demos."""
    global _camera_active, _fallback_frame
    from js import navigator, JSON

    cam_cfg = WEB_CONFIG["camera"]
    req_w = cam_cfg.get("width", 640)
    req_h = cam_cfg.get("height", 480)

    video = document.createElement("video")
    video.setAttribute("playsinline", "")
    video.setAttribute("autoplay", "")
    video.style.display = "none"
    document.body.appendChild(video)

    hcanvas = document.createElement("canvas")
    hcanvas.style.display = "none"
    document.body.appendChild(hcanvas)

    cam_ok = False
    cam_w, cam_h = req_w, req_h

    try:
        constraints = JSON.parse(
            f'{{"video": {{"width": {{"ideal": {req_w}}}, '
            f'"height": {{"ideal": {req_h}}}}}}}'
        )
        stream = await navigator.mediaDevices.getUserMedia(constraints)
        video.srcObject = stream
        await video.play()

        import asyncio
        for _ in range(50):
            if video.videoWidth > 0:
                break
            await asyncio.sleep(0.05)

        cam_w = video.videoWidth or req_w
        cam_h = video.videoHeight or req_h
        hcanvas.width = cam_w
        hcanvas.height = cam_h
        cam_ok = True
        print(f"Camera ready: {cam_w}x{cam_h}")
    except Exception as e:
        print(f"Camera unavailable: {e}")

    # Show UI
    document.getElementById("loading").style.display = "none"
    document.getElementById("demo-container").style.display = "block"
    _update_visibility()

    if cam_ok:
        import asyncio
        _camera_active = True
        frame_times = []
        last_frame = None

        while True:
            now = time.time()
            frame_times.append(now)
            while frame_times and frame_times[0] < now - 1.0:
                frame_times.pop(0)
            fps = len(frame_times)

            # Read controls
            state = _read_state()
            _update_visibility()

            # Cat mode
            cat_el = document.getElementById("ctrl-cat-mode")
            cat_mode = cat_el is not None and cat_el.checked

            # Pause support: reuse last frame if pause control is True
            paused = state.get("pause", False)

            if paused and last_frame is not None:
                frame = last_frame.copy()
            elif cat_mode and _fallback_image is not None:
                frame = _fallback_image.copy()
            else:
                frame = _grab_frame(video, hcanvas, cam_w, cam_h)
                if (cam_w, cam_h) != (req_w, req_h):
                    frame = cv2.resize(frame, (req_w, req_h))
                last_frame = frame.copy()

            state["input_image"] = frame
            results = web_frame(state)

            # Append FPS + mode to status
            status = results.get("status", "")
            mode = "paused" if paused else ("cat" if cat_mode else "cam")
            results["status"] = f"{status}  |  {fps} fps [{mode}]" if status else f"{fps} fps [{mode}]"

            _push_results(results)

            await asyncio.sleep(0)
    else:
        # No camera: use fallback image or test pattern, slider-driven.
        # _camera_active stays False, so _do_update() will inject _fallback_frame
        # and all generic event handlers (sliders, buttons, reset, mouse) work.
        _fallback_frame = (_fallback_image if _fallback_image is not None
                           else _make_test_pattern(req_w, req_h))
        _do_update()  # Initial render


# ── Step 13: Wire everything up ─────────────────────────────────────────

_build_ui()

# Wire control event listeners
_update_proxy = create_proxy(_update)
for _ctrl_id, _ctrl in _controls.items():
    _ctype = _ctrl.get("type", "float")

    if _ctype == "button":
        _el = document.getElementById(f"ctrl-{_ctrl_id}")
        if _el is not None:
            _btn_id = _ctrl_id  # capture in closure
            _el.addEventListener("click", create_proxy(
                lambda e, bid=_btn_id: _on_button_click(bid)))
        continue

    _el = document.getElementById(f"ctrl-{_ctrl_id}")
    if _el is not None:
        _el.addEventListener("input", _update_proxy)
        _el.addEventListener("change", _update_proxy)

    # Wire reset buttons for sliders
    if _ctype in ("float", "int"):
        _rbtn = document.querySelector(f'button.reset-btn[data-ctrl="{_ctrl_id}"]')
        if _rbtn is not None:
            _rid = _ctrl_id  # capture
            _rbtn.addEventListener("click", create_proxy(
                lambda e, rid=_rid: _on_reset_click(rid)))

# Wire canvas mouse events
for _canvas_id in _mouse_canvases:
    _cel = document.getElementById(f"canvas-{_canvas_id}")
    if _cel is None:
        continue
    for _evt_type in ("click", "contextmenu", "mousemove", "mousedown", "mouseup", "wheel"):
        _handler = _make_mouse_handler(_canvas_id, _evt_type)
        _cel.addEventListener(_evt_type, create_proxy(_handler))

# Initial visibility
_update_visibility()

if _has_camera:
    import asyncio
    asyncio.ensure_future(_camera_main())
else:
    # Hide loading, show demo
    document.getElementById("loading").style.display = "none"
    document.getElementById("demo-container").style.display = "block"

    # Initial render
    _do_update()
