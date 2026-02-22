"""
Web frame module for livePlaneSweep demo.
Imports computation from the original demo (via vendor/), defines WEB_CONFIG
and web_frame() for the generic adapter.
"""

from livePlaneSweep import (
    REF_IMG, OTHER_IMG,
    M_REF, M_OTHER,
    K_SHARED, _Rt_ref, _Rt_other,
    IMG_W, IMG_H, OVERVIEW_SIZE,
    LAM_MIN, LAM_MAX, LAM_TRUE,
    compute_H_lam, warp_other_to_ref, compute_ncc_score,
    draw_ncc_curve,
    _OV_K, _OV_Rt,
)
from utils.demo_3d import render_scene, make_frustum_mesh, make_axis_mesh


WEB_CONFIG = {
    "title": "Plane Sweep Stereo",
    "description": (
        "Interactive H(\u03bb) plane sweep demo. "
        "Sweep the depth slider to warp one camera view into the other. "
        "NCC peaks when \u03bb matches the true scene depth (\u03bb = 3.0)."
    ),
    "outputs": [
        {"id": "ref",       "label": "Reference image (fixed)",  "width": 400, "height": 400},
        {"id": "warped",    "label": "Warped other [H(\u03bb)]", "width": 400, "height": 400},
        {"id": "ncc_curve", "label": "NCC vs \u03bb",            "width": 400, "height": 300},
        {"id": "overview",  "label": "3D Overview",               "width": 400, "height": 400},
    ],
    "controls": {
        "lam": {
            "type": "float", "min": LAM_MIN, "max": LAM_MAX, "step": 0.05,
            "default": 3.0, "label": "Depth \u03bb",
        },
    },
    "layout": {
        "rows": [
            ["ref", "warped", "ncc_curve"],
            ["overview"],
        ],
    },
}


def web_frame(state):
    """Compute one frame. Returns dict of output_id -> BGR ndarray or str."""
    lam = state["lam"]

    # Core plane-sweep computation (reuses original demo functions)
    H = compute_H_lam(M_REF, M_OTHER, lam)
    warped = warp_other_to_ref(OTHER_IMG, H, IMG_W, IMG_H)
    ncc = compute_ncc_score(REF_IMG, warped)

    # 3D overview: two camera frustums + world axes
    frustum_ref = make_frustum_mesh(K_SHARED, _Rt_ref, IMG_W, IMG_H,
                                    near=0.3, far=6.0)
    frustum_other = make_frustum_mesh(K_SHARED, _Rt_other, IMG_W, IMG_H,
                                      near=0.3, far=6.0)
    axes = make_axis_mesh(origin=(0, 0, 0), length=1.0)
    ov_img = render_scene(frustum_ref + frustum_other + axes,
                          _OV_K, _OV_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE)

    # NCC curve with current position highlighted
    curve = draw_ncc_curve(lam, ncc)

    # Status text
    ncc_str = f"{ncc:.4f}" if ncc >= -0.5 else "invalid"
    peak_hint = "  >>> PEAK NEAR HERE <<<" if abs(lam - LAM_TRUE) < 0.3 else ""
    status = (
        f"\u03bb = {lam:.3f}  |  NCC = {ncc_str}  |  "
        f"True \u03bb = {LAM_TRUE:.1f}{peak_hint}"
    )

    return {
        "ref": REF_IMG,
        "warped": warped,
        "ncc_curve": curve,
        "overview": ov_img,
        "status": status,
    }
