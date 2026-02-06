"""
Utility modules for CSCI 1430 computer vision demos.
"""

from .demo_utils import (
    DATA_DIR,
    convert_cv_to_dpg,
    convert_cv_to_dpg_float,
    resize_with_letterbox,
    init_camera,
    load_fallback_image,
    crop_to_square,
    apply_affine_transform,
    apply_brightness,
)

from .demo_ui import (
    setup_viewport,
    make_state_updater,
    make_reset_callback,
    add_global_controls,
)

from .demo_kernels import (
    KERNEL_PRESETS,
    SIGMA_KERNELS,
    ZERO_DC_KERNELS,
    create_kernel,
    resize_kernel,
    pad_kernel_to_image_size,
    create_gaussian_kernel_fft,
    visualize_kernel,
)

from .demo_fft import (
    visualize_fft_amplitude,
    process_convolution,
    process_deconvolution,
)
