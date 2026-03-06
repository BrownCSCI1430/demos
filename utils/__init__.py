"""
Utility modules for CSCI 1430 computer vision demos.
"""

from .demo_utils import (
    convert_cv_to_dpg,
    convert_cv_to_dpg_float,
    crop_to_square,
    apply_affine_transform,
    apply_brightness,
)

from .demo_webcam import (
    DATA_DIR,
    probe_cameras,
    init_camera,
    load_fallback_image,
    resize_with_letterbox,
    get_frame,
    init_camera_demo,
    switch_camera,
    cleanup_camera_demo,
)

from .demo_ui import (
    setup_viewport,
    make_state_updater,
    make_reset_callback,
    make_reset_all_callback,
    make_camera_callback,
    add_global_controls,
    add_guide_button,
    create_guide_window,
    control_panel,
    poll_collapsible_panels,
    get_image_pixel_coords,
    auto_resize_images,
    create_blank_texture,
)

from .demo_kernels import (
    KERNEL_PRESETS,
    SIGMA_KERNELS,
    ZERO_DC_KERNELS,
    make_kernel,
    resize_kernel,
    pad_kernel_to_image_size,
    make_gaussian_kernel_fft,
    visualize_kernel,
)

from .demo_fft import (
    visualize_fft_amplitude,
    process_convolution,
    process_deconvolution,
)

from .demo_3d import (
    build_intrinsic,
    build_rotation,
    build_extrinsic,
    euler_from_rotation,
    fov_to_focal,
    make_lookat_Rt,
    render_scene,
    make_default_scene,
    make_textured_scene,
    make_cube,
    make_sphere,
    make_cylinder,
    make_ground_grid,
    make_frustum_mesh,
    make_axis_mesh,
    make_camera_axes_mesh,
    format_matrix,
    compute_F_from_cameras,
    epipolar_line_endpoints,
    decompose_H,
    compute_H_lam,
    flip_y_matrix,
    compute_stereo_cameras,
    OrbitCamera,
)
