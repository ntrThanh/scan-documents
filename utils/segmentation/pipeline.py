from .watershed import watershed_segmentation
from .core import get_display_and_intensity
from .threshold import (
    adaptive_threshold_segmentation,
    otsu_threshold_segmentation
)
from .kmeans import (
    kmeans_pixel_segmentation,
    kmeans_spatial_segmentation
)
from .seg_visualize import (
    overlay_mask_on_image
)

def run_watershed_experiment(
    img,
    mode="rgb",
    blur_ksize=5,
    open_kernel_size=3,
    open_iterations=2,
    dilate_iterations=3,
    fg_threshold_ratio=0.4
):
    """
    Chạy riêng Watershed Segmentation để thực nghiệm với nhiều tham số khác nhau.
    """

    ws_result = watershed_segmentation(
        img,
        mode=mode,
        blur_ksize=blur_ksize,
        open_kernel_size=open_kernel_size,
        open_iterations=open_iterations,
        dilate_iterations=dilate_iterations,
        fg_threshold_ratio=fg_threshold_ratio
    )

    display_img = ws_result["display_img"]
    watershed_mask = ws_result["watershed_mask"]

    watershed_overlay = overlay_mask_on_image(
        display_img,
        watershed_mask,
        alpha=0.45
    )

    result = {
        "params": {
            "blur_ksize": blur_ksize,
            "open_kernel_size": open_kernel_size,
            "open_iterations": open_iterations,
            "dilate_iterations": dilate_iterations,
            "fg_threshold_ratio": fg_threshold_ratio
        },
        "original": display_img,
        "intensity": ws_result["intensity"],
        "blur": ws_result["blur"],
        "binary": ws_result["binary"],
        "opening": ws_result["opening"],
        "sure_bg": ws_result["sure_bg"],
        "dist_transform": ws_result["dist_transform"],
        "sure_fg": ws_result["sure_fg"],
        "unknown": ws_result["unknown"],
        "markers": ws_result["markers"],
        "watershed_mask": watershed_mask,
        "watershed_overlay": watershed_overlay
    }

    return result

def run_kmeans_threshold_segmentation_methods(img, mode="rgb", K=2):
    """
    Chạy các phương pháp phân đoạn cơ bản:
    - K-Means Pixel
    - K-Means có thêm tọa độ x,y
    - Otsu
    - Adaptive Threshold

    Không chạy Watershed trong hàm này.
    """

    display_img, intensity = get_display_and_intensity(img, mode)

    masks = {}

    masks["K-Means Pixel"] = kmeans_pixel_segmentation(
        img,
        mode=mode,
        K=K
    )

    masks["K-Means x,y"] = kmeans_spatial_segmentation(
        img,
        mode=mode,
        K=K,
        spatial_weight=0.6
    )

    masks["Otsu"] = otsu_threshold_segmentation(
        img,
        mode=mode
    )

    masks["Adaptive"] = adaptive_threshold_segmentation(
        img,
        mode=mode,
        block_size=35,
        C=5
    )

    overlays = {}

    for method_name, mask in masks.items():
        overlays[method_name] = overlay_mask_on_image(
            display_img,
            mask,
            alpha=0.45
        )

    result = {
        "original": display_img,
        "intensity": intensity,
        "masks": masks,
        "overlays": overlays
    }

    return result