import cv2
import numpy as np
from .core import get_display_and_intensity

def watershed_segmentation(
    img,
    mode="rgb",
    blur_ksize=5,
    open_kernel_size=3,
    open_iterations=2,
    dilate_iterations=3,
    fg_threshold_ratio=0.4
):
    """
    Watershed Segmentation với tham số tùy chỉnh.

    Parameters
    ----------
    blur_ksize : int
        Kích thước kernel Gaussian blur
    open_kernel_size : int
        Kích thước kernel cho opening
    open_iterations : int
        Số lần opening
    dilate_iterations : int
        Số lần dilate để tạo sure background
    fg_threshold_ratio : float
        Ngưỡng lấy sure foreground theo distance transform
    """

    display_img, intensity = get_display_and_intensity(img, mode)

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize <= 1:
        blur_ksize = 3

    blur = cv2.GaussianBlur(intensity, (blur_ksize, blur_ksize), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations
    )

    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=dilate_iterations
    )

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    _, sure_fg = cv2.threshold(
        dist_transform,
        fg_threshold_ratio * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    markers = cv2.watershed(bgr, markers)

    watershed_mask = markers.copy()
    watershed_mask[watershed_mask == -1] = 0

    return {
        "display_img": display_img,
        "intensity": intensity,
        "blur": blur,
        "binary": binary,
        "opening": opening,
        "sure_bg": sure_bg,
        "dist_transform": dist_transform,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "markers": markers,
        "watershed_mask": watershed_mask.astype(np.int32)
    }