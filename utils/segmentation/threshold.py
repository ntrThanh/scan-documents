import cv2
from .core import get_display_and_intensity

def otsu_threshold_segmentation(img, mode="rgb"):
    """
    Otsu Thresholding.

    RGB  -> dùng gray từ RGB
    Gray -> dùng trực tiếp gray
    HSV  -> dùng kênh V
    LAB  -> dùng kênh L
    """

    display_img, intensity = get_display_and_intensity(img, mode)

    blur = cv2.GaussianBlur(intensity, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask

def adaptive_threshold_segmentation(img, mode="rgb", block_size=35, C=5):
    """
    Adaptive Thresholding.

    RGB  -> dùng gray từ RGB
    Gray -> dùng trực tiếp gray
    HSV  -> dùng kênh V
    LAB  -> dùng kênh L
    """

    display_img, intensity = get_display_and_intensity(img, mode)

    blur = cv2.GaussianBlur(intensity, (5, 5), 0)

    if block_size % 2 == 0:
        block_size += 1

    if block_size <= 1:
        block_size = 3

    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return mask
