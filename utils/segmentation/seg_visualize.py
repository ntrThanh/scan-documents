import cv2
import numpy as np
from ..visualize import show

def colorize_label_mask(mask):
    """
    Chuyển mask label thành ảnh màu.
    """

    mask = mask.astype(np.int32)

    if mask.max() == mask.min():
        norm_mask = np.zeros_like(mask, dtype=np.uint8)
    else:
        norm_mask = cv2.normalize(
            mask,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    colored = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    colored[mask == 0] = [0, 0, 0]

    return colored


def overlay_mask_on_image(display_img, mask, alpha=0.45):
    """
    Overlay mask màu lên ảnh hiển thị RGB.
    """

    colored_mask = colorize_label_mask(mask)

    overlay = display_img.copy().astype(np.float32)
    colored_mask = colored_mask.astype(np.float32)

    mask_area = mask > 0

    overlay[mask_area] = (
        (1 - alpha) * overlay[mask_area]
        + alpha * colored_mask[mask_area]
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay

def show_segmentation_comparison(result, title_prefix=""):
    """
    Hiển thị từng thuật toán:
    Ảnh gốc | Mask | Overlay
    """

    original = result["original"]
    masks = result["masks"]
    overlays = result["overlays"]

    for method_name in masks.keys():
        mask = masks[method_name]
        overlay = overlays[method_name]
        mask_color = colorize_label_mask(mask)

        show(
            [original, mask_color, overlay],
            [
                f"{title_prefix}Ảnh gốc",
                f"{method_name} - Mask",
                f"{method_name} - Overlay"
            ],
            cols=3,
            figsize=(15, 5)
        )

def show_watershed_experiment(result, title_prefix=""):
    """
    Hiển thị các bước trung gian của Watershed.
    """

    params = result["params"]

    original = result["original"]
    intensity = result["intensity"]
    blur = result["blur"]
    binary = result["binary"]
    opening = result["opening"]
    sure_bg = result["sure_bg"]
    dist_transform = result["dist_transform"]
    sure_fg = result["sure_fg"]
    unknown = result["unknown"]
    watershed_mask = result["watershed_mask"]
    watershed_overlay = result["watershed_overlay"]

    watershed_mask_color = colorize_label_mask(watershed_mask)

    title = (
        f"{title_prefix}"
        f"blur={params['blur_ksize']}, "
        f"open_k={params['open_kernel_size']}, "
        f"open_it={params['open_iterations']}, "
        f"dilate_it={params['dilate_iterations']}, "
        f"fg={params['fg_threshold_ratio']}"
    )

    show(
        [
            original,
            # intensity,
            # blur,
            # binary,
            # opening,
            sure_bg,
            dist_transform,
            sure_fg,
            # unknown,
            watershed_mask_color,
            watershed_overlay
        ],
        [
            f"{title}\nẢnh gốc",
            # "Intensity",
            # "Blur",
            # "Binary",
            # "Opening",
            "Sure BG",
            "Distance Transform",
            "Sure FG",
            # "Unknown",
            "Watershed Mask",
            "Watershed Overlay"
        ],
        cols=6,
        figsize=(20, 14)
    )

def show_all_masks_and_overlays_in_one(result, title_prefix=""):
    """
    Hiển thị:
    - Hàng 1: ảnh gốc + intensity + các mask
    - Hàng 2: ảnh gốc + các overlay
    """

    original = result["original"]
    intensity = result["intensity"]
    masks = result["masks"]
    overlays = result["overlays"]

    mask_images = [original, intensity]
    mask_titles = [
        f"{title_prefix}Ảnh gốc",
        f"{title_prefix}Kênh xử lý"
    ]

    for method_name, mask in masks.items():
        mask_images.append(colorize_label_mask(mask))
        mask_titles.append(f"{method_name}\nMask")

    show(
        mask_images,
        mask_titles,
        cols=len(mask_images),
        figsize=(4 * len(mask_images), 4)
    )

    overlay_images = [original]
    overlay_titles = [f"{title_prefix}Ảnh gốc"]

    for method_name, overlay in overlays.items():
        overlay_images.append(overlay)
        overlay_titles.append(f"{method_name}\nOverlay")

    show(
        overlay_images,
        overlay_titles,
        cols=len(overlay_images),
        figsize=(4 * len(overlay_images), 4)
    )