import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show(images, titles=None, cmap=None, figsize=None):
    """
    Hiển thị một hoặc nhiều ảnh linh hoạt.

    Cách dùng:
        show(img)                          # 1 ảnh, không tiêu đề
        show(img, "Ảnh gốc")              # 1 ảnh, có tiêu đề
        show([img1, img2], ["A", "B"])    # nhiều ảnh
        show([img1, img2])                 # nhiều ảnh, không tiêu đề

    Tự động nhận diện ảnh màu (H,W,3) hay ảnh xám (H,W).
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
        titles = [titles] if titles is not None else [None]
    else:
        if titles is None:
            titles = [None] * len(images)
        elif not isinstance(titles, (list, tuple)):
            titles = [titles] * len(images)

    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        ax.axis("off")

        if img is None:
            ax.set_visible(False)
            continue

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img.squeeze(), cmap=cmap or "gray", vmin=0, vmax=255)
        elif img.ndim == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "Ảnh không hợp lệ", ha="center", va="center")

        if title:
            ax.set_title(title, fontsize=11, pad=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def read_gray(path):
    """
    Đọc ảnh và chuyển sang grayscale.

    Args:
        path (str | Path): đường dẫn tới file ảnh

    Returns:
        np.ndarray: ảnh grayscale shape (H, W), dtype uint8
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    print(f"[read_gray] {path.name} → shape={img.shape}, dtype={img.dtype}")
    return img


def read_rgb(path):
    """
    Đọc ảnh màu và trả về ở không gian BGR (chuẩn OpenCV).

    Args:
        path (str | Path): đường dẫn tới file ảnh

    Returns:
        np.ndarray: ảnh màu BGR shape (H, W, 3), dtype uint8
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    print(f"[read_rgb] {path.name} → shape={img.shape}, dtype={img.dtype}")
    return img


_COLOR_MAP = {
    ("bgr", "rgb"):   cv2.COLOR_BGR2RGB,
    ("bgr", "gray"):  cv2.COLOR_BGR2GRAY,
    ("bgr", "hsv"):   cv2.COLOR_BGR2HSV,
    ("bgr", "lab"):   cv2.COLOR_BGR2LAB,
    ("bgr", "hls"):   cv2.COLOR_BGR2HLS,
    ("bgr", "yuv"):   cv2.COLOR_BGR2YUV,
    ("bgr", "ycrcb"): cv2.COLOR_BGR2YCrCb,
    ("bgr", "xyz"):   cv2.COLOR_BGR2XYZ,
    ("rgb", "bgr"):   cv2.COLOR_RGB2BGR,
    ("rgb", "gray"):  cv2.COLOR_RGB2GRAY,
    ("rgb", "hsv"):   cv2.COLOR_RGB2HSV,
    ("rgb", "lab"):   cv2.COLOR_RGB2LAB,
    ("gray", "bgr"):  cv2.COLOR_GRAY2BGR,
    ("gray", "rgb"):  cv2.COLOR_GRAY2RGB,
    ("hsv", "bgr"):   cv2.COLOR_HSV2BGR,
    ("hsv", "rgb"):   cv2.COLOR_HSV2RGB,
    ("lab", "bgr"):   cv2.COLOR_LAB2BGR,
    ("lab", "rgb"):   cv2.COLOR_LAB2RGB,
    ("hls", "bgr"):   cv2.COLOR_HLS2BGR,
    ("yuv", "bgr"):   cv2.COLOR_YUV2BGR,
    ("ycrcb", "bgr"): cv2.COLOR_YCrCb2BGR,
    ("xyz", "bgr"):   cv2.COLOR_XYZ2BGR,
}


def convert(img, src, dst):
    """
    Chuyển đổi ảnh giữa các hệ màu.

    Args:
        img:  ảnh numpy array
        src:  hệ màu nguồn  — "bgr" | "rgb" | "gray" | "hsv" | "lab" | "hls" | "yuv" | "ycrcb" | "xyz"
        dst:  hệ màu đích   — như trên

    Returns:
        np.ndarray: ảnh đã chuyển đổi

    Ví dụ:
        gray = convert(img, "bgr", "gray")
        hsv  = convert(img, "bgr", "hsv")
        lab  = convert(img, "bgr", "lab")
    """
    src, dst = src.lower(), dst.lower()
    if src == dst:
        return img.copy()
    key = (src, dst)
    if key not in _COLOR_MAP:
        supported = sorted({k[0] for k in _COLOR_MAP} | {k[1] for k in _COLOR_MAP})
        raise ValueError(f"Không hỗ trợ chuyển '{src}' → '{dst}'. Các hệ màu hiện có: {supported}")
    result = cv2.cvtColor(img, _COLOR_MAP[key])
    print(f"[convert] {src.upper()} → {dst.upper()} | shape: {img.shape} → {result.shape}")
    return result


def blur_gaussian(img, ksize=5, sigma=0):
    """
    Làm mờ Gaussian — khử nhiễu nhẹ, giữ biên tương đối.

    Args:
        ksize (int): kích thước kernel, số lẻ (VD: 3, 5, 7)
        sigma (float): độ lệch chuẩn; 0 = tự tính từ ksize
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def blur_median(img, ksize=5):
    """
    Làm mờ Median — hiệu quả với nhiễu muối tiêu (salt & pepper).

    Args:
        ksize (int): kích thước kernel, số lẻ
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def blur_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """
    Làm mờ Bilateral — làm mịn nhưng bảo toàn biên.

    Args:
        d (int): đường kính vùng lân cận
        sigma_color: phạm vi màu sắc được làm mờ
        sigma_space: phạm vi không gian được làm mờ
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def blur_box(img, ksize=5):
    """
    Làm mờ Box (trung bình đơn giản).

    Args:
        ksize (int): kích thước kernel, số lẻ
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.boxFilter(img, -1, (ksize, ksize))


def sharpen(img, strength=1.0):
    """
    Làm sắc nét ảnh bằng unsharp masking.

    Args:
        strength (float): mức độ sắc nét, 0.5–2.0 là hợp lý
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def equalize_hist(img):
    """
    Cân bằng histogram — tăng tương phản toàn cục.
    Chỉ áp dụng cho ảnh xám.
    """
    if img.ndim != 2:
        raise ValueError("equalize_hist chỉ dùng cho ảnh xám (grayscale)")
    return cv2.equalizeHist(img)


def clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization.
    Tăng tương phản cục bộ, tránh khuếch đại nhiễu quá mức.
    Phù hợp cho tài liệu có bóng đổ hoặc ánh sáng không đều.

    Args:
        clip_limit (float): ngưỡng giới hạn tương phản, thường 2.0–4.0
        tile_grid (tuple): kích thước ô lưới
    """
    if img.ndim != 2:
        raise ValueError("clahe chỉ dùng cho ảnh xám")
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return engine.apply(img)


def denoise_nlm(img, h=10, template_window=7, search_window=21):
    """
    Non-Local Means Denoising — khử nhiễu mạnh, giữ chi tiết tốt.

    Args:
        h (int): cường độ lọc, cao hơn = mờ hơn
        template_window (int): kích thước cửa sổ so sánh patch
        search_window (int): kích thước vùng tìm kiếm
    """
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)


def threshold_otsu(img):
    """
    Phân ngưỡng Otsu — tự động tìm ngưỡng tối ưu.

    Returns:
        thresh_val (float): giá trị ngưỡng được chọn
        binary (np.ndarray): ảnh nhị phân
    """
    if img.ndim != 2:
        img = convert(img, "bgr", "gray")
    thresh_val, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"[threshold_otsu] Ngưỡng tự động = {thresh_val:.1f}")
    return thresh_val, binary


def threshold_adaptive(img, block_size=11, C=2, method="gaussian"):
    """
    Phân ngưỡng thích nghi — xử lý ánh sáng không đều tốt hơn Otsu.

    Args:
        block_size (int): kích thước vùng lân cận, số lẻ (VD: 11, 15, 21)
        C (int): hằng số trừ khỏi trung bình, thường 2–5
        method (str): "gaussian" | "mean"
    """
    if img.ndim != 2:
        img = convert(img, "bgr", "gray")
    method_code = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                   if method == "gaussian"
                   else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(img, 255, method_code, cv2.THRESH_BINARY, block_size, C)


def canny_edge(img, low=50, high=150, blur_ksize=5):
    """
    Phát hiện biên Canny.

    Args:
        low (int): ngưỡng thấp
        high (int): ngưỡng cao, thường = 2–3x low
        blur_ksize (int): kích thước kernel Gaussian làm mờ trước Canny
    """
    if img.ndim == 3:
        img = convert(img, "bgr", "gray")
    blurred = blur_gaussian(img, ksize=blur_ksize)
    return cv2.Canny(blurred, low, high)

