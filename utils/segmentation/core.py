import cv2
import numpy as np

def get_display_and_intensity(img, mode="rgb"):
    """
    Trả về:
    - display_img: ảnh RGB để hiển thị / overlay
    - intensity: ảnh 1 kênh dùng cho Otsu, Adaptive, Watershed

    mode:
    - rgb  : display = RGB, intensity = RGB -> Gray
    - gray : display = Gray -> RGB, intensity = Gray
    - hsv  : display = HSV -> RGB, intensity = V channel
    - lab  : display = LAB -> RGB, intensity = L channel
    """

    if mode == "rgb":
        display_img = img.copy()
        intensity = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif mode == "gray":
        intensity = img.copy()
        display_img = cv2.cvtColor(intensity, cv2.COLOR_GRAY2RGB)

    elif mode == "hsv":
        display_img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # HSV: dùng kênh V cho threshold / watershed
        H, S, V = cv2.split(img)
        intensity = V

    elif mode == "lab":
        display_img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        # LAB: dùng kênh L cho threshold / watershed
        L, A, B = cv2.split(img)
        intensity = L

    else:
        raise ValueError("mode phải là: rgb, gray, hsv hoặc lab")

    display_img = np.clip(display_img, 0, 255).astype(np.uint8)
    intensity = np.clip(intensity, 0, 255).astype(np.uint8)

    return display_img, intensity

def get_kmeans_features(img, mode="rgb"):
    """
    Trả về feature dùng cho K-Means.

    mode = rgb  -> [R, G, B]
    mode = gray -> [gray]
    mode = hsv  -> [H, S, V]
    mode = lab  -> [L, A, B]
    """

    if mode == "gray":
        h, w = img.shape
        features = img.reshape((-1, 1)).astype(np.float32)

    elif mode in ["rgb", "hsv", "lab"]:
        h, w = img.shape[:2]
        features = img.reshape((-1, 3)).astype(np.float32)

    else:
        raise ValueError("mode phải là: rgb, gray, hsv hoặc lab")

    return features, h, w