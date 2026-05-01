import cv2

_COLOR_MAP = {
    ("bgr", "rgb"):   cv2.COLOR_BGR2RGB,
    ("bgr", "gray"):  cv2.COLOR_BGR2GRAY,
    ("bgr", "hsv"):   cv2.COLOR_BGR2HSV,
    ("bgr", "lab"):   cv2.COLOR_BGR2LAB,
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
}


def convert(img, src, dst):
    src, dst = src.lower(), dst.lower()
    if src == dst:
        return img.copy()
    key = (src, dst)
    if key not in _COLOR_MAP:
        supported = sorted({k[0] for k in _COLOR_MAP} | {k[1] for k in _COLOR_MAP})
        raise ValueError(f"Không hỗ trợ chuyển '{src}' => '{dst}'. Các hệ màu hiện có: {supported}")
    result = cv2.cvtColor(img, _COLOR_MAP[key])
    return result