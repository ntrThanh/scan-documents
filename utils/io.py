import cv2
from pathlib import Path

def read_rgb(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray_file(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh gray: {path}")
    return img


def read_raw_color_file(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không đọc được ảnh raw color: {path}")
    return img