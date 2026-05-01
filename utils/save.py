import cv2
from pathlib import Path
from utils.dataset import ensure_dir

def save_rgb(img, path):
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_gray(img, path):
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def save_raw_color(img, path):
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def save_image_list(images, names, folder, mode="rgb"):
    folder = ensure_dir(folder)
    for img, name in zip(images, names):
        path = folder / f"{Path(name).stem}.png"
        if mode == "rgb":
            save_rgb(img, path)
        elif mode == "gray":
            save_gray(img, path)
        else:
            save_raw_color(img, path)