from pathlib import Path
from utils.io import read_rgb, read_gray_file, read_raw_color_file

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_files(folder):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_rgb_list(folder):
    paths = image_files(folder)
    return [read_rgb(p) for p in paths], [p.stem for p in paths]

def load_gray_list(folder):
    paths = image_files(folder)
    return [read_gray_file(p) for p in paths], [p.stem for p in paths]


def load_raw_color_list(folder):
    paths = image_files(folder)
    return [read_raw_color_file(p) for p in paths], [p.stem for p in paths]