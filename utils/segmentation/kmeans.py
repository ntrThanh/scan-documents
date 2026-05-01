import cv2
import numpy as np
from .core import get_kmeans_features

def kmeans_pixel_segmentation(img, mode="rgb", K=2):
    """
    K-Means theo giá trị pixel.

    RGB  -> phân cụm theo R, G, B
    Gray -> phân cụm theo mức xám
    HSV  -> phân cụm theo H, S, V
    LAB  -> phân cụm theo L, A, B
    """

    features, h, w = get_kmeans_features(img, mode)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        features,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    label_mask = labels.reshape((h, w))

    # Cộng 1 để label 0 không bị xem là nền
    label_mask = label_mask + 1

    return label_mask.astype(np.uint8)

def kmeans_spatial_segmentation(img, mode="rgb", K=2, spatial_weight=0.6):
    """
    K-Means kết hợp feature màu và tọa độ.

    RGB  -> [R, G, B, x, y]
    Gray -> [gray, x, y]
    HSV  -> [H, S, V, x, y]
    LAB  -> [L, A, B, x, y]
    """

    color_features, h, w = get_kmeans_features(img, mode)

    y_coords, x_coords = np.indices((h, w))

    x_coords = x_coords.reshape((-1, 1)).astype(np.float32)
    y_coords = y_coords.reshape((-1, 1)).astype(np.float32)

    # Chuẩn hóa tọa độ về 0-255
    x_coords = x_coords / w * 255
    y_coords = y_coords / h * 255

    spatial_features = np.hstack([x_coords, y_coords])
    spatial_features = spatial_features * spatial_weight

    features = np.hstack([color_features, spatial_features]).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        features,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    label_mask = labels.reshape((h, w))

    # Cộng 1 để overlay rõ
    label_mask = label_mask + 1

    return label_mask.astype(np.uint8)