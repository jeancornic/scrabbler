
import numpy as np
import cv2

FEATURE_SIZE=(16,16)


def pad(image):
    """Pad image to a squared one"""
    (h, w) = image.shape
    s = max(h, w)

    squared = np.zeros((s, s), np.uint8)
    squared.fill(255)

    # Padding is made on the smallest dimension,
    # on both sides to keep symetry
    start_x = int((s - w) / 2)
    start_y = int((s - h) / 2)

    squared[start_y:start_y+h,start_x:start_x+w] = image

    return squared


def scale(image):

    scaled = cv2.resize(image, dsize=FEATURE_SIZE, interpolation=cv2.INTER_CUBIC)

    return image



