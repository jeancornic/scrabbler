
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
    (t, scaled) = cv2.threshold(scaled, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return scaled


def diff(im1, im2):
    """Distance between 2 binary images"""
    (h, w) = im1.shape
    diff = 0

    for i in range(0, h):
        for j in range(0, w):
            if im1[i,j] != im2[i,j]:
                diff = diff+1

    return diff
