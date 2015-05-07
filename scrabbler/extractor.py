#!/usr/bin/end python

"""
For now, detect grid and extract letters.
It only works for a Scrabble screenshot.
"""

import numpy as np
import cv2
import imaging
from matplotlib import pyplot as plt

def extract_letters(im):
    """
    Returns a list of images of letters, with their coordinates.
    """

    ## Treshold image to black and white
    (threshold, im_bw) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    ## Contours detection
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    letters = []

    for i, contour in enumerate(contours):
        (a, b, c, d) = bounding_box = cv2.boundingRect(contour)

        ## We keep contours that have size of letters
        if not ((c > 20 or d > 20) and (d > 27 and d < 30) and b < 1000):
            continue

        letter = extract_contour(contours, hierarchy, i, bounding_box)

        letter = imaging.pad(letter)
        letter = imaging.scale(letter)
        # plt.imshow(letter, cmap=plt.cm.Greys_r)
        # plt.show()

        # Compute center of letter
        center = (a + c / 2, b + d / 2)

        letters.append((center, letter))

    return letters


def extract_contour(contours, hierarchy, i, bounding_box):
    """
    Extracts a letter from a (parent) contour as a new image
    """

    (a,b,c,d) = bounding_box

    ## Print into a separated image
    letter = np.zeros((d, c), np.uint8)
    letter.fill(255)

    ## Draw letter as black (filled)
    cv2.drawContours(letter, contours, i, (0,0,0), thickness=-1, offset=(-a,-b))

    # plt.imshow(letter), plt.show()
    child_id = hierarchy[0][i][2]

    ## Draw all child interior as white filled (letter holes)
    while child_id != -1:
        cv2.drawContours(letter, contours, child_id, (255,255,255), thickness=-1, offset=(-a,-b))
        child_id = hierarchy[0][child_id][0]

    return letter
