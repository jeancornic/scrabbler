"""
Neural netork : train & classify
Detect letter with distance to training set
"""

import numpy as np
import os
import cv2
import imaging
from matplotlib import pyplot as plt

def train(training_set_path):

    # Read trainingset
    # For each image:
    #   crop letter
    #   scale letter
    #   convert to 1 & 0
    #   insert in neural network
    #
    # Export xml file
    return

def classify(letter, ann_conf):
    return


def get_letter(im, letter_dirs):
    """Detect letter by computing distance to a training set"""
    min_distance = None
    nearest_letter = None
    nearest_letter_im = None

    # Get the nearest image
    for directory in letter_dirs:
        name = os.listdir(directory)[0]
        letter_im = cv2.imread(os.path.join(directory, name), 0)
        distance = imaging.diff(im, letter_im)

        if (min_distance is None) or (min_distance > distance):
            min_distance = distance
            nearest_letter = os.path.basename(directory)
            nearest_letter_im = letter_im

    return nearest_letter, min_distance

