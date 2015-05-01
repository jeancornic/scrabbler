#!/usr/bin/env python

import numpy as np
import cv2
from os import path
import argparse
from matplotlib import pyplot as plt


def main(image_path):
    image_dir = path.dirname(image_path)
    image_file = path.basename(image_path)
    image_name, extension = path.splitext(image_file)

    def write_image(image, suffix):
        cv2.imwrite(path.join(image_dir, image_name + suffix + extension), image)

    im = cv2.imread(image_path, 0)
    ## Fill image with white
    h, w = im.shape
    im = np.zeros((h, w), np.uint8)
    im.fill(255)

    ## Read image as grayscale
    im_gray = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    ## Treshold image to black and white
    (threshold, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    write_image(im_bw, '_bw')

    ## Contours detection
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        (a, b, c, d) = cv2.boundingRect(contour)

        ## TODO: Use a clever filter to target letters
        ## We keep contours that have size of letters
        if not ((c > 20 or d > 20) and (d > 27 and d < 30) and b < 1000):
            continue

        ## Draw letter as black (filled)
        cv2.drawContours(im, contours, i, (0,0,0), thickness=-1)

        child_id = hierarchy[0][i][2]

        ## Draw all child interior as white filled (letter holes)
        while child_id != -1:
            cv2.drawContours(im, contours, child_id, (255,255,255), thickness=-1)
            child_id = hierarchy[0][child_id][0]

    write_image(im, '_contours')



if __name__ == '__main__':
    # Read path of image from args
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="The image to recognize")
    args = parser.parse_args()

    image = args.image
    image_path = path.abspath(image)

    main(image_path)

