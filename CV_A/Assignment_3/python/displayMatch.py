import numpy as np
import cv2
import time
from matchPics import matchPics
from scipy import ndimage
from helper import plotMatches
from opts import get_opts


def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """

    start = time.time()
    matches, locs1, locs2 = matchPics(image1, image2, opts)
    end = time.time()
    print("processing time is", round((end-start), 2))

    # display matched features
    plotMatches(image1, image2, matches, locs1, locs2)

if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
