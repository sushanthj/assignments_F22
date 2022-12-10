import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

from skimage import data
from skimage.filters.thresholding import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None

    # apply threshold
    image = skimage.color.rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(3))
    black_white = deepcopy(bw)

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    bbox_list = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 200:
            bbox_list.append(region.bbox)

    # print(bbox_list)

    return bbox_list, bw