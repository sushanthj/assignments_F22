import numpy as np
import cv2
import skimage.io 
import skimage.color
from planarH import *
from opts import get_opts
from matchPics import matchPics

def warpImage(opts):
    """
    Warp template image based on homography transform
    Args:
        opts: user inputs
    """
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    template_img = cv2.imread('../data/hp_cover.jpg')

    # make sure harry_potter image is same size as CV book
    x,y,z = image1.shape
    template_img = cv2.resize(template_img, (y,x))

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    # invert the columns of locs1 and locs2
    locs1[:, [1, 0]] = locs1[:, [0, 1]]
    locs2[:, [1, 0]] = locs2[:, [0, 1]]

    matched_points = create_matched_points(matches, locs1, locs2)
    h, inlier = computeH_ransac(matched_points[:,0:2], matched_points[:,2:], opts)

    print("homography matrix is \n", h)
    
    # compositeH(h, source, destination)
    composite_img = compositeH(h, template_img, image2)

    # Display images
    cv2.imshow("Composite Image :)", composite_img)
    cv2.waitKey()

if __name__ == "__main__":
    
    opts = get_opts()
    warpImage(opts)


