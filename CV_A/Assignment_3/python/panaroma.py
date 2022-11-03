import numpy as np
import cv2
import skimage.io 
import skimage.color
from planarH import *
from opts import get_opts
from matchPics import matchPics

def panorama_computation(opts):
    """
    Compute Panorama from two input images
    Args:
        opts: user inputs

    Returns:
        image1        : pano left image
        image2        : pano right image
        composite_img : stitched panorama image
    """
    pano_left = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/a.jpg'
    pano_right = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/b.jpg'
    image1 = cv2.imread(pano_left)
    image2 = cv2.imread(pano_right)

    image1 = cv2.resize(image1, (800, 450), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (800, 450), interpolation=cv2.INTER_LINEAR)

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    # invert the columns of locs1 and locs2
    locs1[:, [1, 0]] = locs1[:, [0, 1]]
    locs2[:, [1, 0]] = locs2[:, [0, 1]]

    matched_points = create_matched_points(matches, locs1, locs2)
    h, inlier = computeH_ransac(matched_points[:,0:2], matched_points[:,2:], opts)

    print("homography matrix is \n", h)
    # compositeH (h, source, destination)
    # composite_img = panorama(h, image1, image2)
    composite_img = panorama_composite(h, image2, image1)
    composite_img = trim_images(composite_img, image2)

    cv2.imwrite('/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/outputs/pano_image_2.png', composite_img)
    cv2.imshow("pano image", composite_img)
    cv2.waitKey()
    
    return image1, image2, composite_img


if __name__ == '__main__':
    opts = get_opts()
    ref_img_1, ref_img_2, pano_img = panorama_computation(opts)