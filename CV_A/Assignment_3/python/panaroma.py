import numpy as np
import cv2
import skimage.io 
import skimage.color
from planarH import *
from opts import get_opts
from matchPics import matchPics

def main(opts):
    pano_left = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/a.jpg'
    pano_right = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/b.jpg'
    image1 = cv2.imread(pano_left)
    image2 = cv2.imread(pano_right)

    # row1 = np.array([controlpointlist[0]['img1_x'], controlpointlist[0]['img1_y']], dtype=np.int8)
    # row2 = np.array([controlpointlist[1]['img1_x'], controlpointlist[1]['img1_y']], dtype=np.int8)
    # row3 = np.array([controlpointlist[2]['img1_x'], controlpointlist[2]['img1_y']], dtype=np.int8)
    # row4 = np.array([controlpointlist[3]['img1_x'], controlpointlist[3]['img1_y']], dtype=np.int8)

    # row5 = np.array([controlpointlist[0]['img2_x'], controlpointlist[0]['img2_y']], dtype=np.int8)
    # row6 = np.array([controlpointlist[1]['img2_x'], controlpointlist[1]['img2_y']], dtype=np.int8)
    # row7 = np.array([controlpointlist[2]['img2_x'], controlpointlist[2]['img2_y']], dtype=np.int8)
    # row8 = np.array([controlpointlist[3]['img2_x'], controlpointlist[3]['img2_y']], dtype=np.int8)

    # x1 = np.stack((row1, row2, row3, row4), axis=0)
    # x2 = np.stack((row5, row6, row7, row8), axis=0)

    # print("x1 is", x1)
    # print("x2 is", x2)

    # h = computeH_norm(x1, x2)
    # h = np.linalg.inv(h)

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
    composite_img = panorama(h, image1, image2)
    # composite_img = panorama_composite(h, image2, image1)

    cv2.imwrite('/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/outputs/pano_image_2.png', composite_img)
    cv2.imshow("pano image", composite_img)
    cv2.waitKey()


if __name__ == '__main__':
    opts = get_opts()
    main(opts)