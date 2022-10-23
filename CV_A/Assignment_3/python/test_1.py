import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color
from scipy import ndimage
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

def computeH(x1, x2):
    """
    Computes the homography based on 
    matching points in both images

    Args:
        x1: keypoints in image 1
        x2: keypoints in image 2

    Returns:
        H2to1: the homography matrix
    """

    # Define a dummy H matrix
    A_build = []
    
    #? Define the A matrix for (Ah = 0) (A matrix size = N*2 x 9)
    for i in range(x1.shape[0]):
        row_1 = np.array([ x2[i,0], x2[i,1], 1, 0, 0, 0, -x1[i,0]*x2[i,0], -x1[i,0]*x2[i,1], -x1[i,0] ])
        row_2 = np.array([ 0, 0, 0, x2[i,0], x2[i,1], 1, -x1[i,1]*x2[i,0], -x1[i,1]*x2[i,1], -x1[i,1] ])
        A_build.append(row_1)
        A_build.append(row_2)
    
    A = np.stack(A_build, axis=0)

    #? Do the least squares minimization to get the homography matrix
    #? this is done as eigenvector coresponding to smallest eigen value of A`A = H matrix
    #! Which one is correct?
    u, s, v = np.linalg.svd(np.matmul(A.T, A))
    # u, s, v = np.linalg.svd(A)

    # here the linalg.svd gives v_transpose
    # but we need just V therefore we again transpose
    v = np.transpose(v)

    # print("the V matrix is \n", np.round(v, 3))
    # print("\n the singular matrix is \n", s)

    H2to1 = np.reshape(v[:,-1], (3,3))
    # print("unnormed Homography matrix is \n", H2to1)
    return H2to1

def computeH_norm(x1, x2):
    """
    Compute the normalized coordinates
    and also the homography matrix using computeH

    Args:
        x1 (Mx2): the matched locations of corners in img1
        x2 (Mx2): the matched locations of corners in img2

    Returns:
        H2to1: Hmography matrix after denormalization
    """
    # Q2.2.2
    # Compute the centroid of the points
    centroid_img_1 = np.sum(x1, axis=0)/x1.shape[0]
    centroid_img_2 = np.sum(x2, axis=0)/x2.shape[0]

    # print(f'centroid of img1 is {centroid_img_1} \n centroid of img2 is {centroid_img_2}')

    # Shift the origin of the points to the centroid
    # let origin for img1 be centroid_img_1 and similarly for img2
    #? Now translate every point such that centroid is at [0,0]
    moved_x1 = x1 - centroid_img_1
    moved_x2 = x2 - centroid_img_2

    #! Remove below lines after testing
    moved_centroid_img1 = np.sum(moved_x1, axis=0)/x1.shape[0]
    moved_centroid_img2 = np.sum(moved_x2, axis=0)/x2.shape[0]

    # print(f'updated centroid of img1 is {moved_centroid_img1} \n updated centroid of img2 is {moved_centroid_img2}')

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    # current_avg_dist_img1 = np.sqrt(np.sum(moved_x1[:,0]*moved_x1[:,0], axis=0) + np.sum(moved_x1[:,1]*moved_x1[:,1], axis=0))
    # current_avg_dist_img2 = np.sqrt(np.sum(moved_x2[:,0]*moved_x2[:,0], axis=0) + np.sum(moved_x2[:,1]*moved_x2[:,1], axis=0))

    current_avg_dist_img1 = np.sum(np.sqrt(( (moved_x1[:,0] * moved_x1[:,0])+(moved_x1[:,1]*moved_x1[:,1]) )))/x1.shape[0]
    current_avg_dist_img2 = np.sum(np.sqrt(( (moved_x2[:,0] * moved_x2[:,0])+(moved_x2[:,1]*moved_x2[:,1]) )))/x2.shape[0]
    
    # print(f'avg dist of img1 is {current_avg_dist_img1} \n avg dist of img2 is {current_avg_dist_img2}')
    # moved and scaled image 1 points

    scale1 = (1 / (current_avg_dist_img1)) * np.sqrt(2)
    scale2 = (1 / (current_avg_dist_img2)) * np.sqrt(2)
    moved_scaled_x1 = moved_x1 * scale1
    moved_scaled_x2 = moved_x2 * scale2

    # print("scale1 is", scale1)
    # print("scale2 is", scale2)

    #! Remove the below lines after testing
    updated_avg_dist_img1 = np.sum(np.sqrt(( (moved_scaled_x1[:,0] * moved_scaled_x1[:,0])+(moved_scaled_x1[:,1]*moved_scaled_x1[:,1]) )))/x1.shape[0]
    updated_avg_dist_img2 = np.sum(np.sqrt(( (moved_scaled_x2[:,0] * moved_scaled_x2[:,0])+(moved_scaled_x2[:,1]*moved_scaled_x2[:,1]) )))/x2.shape[0]
    # print(f'avg dist of img1 is {updated_avg_dist_img1} \n avg dist of img2 is {updated_avg_dist_img2}')

    # Similarity transform 1
    #? We construct the transformation matrix to be 3x3 as it has to be same shape of Homography
    t1 = np.diag([scale1, scale1, 1])
    t1[0:2,2] = -scale1*centroid_img_1

    # Similarity transform 2
    t2 = np.diag([scale2, scale2, 1])
    t2[0:2,2] = -scale2*centroid_img_2

    # Compute homography
    H = computeH(moved_scaled_x1, moved_scaled_x2)

    # Denormalization
    H2to1 = np.matmul(np.linalg.inv(t1), np.matmul(H, t2))

    return H2to1

def create_matched_points(matches, locs1, locs2):
    """
    Match the corners in img1 and img2 according to the BRIEF matched points

    Args:
        matches (Mx2): Vector containing the index of locs1 and locs2 which matches
        locs1 (Nx2): Vector containing corner positions for img1
        locs2 (Nx2):  Vector containing corner positions for img2

    Returns:
        _type_: _description_
    """
    matched_points = np.array([0.0,0.0,0.0,0.0], dtype=np.float32)
    for i in range(matches.shape[0]):
        matched_points = np.vstack((matched_points,
                                    np.array([locs1[matches[i,0],0],
                                              locs1[matches[i,0],1],
                                              locs2[matches[i,1],0],
                                              locs2[matches[i,1],1]])))
    
    # remove the first dummy value and return
    return matched_points[1:,:]

def test_homography(im_src, im_dst, h):
    h = np.linalg.inv(h)
    # Warp destination image to source image based on homography
    im_out = cv2.warpPerspective(im_dst, h, (im_dst.shape[1],im_dst.shape[0]))

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Destination Image", im_out)
    cv2.waitKey()

def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    #? Create a boolean vector of length N where 1 = inlier and 0 = outlier
    

    return bestH2to1, inliers

if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = ndimage.rotate(image1, 3)
    
    image1_gray = skimage.color.rgb2gray(image1)
    image2_gray = skimage.color.rgb2gray(image2)

    matches, locs1, locs2 = matchPics(image1, image2, opts)
    matched_points = create_matched_points(matches, locs1, locs2)
    h = computeH_ransac(matched_points[:,0:2], matched_points[:,2:])

    print("homography matrix is \n", h)

    test_homography(image1_gray, image2_gray, h)
