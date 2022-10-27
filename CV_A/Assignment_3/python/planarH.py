from copy import deepcopy
from dataclasses import replace
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.color
import math
import random
from scipy import ndimage
from scipy.spatial import distance
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts


def computeH(x1, x2):
    #Q2.2.1
    
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
    
    # Define the A matrix for (Ah = 0) (A matrix size = N*2 x 9)
    for i in range(x1.shape[0]):
        row_1 = np.array([ x2[i,0], x2[i,1], 1, 0, 0, 0, -x1[i,0]*x2[i,0], -x1[i,0]*x2[i,1], -x1[i,0] ])
        row_2 = np.array([ 0, 0, 0, x2[i,0], x2[i,1], 1, -x1[i,1]*x2[i,0], -x1[i,1]*x2[i,1], -x1[i,1] ])
        A_build.append(row_1)
        A_build.append(row_2)
    
    A = np.stack(A_build, axis=0)

    # Do the least squares minimization to get the homography matrix
    # this is done as eigenvector coresponding to smallest eigen value of A`A = H matrix
    u, s, v = np.linalg.svd(np.matmul(A.T, A))

    # here the linalg.svd gives v_transpose
    # but we need just V therefore we again transpose
    H2to1 = np.reshape(v.T[:,-1], (3,3))
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
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

    current_max_dist_img1 = np.max(np.linalg.norm(moved_x1),axis=0)
    current_max_dist_img2 = np.max(np.linalg.norm(moved_x2),axis=0)

    
    # moved and scaled image 1 points
    scale1 = np.sqrt(2) / (current_max_dist_img1)
    scale2 = np.sqrt(2) / (current_max_dist_img2)
    moved_scaled_x1 = moved_x1 * scale1
    moved_scaled_x2 = moved_x2 * scale2

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
    matched_pts = []
    for i in range(matches.shape[0]):
        matched_pts.append(np.array([locs1[matches[i,0],0],
                                              locs1[matches[i,0],1],
                                              locs2[matches[i,1],0],
                                              locs2[matches[i,1],1]]))
    
    # remove the first dummy value and return
    matched_points = np.stack(matched_pts, axis=0)
    return matched_points

def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    # define size of both locs1 and locs2
    num_rows = locs1.shape[0]

    # define a container for keeping track of inlier counts
    final_inlier_count = 0
    final_distance_error = 10000

    #? Create a boolean vector of length N where 1 = inlier and 0 = outlier
    for i in range(max_iters):
        test_locs1 = deepcopy(locs1)
        test_locs2 = deepcopy(locs2)
        # chose a random sample of 4 points to find H
        rand_index = []
        
        rand_index = random.sample(range(int(locs1.shape[0])),k=4)
        
        rand_points_1 = []
        rand_points_2 = []
        
        for j in rand_index:
            rand_points_1.append(locs1[j,:])
            rand_points_2.append(locs2[j,:])
        
        test_locs1 = np.delete(test_locs1, rand_index, axis=0)
        test_locs2 = np.delete(test_locs2, rand_index, axis=0)
            
        correspondence_points_1 = np.vstack(rand_points_1)
        correspondence_points_2 = np.vstack(rand_points_2)

        ref_H = computeH_norm(correspondence_points_1, correspondence_points_2)
        inliers, inlier_count, distance_error = compute_inliers(ref_H, test_locs1, test_locs2, inlier_tol)

        if (inlier_count > final_inlier_count) and (distance_error < final_distance_error):
            final_inlier_count = inlier_count
            final_inliers = inliers
            final_corresp_points_1 = correspondence_points_1
            final_corresp_points_2 = correspondence_points_2
            final_distance_error = distance_error
            final_test_locs1 = test_locs1
            final_test_locs2 = test_locs2

    print("original point count is", locs1.shape[0])
    print("final inlier count is", final_inlier_count)
    print("final inlier's cumulative distance error is", final_distance_error)

    delete_indexes = np.where(final_inliers==0)
    print("delete indexes is", delete_indexes)
    final_locs_1 = np.delete(final_test_locs1, delete_indexes, axis=0)
    final_locs_2 = np.delete(final_test_locs2, delete_indexes, axis=0)

    final_locs_1 = np.vstack((final_locs_1, final_corresp_points_1))
    final_locs_2 = np.vstack((final_locs_2, final_corresp_points_2))

    print("refined_locs1 shape is", final_locs_1.shape)
    print("refined_locs2 shape is", final_locs_2.shape)

    bestH2to1 = computeH_norm(final_locs_1, final_locs_2)
    return bestH2to1, final_inliers

def compute_inliers(h, x1, x2, tol):
    # take H inv to map points in x1 to x2
    H = np.linalg.inv(h)

    x2_extd = np.append(x2, np.ones((x2.shape[0],1)), axis=1)
    x1_extd = (np.append(x1, np.ones((x1.shape[0],1)), axis=1))
    x2_est = np.zeros((x2_extd.shape), dtype=x2_extd.dtype)

    for i in range(x1.shape[0]):
        x2_est[i,:] = H @ x1_extd[i,:]
    
    x2_est = x2_est/np.expand_dims(x2_est[:,2], axis=1)
    dist_error = np.linalg.norm((x2_extd-x2_est),axis=1)
    
    # print("dist error is", dist_error)
    inliers = np.where((dist_error < tol), 1, 0)
    inlier_count = np.count_nonzero(inliers == 1)
    
    return inliers, inlier_count, np.sum(dist_error)


def compositeH(H2to1, template, img):
    # destination_img = img
    # source_img = template
    h = np.linalg.inv(H2to1)
    # im_out = cv2.warpPerspective(im_dst, h, (im_src.shape[1],im_src.shape[0]))
    
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    mask = np.ones((template.shape[0], template.shape[1]))*255
    mask = np.stack((mask, mask, mask), axis=2)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, h, (img.shape[1],img.shape[0]))

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, h, (img.shape[1],img.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = np.where(warped_mask, warped_template, img)
    
    return composite_img


