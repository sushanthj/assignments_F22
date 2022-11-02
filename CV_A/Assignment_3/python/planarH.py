from copy import deepcopy
from dataclasses import replace
from platform import python_branch
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
from tqdm import tqdm


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
    """
    Every iteration we init a Homography matrix using 4 corresponding
    points and calculate number of inliers. Finally use the Homography
    matrix which had max number of inliers (and these inliers as well)
    to find the final Homography matrix
    Args:
        locs1: location of matched points in image1
        locs2: location of matched points in image2
        opts: user inputs used for distance tolerance in ransac

    Returns:
        bestH2to1     : The homography matrix with max number of inliers
        final_inliers : Final list of inliers considered for homography
    """
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
    print("Computing RANSAC")
    for i in tqdm(range(max_iters)):
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
        inliers, inlier_count, distance_error, error_state = compute_inliers(ref_H, 
                                                                            test_locs1,
                                                                            test_locs2, 
                                                                            inlier_tol)

        if error_state == 1:
            continue

        if (inlier_count > final_inlier_count) and (distance_error < final_distance_error):
            final_inlier_count = inlier_count
            final_inliers = inliers
            final_corresp_points_1 = correspondence_points_1
            final_corresp_points_2 = correspondence_points_2
            final_distance_error = distance_error
            final_test_locs1 = test_locs1
            final_test_locs2 = test_locs2

    if final_distance_error != 10000:
        # print("original point count is", locs1.shape[0])
        # print("final inlier count is", final_inlier_count)
        # print("final inlier's cumulative distance error is", final_distance_error)

        delete_indexes = np.where(final_inliers==0)
        final_locs_1 = np.delete(final_test_locs1, delete_indexes, axis=0)
        final_locs_2 = np.delete(final_test_locs2, delete_indexes, axis=0)

        final_locs_1 = np.vstack((final_locs_1, final_corresp_points_1))
        final_locs_2 = np.vstack((final_locs_2, final_corresp_points_2))

        bestH2to1 = computeH_norm(final_locs_1, final_locs_2)
        return bestH2to1, final_inliers
    
    else:
        bestH2to1 = computeH_norm(correspondence_points_1, correspondence_points_2)
        return bestH2to1, 0

def compute_inliers(h, x1, x2, tol):
    """
    Compute the number of inliers for a given
    homography matrix
    Args:
        h: Homography matrix
        x1 : matched points in image 1
        x2 : matched points in image 2
        tol: tolerance value to check for inliers

    Returns:
        inliers         : indexes of x1 or x2 which are inliers
        inlier_count    : number of total inliers
        dist_error_sum  : Cumulative sum of errors in reprojection error calc
        flag            : flag to indicate if H was invertible or not
    """
    # take H inv to map points in x1 to x2
    try:
        H = np.linalg.inv(h)
    except:
        return [1,1,1], 1, 1, 1

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
    
    return inliers, inlier_count, np.sum(dist_error), 0


def compositeH(H2to1, template, img):
    """
    Create a composite image after warping the template image on top
    of the image using the homography

    Args:
        H2to1 : Existing(already found) homography matrix
        template: Harry Potter (template image)
        img: Base image onto which we overlay Harry Potter image

    Returns:
        composite_img: Base image with overlayed Harry Potter cover
    """
    output_shape = (img.shape[1],img.shape[0])
    # destination_img = img
    # source_img = template
    h = np.linalg.inv(H2to1)

    # Create mask of same size as template
    mask = np.ones((template.shape[0], template.shape[1]))*255
    mask = np.stack((mask, mask, mask), axis=2)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, h, output_shape)

    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, h, output_shape)

    # Use mask to combine the warped template and the image
    composite_img = np.where(warped_mask, warped_template, img)
    
    return composite_img

def panorama(H2to1, template, img):
    h = np.linalg.inv(H2to1)
    h1,w1 = template.shape[:2]
    h2,w2 = img.shape[:2]

    # build corresp points to check for translation
    row1 = np.array([0,0])
    row2 = np.array([h1,0])
    row3 = np.array([0,w1])
    row4 = np.array([h1, w1])

    x1 = np.stack((row1, row2, row3, row4), axis=0)

    x1_extd = (np.append(x1, np.ones((x1.shape[0],1)), axis=1))
    x2_est = np.zeros((x1_extd.shape), dtype=x1_extd.dtype)

    for i in range(x1.shape[0]):
        x2_est[i,:] = h @ x1_extd[i,:]
    
    x2_est = x2_est/np.expand_dims(x2_est[:,2], axis=1)

    max_arr = np.max(x2_est.astype(int), axis=0)
    min_arr = np.min(x2_est.astype(int), axis=0)

    t = np.array([-min_arr[0],-min_arr[1]])
    H_t = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    # shape_arr = max_arr-min_arr
    # output_shape = (shape_arr[1], shape_arr[0])

    output_shape = (w2+t[0], h2+t[1]+50)

    print("H_t is", H_t)
    print("output shape is", output_shape)
    print("t is", t)

    h = np.matmul(H_t, h)

    warped_template = cv2.warpPerspective(template, h, output_shape)

    cv2.imshow("warped template", warped_template)
    cv2.waitKey()

    warped_template[t[1]:(h2+t[1]),t[0]:(w2+t[0]),:] = img[:,:,:]
    
    return warped_template

def panorama_composite(H2to1, template, img):
    """
    Stitch two images together to form a panorama

    Args:
        H2to1: Homography Matrix
        template: The pano_right image
        img: The pano_left image

    Returns:
        composite_img: Stitched image (panorama)
    """
    output_shape = (img.shape[1]+240,img.shape[0]+240)
    # destination_img = img
    # source_img = template
    h = H2to1
    
    img_padded = np.zeros((img.shape[0]+240,img.shape[1]+240,3), dtype=img.dtype)
    img_padded[0:img.shape[0], 0:img.shape[1], :] = img[:,:,:]

    # Create mask of same size as template
    mask = np.ones((template.shape[0], template.shape[1]))*255
    mask = np.stack((mask, mask, mask), axis=2)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, h, output_shape)

    # Warp template by appropriate homography
    cv2.imshow("template image", template)
    cv2.waitKey()
    cv2.imshow("destination image", img)
    cv2.waitKey()
    warped_template = cv2.warpPerspective(template, h, output_shape)

    cv2.imshow("warped template", warped_template)
    cv2.waitKey()

    # Use mask to combine the warped template and the image
    composite_img = np.where(warped_mask, warped_template, img_padded)
    
    return composite_img

def trim_images(img, ref_img):
    """
    Args:
        img: panorama image to be trimmed
        ref_img: reference image to get dimensions

    Returns:
        trimmed_img: trimmed panorama image
    """
    desired_height, _, _ = ref_img.shape
    _, desired_width, _ = img.shape

    trimmed_img = img[0:desired_height, 0:desired_width, :]
    cv2.imshow("trimmed image", trimmed_img)
    cv2.waitKey()
    cv2.imwrite('/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/outputs/pano_trim_image_2.png', trimmed_img)

    return trimmed_img
    


