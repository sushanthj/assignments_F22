import random
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, im1, im2, nIters=500, tol=2):
    """
    Every iteration we init a Homography matrix using 4 corresponding
    points and calculate number of inliers. Finally use the Homography
    matrix which had max number of inliers (and these inliers as well)
    to find the final Homography matrix
    Args:
        pts1: location of matched points in image1
        pts2: location of matched points in image2
        opts: user inputs used for distance tolerance in ransac

    Returns:
        bestH2to1     : The homography matrix with max number of inliers
        final_inliers : Final list of inliers considered for homography
    """
    max_iters = nIters # the number of iterations to run RANSAC for
    inlier_tol = tol # the tolerance value for considering a point to be an inlier
    locs1 = pts1
    locs2 = pts2

    # define size of both locs1 and locs2
    num_rows = locs1.shape[0]

    # define a container for keeping track of inlier counts
    final_inlier_count = 0
    final_distance_error = 10000

    #? Create a boolean vector of length N where 1 = inlier and 0 = outlier
    print("Computing RANSAC")
    for i in range(max_iters):
        test_locs1 = deepcopy(locs1)
        test_locs2 = deepcopy(locs2)
        # chose a random sample of 4 points to find H
        rand_index = []
        
        rand_index = random.sample(range(int(locs1.shape[0])), k=8)
        
        rand_points_1 = []
        rand_points_2 = []
        
        for j in rand_index:
            rand_points_1.append(locs1[j,:])
            rand_points_2.append(locs2[j,:])
        
        test_locs1 = np.delete(test_locs1, rand_index, axis=0)
        test_locs2 = np.delete(test_locs2, rand_index, axis=0)
            
        correspondence_points_1 = np.vstack(rand_points_1)
        correspondence_points_2 = np.vstack(rand_points_2)

        ref_F = eightpoint(correspondence_points_1, correspondence_points_2, M)
        inliers, inlier_count, distance_error, error_state = compute_inliers(ref_F, 
                                                                            test_locs1,
                                                                            test_locs2, 
                                                                            inlier_tol,
                                                                            im1,
                                                                            im2)

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

        bestH2to1 = eightpoint(final_locs_1, final_locs_2, M)
        return bestH2to1, final_inliers
    
    else:
        bestH2to1 = eightpoint(correspondence_points_1, correspondence_points_2, M)
        return bestH2to1, 0
    

def compute_inliers(f, x1, x2, tol, im1, im2):
    """
    Compute the number of inliers for a given
    Fundamental matrix
    Args:
        h  : Fundamental matrix
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
    # try:
    #     F = np.linalg.inv(f)
    # except:
    #     return [1,1,1], 1, 1, 1

    x2_est = np.zeros((x2.shape), dtype=x2.dtype)

    for i in range(x1.shape[0]):
        # print("the x and y points are", x1[i,0], x1[i,1])
        x2_est[i,0], x2_est[i,1] = epipolarCorrespondence(im1, im2, f, x1[i,0], x1[i,1])  #F @ x1_extd[i,:]
    
    # print("shape of x2 and x2_est is", x2.shape, x2_est.shape)
    print("diff shape is", (x2-x2_est).shape)
    dist_error = np.linalg.norm((x2-x2_est),axis=1)
    
    print("dist error is", dist_error.shape)
    inliers = np.where((dist_error < tol), 1, 0)
    inlier_count = np.count_nonzero(inliers == 1)
    print("inlier count is", inlier_count)
    
    return inliers, inlier_count, np.sum(dist_error), 0


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    raise NotImplementedError()
    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    templeCoords = np.load('data/templeCoords.npz')
    temple_pts1 = np.hstack([templeCoords["x1"], templeCoords["y1"]])

    M = np.max([*im1.shape, *im2.shape])

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M, im1, im2)

    F_naieve = eightpoint(noisy_pts1, noisy_pts2, M)

    # use displayEpipolarF to compare how ransac_F and naieve_F behave
    displayEpipolarF(im1, im2, F)
    displayEpipolarF(im1, im2, F_naieve)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    
    # YOUR CODE HERE


    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)



    # YOUR CODE HERE