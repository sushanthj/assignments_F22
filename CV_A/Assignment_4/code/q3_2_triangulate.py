import os
import math
import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint, check_and_create_directory
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    """
    Find the 3D coords of the keypoints

    We are given camera matrices and 2D correspondences.
    We can therefore find the 3D points (refer L17 (Camera Models) of CV slides)

    Note. We can't just use x = PX to compute the 3D point X because of scale ambiguity
          i.e the ambiguity can be rep. as x = alpha*Px (we cannot find alpha)
          Therefore we need to do DLT just like the case of homography 
          (see L14 (2D transforms) CVB slide 61)

    Args:
        C1   : the 3x4 camera matrix of camera 1
        pts1 : img coords of keypoints in camera 1 (Nx2)
        C2   : the 3x4 camera matrix of camera 2
        pts2 : img coords of keypoints in camera 2 (Nx2)

    Returns:
        P    : the estimated 3D point for the given pair of keypoint correspondences
        err  : the reprojection error
    """
    P = np.zeros(shape=(1,3))
    err = 0

    for i in range(len(pts1)):
        # get the camera 1 matrix
        p1_1 = C1[0,:]
        p2_1 = C1[1,:]
        p3_1 = C1[2,:]

        # get the camera 2 matrix
        p1_2 = C2[0,:]
        p2_2 = C2[1,:]
        p3_2 = C2[2,:]

        x, y = pts1[i,0], pts1[i,1]
        x2, y2 = pts2[i,0], pts2[i,1]

        # calculate the A matrix for this point correspondence
        A = np.array([y*p3_1 - p2_1, p1_1 - x*p3_1, y2*p3_2 - p2_2, p1_2 - x2*p3_2])
        u, s, v = np.linalg.svd(A)

        # here the linalg.svd gives v_transpose
        # but we need just V therefore we again transpose
        X = v.T[:,-1]
        
        # conver X to homogenous coords
        X = np.reshape(X, newshape=(1,4))
        X = X/X[0,3]

        P = np.concatenate((P, X[:,0:3]), axis=0)
        
        X = X.T

        # find the error for this projection
        # 3x1 = 3x4 . 3x1 
        pt_1 = ((C1 @ X)/(C1 @ X)[2,0])[0:2,0]
        pt_2 = ((C2 @ X)/(C2 @ X)[2,0])[0:2,0]
        
        # print("found point 1 is", pt_1)
        # print("orig point 1 is", pts1[i,:])
        # print("diff vals are", pt_1 - pts1[i,:])
        # print("norm squared is is", np.linalg.norm(pt_1 - pts1[i,:])*np.linalg.norm(pt_1 - pts1[i,:]))

        err += np.linalg.norm(pt_1 - pts1[i,:])**2 + np.linalg.norm(pt_2 - pts2[i,:])**2

    print("error in this iteration is", err)
    P = P[1:,:]
    return P, err

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    Base formula: x = K.M.X
    K = intrinsic
    M = extrinsic
    C = intrinsic*extrinsic = K*M
    '''
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    
    # Knowing E, we can estimate the Transformation (R,t) matrix
    # but we get 4 possbile transformation matrices (stacked along depth axis in M2)
    # (see two view lec2 of CVA slides for better understanding)
    M2 = camera2(E) # M2 = 3x4x4 matrix

    # Assuming the rotation and translation of camera1 is zero
    M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    # keep track of best M2 index
    best_M2_i = None
    err_min = 500
    best_pts_3d = None

    # we need to figure out which one of them is the right orientation (only 1 is right)

    # iterate over M1(fixed) and M2(4 possibilites) by passing them to triangulate
    for i in range(M2.shape[2]):
        M2_current = M2[:,:,i]

        # build the C1 and C2:
        pts_in_3d, err = triangulate(K1 @ M1, pts1, K2 @ M2_current, pts2)    
        if err < err_min and (np.where(pts_in_3d[:,2] == 0)[0].shape[0] == 0):
            print("satisfies the error criteria")
            err_min = err
            best_M2_i = i
            best_pts_3d = pts_in_3d

    if (best_M2_i is not None) and (best_pts_3d is not None):
        print("min err is", err_min)
        return M2[:,:,i], K2 @ M2[:,:,i], best_pts_3d
    else:
        print("could not converge to best M2")
        return 0,0,0




if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    print("saving references to disk")
    out_dir = "/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/outputs"
    check_and_create_directory(out_dir, create=1)
    np.savez_compressed(
                        os.path.join(out_dir, 'q3_3.npz'),
                        M2,
                        C2,
                        P
                        )

    # Tests
    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert(err < 500)
    if (err < 500):
        print("error within bounds")