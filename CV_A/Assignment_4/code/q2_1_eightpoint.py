import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    """
    Compute the normalized coordinates
    and also the fundamental matrix using computeH

    Args:
        x1 (Mx2): the matched locations of corners in img1
        x2 (Mx2): the matched locations of corners in img2

    Returns:
        F2to1: Fundamental matrix after denormalization
    """
    print("M scale is", M)
    # Compute the centroid of the points
    x1, x2 = pts1, pts2

    # Doing the M normaliazation
    moved_scaled_x1 = x1/M
    moved_scaled_x2 = x2/M

    t = np.diag([1/M, 1/M, 1])

    # Compute Fundamental Matrix
    F = computeF(moved_scaled_x1, moved_scaled_x2)

    # Refine and then enforce singularity constraint
    F = refineF(F, moved_scaled_x1, moved_scaled_x2)

    # Denormalization
    F2to1 = np.matmul(t.T, (F @ t))
    F2to1 = F2to1/F2to1[2,2]

    return F2to1


def computeF(x1, x2):
    """
    Computes the fundamental based on 
    matching points in both images

    Args:
        x1: keypoints in image 1
        x2: keypoints in image 2

    Returns:
        H2to1: the fundamental matrix
    """

    # Define a dummy H matrix
    A_build = []
    
    # Define the A matrix for (Ah = 0) (A matrix size = N*2 x 9)
    for i in range(x1.shape[0]):
        row_1 = np.array([ x2[i,0]*x1[i,0], x2[i,0]*x1[i,1], x2[i,0], x2[i,1]*x1[i,0], x2[i,1]*x1[i,1], x2[i,1], x1[i,0], x1[i,1], 1])
        A_build.append(row_1)
    
    A = np.stack(A_build, axis=0)

    # Do the least squares minimization to get the homography matrix
    # this is done as eigenvector coresponding to smallest eigen value of A`A = H matrix
    u, s, v = np.linalg.svd(A)

    # here the linalg.svd gives v_transpose
    # but we need just V therefore we again transpose
    F2to1 = np.reshape(v.T[:,-1], (3,3))
    return F2to1


def check_and_create_directory(dir_path, create):
    """
    Checks for existing directories and creates if unavailable

    [input]
    * dir_path : path to be checked
    * create   : tag to specify only checking path or also creating path
    """
    if create == 1:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    else:
        if not os.path.exists(dir_path):
            warnings.warn(f'following path could not be found: {dir_path}')



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print("the fundamental matrix found was \n", F)

    # Q2.1
    out_dir = "/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/outputs"
    check_and_create_directory(out_dir, create=1)
    np.savez_compressed(
                        os.path.join(out_dir, 'q2_1.npz'),
                        F,
                        np.max([*im1.shape, *im2.shape])
                        )

    displayEpipolarF(im1, im2, F)
    check_and_create_directory(out_dir, 1)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)