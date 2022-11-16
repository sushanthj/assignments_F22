import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF
from q2_1_eightpoint import check_and_create_directory

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    
    x1, x2 = pts1, pts2
    # Doing the M normaliazation
    moved_scaled_x1 = x1/M
    moved_scaled_x2 = x2/M
    t = np.diag([1/M, 1/M, 1])
    F1, F2 = compute_F_mult(moved_scaled_x1, moved_scaled_x2)
    print("F1 is \n", F1)
    print("F2 is \n", F2)
    F_mat = [F1, F2]
    
    # TODO: FIND THE COEFFS
    # create a function to map the polynomial to solve: [det(a*F1 + (1-a)*F2] = [ alpha*(a**3) + beta*(a**2) + gamma(a) + delta ]
    coeffs = lambda a: np.linalg.det(a*F1 + (a-1)*F2)

    delta = coeffs(0)
    beta = (coeffs(1) + coeffs(-1))/2 - coeffs(0)
    gamma = 2*(coeffs(1) - coeffs(-1))/3 - (coeffs(2) - coeffs(-2))/12
    alpha = (coeffs(1) + coeffs(-1))/2 - gamma

    roots = np.roots([alpha, beta, gamma, delta])
    complex_root_locs = np.invert(np.iscomplex(roots))
    roots_pruned = roots[complex_root_locs]
    
    for m in range(roots_pruned.shape[0]):
        a = roots_pruned[m]
        F_tmp = a*F1 + ((1-a)*F2)

        # F_tmp = refineF(F_tmp, moved_scaled_x1, moved_scaled_x2)
        F_tmp = _singularize(F_tmp)
        F_tmp = refineF(F_tmp, moved_scaled_x1, moved_scaled_x2)

        F_tmp = np.matmul(t.T, (F_tmp @ t))
        F_tmp = F_tmp/F_tmp[2,2]

        print("rank is", np.linalg.matrix_rank(F_tmp))

        Farray.append(F_tmp)

    return Farray



def compute_F_mult(x1, x2):
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
    F1 = np.reshape(v.T[:,-1], (3,3))
    F2 = np.reshape(v.T[:,-2], (3,3))

    return F1, F2

if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    out_dir = "/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/outputs"
    check_and_create_directory(out_dir, create=1)
    np.savez_compressed(
                        os.path.join(out_dir, 'q2_2.npz'),
                        F,
                        M,
                        pts1,
                        pts2
                        )

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        # Fs is the Farray
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)