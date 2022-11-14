import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint, check_and_create_directory
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    print("shape of temple_pts1 is", temple_pts1.shape)
    # define a placeholder for all pts2
    pts_im2 = []
    
    # given pts1 find the epipolar correspondences in im2

    # iterate along the number of rows of temple_pts1
    for i in range(temple_pts1.shape[0]):
        pts1 = temple_pts1[i,:]
        x1, y1 = pts1[0], pts1[1]
        # print("pts1 for this iteration is", x1,y1)

        # use the epipolar corresp. to find the pts2
        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
        pts_im2.append(np.array([x2, y2]))
    
    temple_pts2 = np.stack(pts_im2, axis=0)
    print("shape of temple_pts2 is", temple_pts2.shape)

    # having found the correspondences find the correct camera matrix relating the two views
    M2, C2, P, M1, C1 = findM2(F, temple_pts1, temple_pts2, intrinsics)
    return M2, C2, P, M1, C1

def display_3d(P):
    """
    Using the 3D points in P make a scatter plot
    Args:
        P : Nx3 vector containing the 3D points
    """
    # Creating figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Creating plot
    for i in range(P.shape[0]):
        # print("the xyz coords are", P[i,:])
        x,y,z = P[i,0], P[i,1], P[i,2]
        ax.scatter3D(x, y, z, color = "blue")
    
    plt.title("temple 3D point plot")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    # show plot
    plt.show()

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # Find F using point correspondences
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print("F is", F)

    # Assuming we don't have the corresponding points in im2, use epipolarcorrespondences to
    # calculate the respective pts2
    # Having pts1 and pts2 use the triangulate function to find the 3D location of the points
    M2, C2, P, M1, C1 = compute3D_pts(pts1, intrinsics, F, im1, im2)

    out_dir = "/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/outputs"
    check_and_create_directory(out_dir, create=1)
    np.savez_compressed(
                        os.path.join(out_dir, 'q4_2.npz'),
                        F,
                        M1,
                        M2,
                        C1,
                        C2
                        )
    display_3d(P)