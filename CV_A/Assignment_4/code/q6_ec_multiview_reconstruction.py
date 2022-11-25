import numpy as np
import matplotlib.pyplot as plt

import os

from scipy.optimize import minimize
from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
# from q3_2_triangulate import triangulate
from q2_1_eightpoint import check_and_create_directory

# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 200):
    vis_pts_1 = np.where(pts1[:,2] > Thres)
    vis_pts_2 = np.where(pts2[:,2] > Thres)
    vis_pts_3 = np.where(pts3[:,2] > Thres)
    
    # create a dummy vector to save the 3D points for each corresp 2D pt
    pts_3d = np.zeros(pts1.shape)
    reproj_error = np.zeros(12)

    overlap_all = np.intersect1d(vis_pts_1, vis_pts_2, vis_pts_3)
    for i in overlap_all:
        pts_cam_1_2, err1 = triangulate(C1, pts1[i,:-1], C2, pts2[i,:-1])
        pts_cam_2_3, err2 = triangulate(C2, pts2[i,:-1], C3, pts3[i,:-1])
        pts_cam_1_3, err3 = triangulate(C1, pts1[i,:-1], C3, pts3[i,:-1])

        avg_pt_i = (pts_cam_1_2 + pts_cam_2_3 + pts_cam_1_3)/3
        avg_err = (err1+err2+err3)/3
        pts_3d[i,:] = avg_pt_i
        reproj_error[i] = avg_err
    
    for i in vis_pts_1[0]:
        # print("i is", i)
        if i not in overlap_all:
            # print("computing", i)
            if i in vis_pts_2[0]:
                pts_i, err = triangulate(C1, pts1[i,:-1], C2, pts2[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            elif i in vis_pts_3[0]:
                pts_i, err = triangulate(C1, pts1[i,:-1], C3, pts3[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            else:
                print("point not visible in 2 views")

    for i in vis_pts_2[0]:
        # print("i is", i)
        if i not in overlap_all:
            # print("computing", i)
            if i in vis_pts_3[0]:
                pts_i, err = triangulate(C2, pts2[i,:-1], C3, pts3[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            elif i in vis_pts_1[0]:
                pts_i, err = triangulate(C1, pts1[i,:-1], C2, pts2[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            else:
                print("point not visible in 2 views")

    for i in vis_pts_3[0]:
        if i not in overlap_all:
            # print("computing", i)
            if i in vis_pts_1[0]:
                pts_i, err = triangulate(C1, pts1[i,:-1], C3, pts3[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            elif i in vis_pts_2[0]:
                pts_i, err = triangulate(C1, pts1[i,:-1], C2, pts2[i,:-1])
                pts_3d[i,:] = pts_i
                reproj_error[i] = err
            else:
                print("point not visible in 2 views")
    
    print("pts1 shape is", pts1.shape)
    print("3d points shape is", pts_3d.shape)

    return pts_3d, reproj_error, [vis_pts_1, vis_pts_2, vis_pts_3]

def MutliviewReconstructionError(x, C1, pts1, C2, pts2, C3, pts3, vis_pts_list):
    # decompose x
    P_init = x
    P_shape_req = int(P_init.shape[0]/3)
    P_init = np.reshape(P_init, newshape=(P_shape_req,3))
    
    vis_pts_1 = vis_pts_list[0]
    vis_pts_2 = vis_pts_list[1]
    vis_pts_3 = vis_pts_list[2]

    pts1 = pts1[:,0:2]
    pts2 = pts2[:,0:2]
    pts3 = pts3[:,0:2]

    # list to store error values
    err_list = []

    # build a sub_P matrix for all visible points in pts1, pts2, pts3
    sub_pts1 = np.take(pts1, vis_pts_1, axis=0)[0]
    sub_P1 = np.take(P_init, vis_pts_1, axis=0)[0]
    sub_pts2 = np.take(pts2, vis_pts_2, axis=0)[0]
    sub_P2 = np.take(P_init, vis_pts_2, axis=0)[0]
    sub_pts3 = np.take(pts3, vis_pts_3, axis=0)[0]
    sub_P3 = np.take(P_init, vis_pts_3, axis=0)[0]
    
    P_list = [sub_P1, sub_P2, sub_P3]
    pts_list = [sub_pts1, sub_pts2, sub_pts3]
    C_list = [C1, C2, C3]

    for i in range(len(P_list)):
        P = P_list[i]
        p= pts_list[i]
        C = C_list[i]
        
        # homogenize P to contain a 1 in the end (P = Nx3 vector)
        P_homogenous = np.append(P, np.ones((P.shape[0],1)), axis=1)
        
        # Find the projection of P1 onto image 1 (vectorize)
        # Transpose P_homogenous to make it a 4xN vector and left mulitply with C1
        #  3xN =  3x4 @ 4XN
        p_hat = (C @ P_homogenous.T)
        # normalize and transpose to get back to format of p1
        p_hat = ((p_hat/p_hat[2,:])[0:2,:]).T

        error = np.linalg.norm((p-p_hat).reshape([-1]))**2
        err_list.append(error)
    
    err_total = err_list[0] + err_list[1] + err_list[2]
    # print("error overall is", err_total)

    return err_total
    
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


    # get the camera 1 matrix
    p1_1 = C1[0,:]
    p2_1 = C1[1,:]
    p3_1 = C1[2,:]

    # get the camera 2 matrix
    p1_2 = C2[0,:]
    p2_2 = C2[1,:]
    p3_2 = C2[2,:]

    x, y = pts1[0], pts1[1]
    x2, y2 = pts2[0], pts2[1]

    # calculate the A matrix for this point correspondence
    A = np.array([y*p3_1 - p2_1 , p1_1 - x*p3_1 , y2*p3_2 - p2_2 , p1_2 - x2*p3_2])
    u, s, v = np.linalg.svd(A)

    # here the linalg.svd gives v_transpose
    # but we need just V therefore we again transpose
    X = v.T[:,-1]
    # print("X is", X)
    X = X.T
    X = np.expand_dims(X,axis=0)
    # print("X after transpose and expand is", X)
    
    # convert X to homogenous coords
    X = X/X[0,3]
    # print("X after normalizing is", X)

    P = np.concatenate((P, X[:,0:3]), axis=0)
    
    X = X.T

    # find the error for this projection
    # 3x1 = 3x4 . 3x1 
    pt_1 = ((C1 @ X)/(C1 @ X)[2,0])[0:2,0]
    pt_2 = ((C2 @ X)/(C2 @ X)[2,0])[0:2,0]

    # calculate the reporjection error
    err += np.linalg.norm(pt_1 - pts1)**2 + np.linalg.norm(pt_2 - pts2)**2

    # print("error in this iteration is", err)
    P = P[1:,:]
    return P[0], err

'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    fig = plt.figure()
    # num_points = pts_3d.shape[0]
    ax = fig.add_subplot(111, projection='3d')

    vid_len = len(pts_3d_video)
    vals = np.linspace(0.1,1, num=vid_len, endpoint=False)

    for i in range(len(pts_3d_video)):
        pts_3d = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0,0], pts_3d[index1,0]]
            yline = [pts_3d[index0,1], pts_3d[index1,1]]
            zline = [pts_3d[index0,2], pts_3d[index1,2]]
            ax.plot(xline, yline, zline, color=colors[j], alpha=vals[i])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


#Extra Credit
if __name__ == "__main__":
         
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        #Note - Press 'Escape' key to exit img preview and loop further 
        # img = visualize_keypoints(im2, pts2)

        C1 = K1 @ M1
        C2 = K2 @ M2
        C3 = K3 @ M3
        pts_3d, err, vis_pts_list = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3)
        x_start = pts_3d.flatten()
        
        x_optimized_obj = minimize(MutliviewReconstructionError, x_start, args=(C1, pts1, C2, pts2, C3, pts3, vis_pts_list), method='Powell')
        print("x_end shape is", x_optimized_obj.x.shape)
        x_optimized = x_optimized_obj.x

        P_final = x_optimized
        P_shape_req = int(P_final.shape[0]/3)
        P_final = np.reshape(P_final, newshape=(P_shape_req,3))
        plot_3d_keypoint(P_final)
        pts_3d_video.append(P_final)
        visualize_keypoints(im1, pts1, Threshold=200)
    
    plot_3d_keypoint_video(pts_3d_video)
    out_dir = "/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/outputs"
    check_and_create_directory(out_dir, create=1)
    np.savez_compressed(
                        os.path.join(out_dir, 'q6_1.npz'),
                        P_final)
