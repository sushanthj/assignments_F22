import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
from copy import deepcopy
import cv2

def disp_img(img, heading):
    img = np.array(img)
    window_name = heading
    
    # Displaying the image 
    cv2.imshow(window_name, img)
    cv2.waitKey()


def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = np.vstack((M,np.array([0.0, 0.0, 1.0], dtype=M.dtype)))
    M_temp = deepcopy(M)

    delta_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ################### TODO Implement Lucas Kanade Affine ###################
    norm = 100

    MAX_ITERS = 50
    NUM_ITERS = 0

    ################## Template Image #####################
    # extract the template from the It image
    #? Generate spline inputs for theinterpolant funciton
    spline_img_It = It.T
    lower_left_It = np.array([0., 0.])
    upper_right_It = spline_img_It.shape
    x_It = np.arange(lower_left_It[0], upper_right_It[0], 1)
    y_It = np.arange(lower_left_It[1], upper_right_It[1], 1)

    #? prev img shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It = RectBivariateSpline(x_It, y_It, spline_img_It)
    
    x = np.arange(lower_left_It[0], upper_right_It[0], 1)
    y = np.arange(lower_left_It[1], upper_right_It[1], 1)

    xx, yy = np.meshgrid(x,y)

    template_img = interpolant_It.ev(xx,yy)

    # Find gradient of image
    dx_template_crop = interpolant_It.ev(xx, yy, dx=1, dy=0)
    dx_template_crop = affine_transform(dx_template_crop.T,M).T
    dy_template_crop = interpolant_It.ev(xx, yy, dx=0, dy=1)
    dy_template_crop = affine_transform(dy_template_crop.T,M).T

    xdx, ydy = dx_template_crop.shape
    dx_flat = dx_template_crop.reshape((xdx*ydy,1))
    dy_flat = dy_template_crop.reshape((xdx*ydy,1))

    ################## Current Image #######################
    #? Generate spline inputs for theinterpolant funciton
    spline_img_It1 = It1.T
    lower_left_It1 = np.array([0., 0.])
    upper_right_It1 = spline_img_It1.shape
    x_It1 = np.arange(lower_left_It1[0], upper_right_It1[0], 1)
    y_It1 = np.arange(lower_left_It1[1], upper_right_It1[1], 1)

    #? prev img shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It1 = RectBivariateSpline(x_It1, y_It1, spline_img_It1)

    ######## Build the jacobian and do the chunk of precompute ################
    num_rows, num_columns = dx_template_crop.shape
    # num_columns, num_rows = dx_image_crop.shape
    y_count = list(range(0,num_columns,1))*num_rows
    y_count_arr = np.asarray(y_count)
    len_y = y_count_arr.shape
    y_count_arr = y_count_arr.reshape(len_y[0],1)

    x_count = [0]*num_columns
    for i in range(1,num_rows):
        temp_list = [i]*num_columns
        x_count.extend(temp_list)
    x_count_arr = np.asarray(x_count)
    len_x = x_count_arr.shape
    x_count_arr = x_count_arr.reshape(len_x[0],1)

    # swap the x_count and y_counts
    swap_arr = y_count_arr
    y_count_arr = x_count_arr
    x_count_arr = swap_arr

    # compute steepest descent images (should be a nx6)
    dx_x = dx_flat*x_count_arr
    dy_x = dy_flat*x_count_arr
    dx_y = dx_flat*y_count_arr
    dy_y = dy_flat*y_count_arr
    
    stp_des_img = np.hstack((dx_x, dy_x, dx_y, dy_y, dx_flat, dy_flat))

    # compute hessian of shape (6 x n) * (n x 6) = (6 x 6)
    hess = np.matmul(np.transpose(stp_des_img),stp_des_img)
    hess_inv = np.linalg.inv(hess)

    ################# Refining the crop ###################
    while (norm > threshold and NUM_ITERS < MAX_ITERS):
        xIt1 = np.arange(lower_left_It1[0], upper_right_It1[0], 1)
        yIt1 = np.arange(lower_left_It1[1], upper_right_It1[1], 1)

        xxIt1, yyIt1 = np.meshgrid(xIt1,yIt1)

        current_img = interpolant_It1.ev(xxIt1, yyIt1)

        # update M
        M_temp[0,0] = 1 + delta_p[0]
        M_temp[1,0] = delta_p[1]
        M_temp[0,1] = delta_p[2]
        M_temp[1,1] = 1 + delta_p[3]
        M_temp[0,2] = delta_p[4]
        M_temp[1,2] = delta_p[5]
        M_temp[2,2] = 1

        M = np.matmul(M, np.linalg.inv(M_temp))
        # print("updated M is \n", M)

        # USE THE AFFINE WARP 
        current_img = affine_transform(current_img.T,M).T

        #? Finding common images
        current_img, template_img = find_common_images(template_img, current_img)
        
        # compute error image
        err_img = current_img - template_img

        # convert the error image to nx1 shape to multiply with stp_des_img.T
        xer, yer = err_img.shape
        err_img = err_img.reshape((xer*yer,1))

        # compute the update of shape (6 x n) * (n x 1) = (6 x 1)
        update = np.matmul(np.transpose(stp_des_img), err_img)

        # delta_p must be of shape (6 x 6) * (6 x 1) = (6 * 1)
        delta_p = np.matmul(hess_inv, update)
        delta_p = np.squeeze(delta_p)
        
        n = np.linalg.norm((delta_p))
        norm = n
        NUM_ITERS += 1

    # M[0,0] = 1 + p0[0]
    # M[1,0] = p0[1]
    # M[0,1] = p0[2]
    # M[1,1] = 1 + p0[3]
    # M[0,2] = p0[4]
    # M[1,2] = p0[5]
    print("no. of iterations per frame", NUM_ITERS)
    M_dash = M[0:2,:]
    return M_dash

def find_common_images(template, It):
    out_1, out_2 = np.where(It==0)

    x,y = template.shape
    img2_mod = deepcopy(template)

    for i in range(len(out_1)):
        x_coord = out_1[i]
        y_coord = out_2[i]
        img2_mod[x_coord, y_coord] = 0.0
    
    return It, img2_mod
