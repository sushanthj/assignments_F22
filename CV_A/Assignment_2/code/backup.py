from copy import deepcopy
import numpy as np
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import time


def disp_img(img, rect):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window_name = 'Image'
  
    if rect.all() != 0:
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (rect[0], rect[1])
        
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (rect[2], rect[3])
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    
    # Displaying the image 
    cv2.imshow(window_name, img)
    cv2.waitKey()


def LucasKanade(It, It1, rect, threshold, num_iters, p0, rect_0=np.array([0.0,0.0,0.0,0.0])):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    print("received p0 is", p0)
    It = It/255.0
    It1 = It1/255.0
    norm = 100

    MAX_ITERS = 100
    NUM_ITERS = 0

    ################## Template Image #####################
    # extract the template from the It image
    # Generate spline inputs for theinterpolant funciton
    spline_img_It = It.T
    lower_left_It = np.array([0., 0.])
    upper_right_It = spline_img_It.shape
    x_It = np.arange(lower_left_It[0], upper_right_It[0], 1)
    y_It = np.arange(lower_left_It[1], upper_right_It[1], 1)

    # prev img shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It = RectBivariateSpline(x_It, y_It, spline_img_It)
    x_It_crop = np.arange(rect[1],rect[3],1)
    y_It_crop = np.arange(rect[0],rect[2],1)

    xx_It, yy_It = np.meshgrid(x_It_crop, y_It_crop)

    template_crop = interpolant_It.ev(yy_It,xx_It).T
    # print("template crop shape is", template_crop.shape)

    ################## Current Image #######################
    # Generate spline inputs for theinterpolant funciton
    spline_img_It1 = It1.T
    lower_left_It1 = np.array([0., 0.])
    upper_right_It1 = spline_img_It1.shape
    x_It1 = np.arange(lower_left_It1[0], upper_right_It1[0], 1)
    y_It1 = np.arange(lower_left_It1[1], upper_right_It1[1], 1)

    # prev img shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It1 = RectBivariateSpline(x_It1, y_It1, spline_img_It1)

    # Only for the template correction case (handling for track2)
    if rect_0.all() == 0:
        FLAG = 0
    else:
        FLAG = 1
        print("running track2")
        rect2 = rect_0
        print("rect2 is ", rect2)
        rect_cut = np.ceil(rect2)
        rect_remainder = rect2 - rect_cut

    ################# Refining the crop ###################
    while (norm > threshold and NUM_ITERS < MAX_ITERS):

        if FLAG == 1:
            # warp (translate) the It1 image crop according to initial movement vector [dp_x0, dp_y0]
            x_It1_crop = np.arange(rect_cut[1],rect_cut[3],1) + (p0[1] + rect_remainder[1])
            y_It1_crop = np.arange(rect_cut[0],rect_cut[2],1) + (p0[0] + rect_remainder[0])
        else:
            x_It1_crop = np.arange(rect[1],rect[3],1) + p0[1]
            y_It1_crop = np.arange(rect[0],rect[2],1) + p0[0]

        xx_It1, yy_It1 = np.meshgrid(x_It1_crop, y_It1_crop)

        curr_image_crop = interpolant_It1.ev(yy_It1, xx_It1).T
        # print("current image crop shape is", curr_image_crop.shape)
        
        # compute error image
        err_img = template_crop - curr_image_crop

        # Find gradient of image
        dx_image_crop = interpolant_It1.ev(yy_It1, xx_It1, dx=1, dy=0).T
        dy_image_crop = interpolant_It1.ev(yy_It1, xx_It1, dx=0, dy=1).T

        xdx, ydy = dx_image_crop.shape
        dx_flat = dx_image_crop.reshape((xdx*ydy,1))
        dy_flat = dy_image_crop.reshape((xdx*ydy,1))

        # combine gradients of curr image into shape nx2 (n = total no. of pixels)
        grad_current_image = np.hstack((dx_flat, dy_flat))
        # print("shape of grad image is", grad_current_image.shape)

        # define the jacobian of the warp of shape 2x2
        jacobian = np.array(([1,0],[0,1]),dtype=np.float32)

        # compute steepest descent images (should be a nx2)
        stp_des_img = np.matmul(grad_current_image, jacobian)
        # print("shape of stp_des is", stp_des_img.shape)

        # convert the error image to nx1 shape to multiply with stp_des_img.T
        xer, yer = err_img.shape
        err_img = err_img.reshape((xer*yer,1))

        # compute the update of shape (2 x n) * (n x 1) = (2 x 1)
        update = np.matmul(np.transpose(stp_des_img), err_img)
        # print("shape of update is", update.shape)

        # compute hessian of shape (2 x n) * (n x 2) = (2 x 2)
        hess = np.matmul(np.transpose(stp_des_img),stp_des_img)
        hess_inv = np.linalg.inv(hess)
        # try:
        #     hess_inv = np.linalg.inv(hess)
        # except np.linalg.LinAlgError as e:
        #     print("hess shape was", hess.shape)
        #     hess_inv = np.linalg.pinv(hess)

        # delta_p must be of shape (2 x 2) * (2 x 1) = (2 * 1)
        delta_p = np.matmul(hess_inv, update)
        delta_p = np.squeeze(delta_p)
        ############################################################################################################
        
        n = np.linalg.norm((delta_p))
        # print("norm is", n)
        norm = n
        p0 += delta_p
        # print("no. of iterations is", NUM_ITERS)
        NUM_ITERS += 1

    print("returning p0 as", p0)
    return p0


# It0 is the orig image
# It1 is the nth image
# template image, may be the n-1th image of It1, but can't say surely
def strat3(It0, It1, rect, template_rect, 
            threshold, template_threshold, template_img, 
            num_iters, img_index):

    # For the first image, we can't use any strategy
    if img_index == 1:
        p_n_star = LucasKanade(It0, It1, rect, threshold, num_iters, np.array([0.0,0.0]))
        p_n_final = p_n_star
    else:
        print("ENTERING THE PLUS 1 CASE")
        # Compute the Track1 (see Fig 3)
        print("given rect for p_n is", rect)
        p_n = LucasKanade(template_img, It1, rect, threshold, num_iters, np.array([0.0,0.0]))

        # to prevent data merging
        p_n_for_track_2 = deepcopy(p_n)

        # Track 2
        # use p_n and add it to template_rect in LK
        p_n_star = LucasKanade(It0, It1, template_rect, threshold, num_iters, p_n_for_track_2, rect)
        
        # if the strategy 3 equation is satisfied
        print("norm is", np.linalg.norm((p_n_star-p_n)))
        if np.linalg.norm((p_n_star-p_n)) < template_threshold:
            p_n_final = p_n_star
        
        # if start3 equation is not satisfied
        else:
            print("EQ 3 NOT SATISFIED")
            p_n_final = p_n
    
    return p_n_final