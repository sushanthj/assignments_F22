from copy import deepcopy
import numpy as np
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
    parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
    parser.add_argument('--tolerance', type=float, default=0.03, help='binary threshold of intensity difference when computing the mask')
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold
    tolerance = args.tolerance
    seq = np.load('../data/aerialseq.npy')
    # seq = np.load('../data/carseq.npy')
    # seq = np.load('../data/antseq.npy')
    plot_idx = [30,60,90,120]

    It = seq[:,:,0]
    It1 = seq[:,:,1]
    # It = It/255.0
    # It1 = It1/255.0

    #M = LucasKanadeAffine(It, It1, threshold, num_iters)
    SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)

def disp_img(img, heading):
    img = np.array(img)
    window_name = heading
    
    # Displaying the image 
    cv2.imshow(window_name, img)
    cv2.waitKey()


def LucasKanadeAffine(It, It1, threshold, num_iters):
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

    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ################### TODO Implement Lucas Kanade Affine ###################
    # print("shape of first image is", It.shape)
    # print("shape of second image is", It1.shape)
    # disp_img(It, "zero image")
    # disp_img(It1, "random image")
    # print("received M is", M)
    # It = It/255.0
    # It1 = It1/255.0
    norm = 100

    MAX_ITERS = 1000
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
    # print("template crop shape is", template_img.shape)
    # disp_img(template_img, "template crop image")

    ################## Current Image #######################
    #? Generate spline inputs for the interpolant funciton
    spline_img_It1 = It1.T
    lower_left_It1 = np.array([0., 0.])
    upper_right_It1 = spline_img_It1.shape
    x_It1 = np.arange(lower_left_It1[0], upper_right_It1[0], 1)
    y_It1 = np.arange(lower_left_It1[1], upper_right_It1[1], 1)

    #? prev img shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It1 = RectBivariateSpline(x_It1, y_It1, spline_img_It1)

    #! ################ Refining the crop ###################
    while (norm > threshold and NUM_ITERS < MAX_ITERS):
        xIt1 = np.arange(lower_left_It1[0], upper_right_It1[0], 1)
        yIt1 = np.arange(lower_left_It1[1], upper_right_It1[1], 1)

        xxIt1, yyIt1 = np.meshgrid(xIt1,yIt1)

        current_img = interpolant_It1.ev(xxIt1, yyIt1)

        # update M
        M[0,0] = 1 + p0[0]
        M[1,0] = p0[1]
        M[0,1] = p0[2]
        M[1,1] = 1 + p0[3]
        M[0,2] = p0[4]
        M[1,2] = p0[5]
        print("updated M is \n", M)

        # USE THE AFFINE WARP 
        current_img = scipy.ndimage.affine_transform(current_img.T,M).T
        # print("Current image crop shape is", current_img.shape)
        disp_img(current_img, "current image crop")

        #? Finding common images
        current_img, template_img = find_common_images(template_img, current_img)
        
        # print("new template shape is", current_img.shape)
        disp_img(template_img, "updated template")
        
        # compute error image
        err_img = template_img - current_img

        # Find gradient of image
        dx_image_crop = interpolant_It1.ev(xxIt1, yyIt1, dx=1, dy=0)
        disp_img(dx_image_crop, "DX Image")
        dx_image_crop = scipy.ndimage.affine_transform(dx_image_crop.T,M).T
        dy_image_crop = interpolant_It1.ev(xxIt1, yyIt1, dx=0, dy=1)
        disp_img(dy_image_crop, "DY Image")
        dy_image_crop = scipy.ndimage.affine_transform(dy_image_crop.T,M).T

        xdx, ydy = dx_image_crop.shape
        dx_flat = dx_image_crop.reshape((xdx*ydy,1))
        dy_flat = dy_image_crop.reshape((xdx*ydy,1))

        # combine gradients of curr image into shape nx2 (n = total no. of pixels)
        # grad_current_image = np.hstack((dx_flat, dy_flat))
        # print("shape of grad image is", grad_current_image.shape)

        ##### building the jacobian #####
        num_rows, num_columns = dx_image_crop.shape
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

        print("dx_x ARRAY IS", dx_x[200:210,:])
        print("dy_x ARRAY IS", dy_x[200:210,:])
        # print("dx_y ARRAY IS", dx_y[0:10,:])
        # print("dy_y ARRAY IS", dy_y[0:10,:])
        
        stp_des_img = np.hstack((dx_x, dy_x, dx_y, dy_y, dx_flat, dy_flat))
        print("stp_des is", stp_des_img[0:10,:])

        # convert the error image to nx1 shape to multiply with stp_des_img.T
        xer, yer = err_img.shape
        err_img = err_img.reshape((xer*yer,1))

        # compute the update of shape (6 x n) * (n x 1) = (6 x 1)
        update = np.matmul(np.transpose(stp_des_img), err_img)
        # print("shape of update is", update.shape)

        # compute hessian of shape (6 x n) * (n x 6) = (6 x 6)
        # print("shape of stp_des_transp IS !!", np.transpose(stp_des_img).shape)
        hess = np.matmul(np.transpose(stp_des_img),stp_des_img)
        # print("the shape of the hess is \n", hess.shape)
        hess_inv = np.linalg.inv(hess)

        # delta_p must be of shape (6 x 6) * (6 x 1) = (6 * 1)
        delta_p = np.matmul(hess_inv, update)
        delta_p = np.squeeze(delta_p)
        # print("delta p was :", delta_p)
        ############################################################################################################
        
        n = np.linalg.norm((delta_p))
        print("NORM IS", n)
        norm = n
        p0 += delta_p
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

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    ################### TODO Implement Substract Dominent Motion ###################
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = np.vstack((M,np.array([0.0, 0.0, 1.0], dtype=M.dtype)))
    # disp_img(image1, "input image")
    transfomrmed_img1 = scipy.ndimage.affine_transform(image1.T,np.linalg.inv(M)).T
    disp_img(transfomrmed_img1, "transformed image")
    disp_img(image2, "It1 image")
    diff_img = image2 - transfomrmed_img1
    out_1, out_2 = np.where(diff_img > tolerance)
    
    # build a zero image with same shape as image1
    # x,y = image1.shape
    # zero_img = np.zeros((x,y))

    for i in range(len(out_1)):
        x_coord = out_1[i]
        y_coord = out_2[i]
        mask[x_coord, y_coord] = 0
    disp_img(mask.astype(np.float32), "mask image pre opening")
    # mask = binary_dilation(mask)
    mask = binary_erosion(mask)
    disp_img(mask.astype(np.float32), "mask image AFTER opening")
    
    return mask.astype(bool)

if __name__ == '__main__':
    main()