import numpy as np
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
import cv2
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
    parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold

    seq = np.load("../data/carseq.npy")
    ##### Run LK Tracking
    rect = np.array([59, 116, 145, 151]).T

    lk_res = []
    lk_res.append(rect)
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    It = seq[:,:,0]
    It1 = seq[:,:,25]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    # rect = np.concatenate((pt_topleft + p, pt_bottomright + p))
    # lk_res.append(rect)

def disp_img(img, rect):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    window_name = 'Image'
  
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
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    
    # Displaying the image 
    cv2.imshow(window_name, image)
    cv2.waitKey()


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold

    It = It/255.0
    It1 = It1/255.0

    spline_img = It.T
    lower_left = np.array([0., 0.])
    upper_right = spline_img.shape
    # print("orig image shpae is", It.shape) # ans = 240,320
    print("spline image shape is", upper_right) # ans = 320,240

    # Construct a spline interpolant to use as a target
    x = np.arange(lower_left[0], upper_right[0], 1)
    print("x shape is", x.shape)
    y = np.arange(lower_left[1], upper_right[1], 1)
    print("y shape is", y.shape)

    # prev shape = 240, 320 -- viz 240 rows and 320 columns -- viz y_len = 240 and x_len = 320
    interpolant_It = RectBivariateSpline(x, y, spline_img)

    # evaluate the template
    rect2 = np.array([59.65, 116.65, 145.65, 151.65])
    rect = np.array([59, 116, 145, 151])
    x = np.arange(rect2[1],rect2[3],1)
    y = np.arange(rect2[0],rect2[2],1)

    xx, yy = np.meshgrid(x,y)

    template_crop = interpolant_It.ev(yy,xx).T
    print("template crop shape is", template_crop.shape)
    

    template = It[rect[1]:rect[3], rect[0]:rect[2]]
    It1_warped = scipy.ndimage.shift(It, np.array([100.0, 200.0]))
    new_warped = It1_warped[rect[1]:rect[3], rect[0]:rect[2]]


    plt.imsave("./tracking/orig.jpeg", It)
    plt.imsave("./tracking/template.jpeg", template)
    plt.imsave("./tracking/bivar_template.jpeg", template_crop)
    #plt.imsave("./tracking/next.jpeg", It1_warped)
    time.sleep(1)
    disp_img("./tracking/orig.jpeg", rect)
    disp_img("./tracking/template.jpeg", rect)
    disp_img("./tracking/bivar_template.jpeg", rect)
    #disp_img("./tracking/next.jpeg", rect)

    ################### TODO Implement Lucas Kanade ###################
    
    # warp the It1 image according to initial movement vector [dp_x0, dp_y0]
    It1_warped = scipy.ndimage.shift(It1, p0)

    # compute error image
    err_img = template - new_warped

    # Find gradient of image
    grads = np.gradient(It1)
    plt.imsave("./tracking/grad1.jpeg", grads[0])
    plt.imsave("./tracking/grad2.jpeg", grads[1])
    time.sleep(1)
    disp_img("./tracking/grad1.jpeg", rect)
    disp_img("./tracking/grad2.jpeg", rect)
    warped_grads = []
    warped_grads.append(scipy.ndimage.shift(grads[0], p0))
    warped_grads.append(scipy.ndimage.shift(grads[1], p0))

    # define the jacobian of the warp
    jacobian = np.array(([1,0],[0,1]),dtype=np.float32)

    # compute steepest descent images (should be a 2x1)
    stp_des_img = []

    # each row of warped_grads gets multiplied with the columns of jacobian
    for i in range(len(warped_grads)):
        # iterate across each element (column-wise) of jacobian
        current_column = jacobian[:,0]
        for j in range(len(current_column.shape)):
            temp_container = 0.0
            temp_container += current_column[j]*warped_grads[j]
        
        stp_des_img.append(temp_container)
    
    # compute the hessian
    hess = [[],[]]
    
    # iterate over the first row of our to-be hessian
    for i in range(len(stp_des_img)):
        ref_elem = stp_des_img[i]

        # iterate over the no. of elements in stp_des (2nd matrix in the hessian formula)
        for j in range(len(stp_des_img)):
            hess[i].append(ref_elem*stp_des_img[j])

    ##################################### Alternative approach ##################################################
    stp_des_img_2 = np.array([])
    for i in range(len(stp_des_img)):
        if i == 0:
            stp_des_img_2 = stp_des_img[i]
        else:
            stp_des_img_2 = np.hstack((stp_des_img_2, stp_des_img[i]))

    hess_2 = np.matmul(np.transpose(stp_des_img_2),stp_des_img_2)
    hess_2_inv = np.linalg.pinv(hess_2)
    hess_2_sum = np.sum(hess_2)
    ############################################################################################################

    update = np.matmul(np.transpose(stp_des_img_2),err_img)
    delta_p = np.matmul(hess_2_inv,update)

    x,y = delta_p.shape
    delta_p1 = delta_p[0:int(x/2),:]
    delta_p2 = delta_p[int(x/2):,:]
    delta_p_sum = np.array([np.sum(delta_p1), np.sum(delta_p2)])

    print("image shape is", template.shape)
    print("delta_p is", delta_p_sum)
    return delta_p_sum

if __name__ == '__main__':
    main()