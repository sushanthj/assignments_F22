import numpy as np
import cv2
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

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
    mask = np.zeros(image1.shape, dtype=bool)
    ################### TODO Implement Substract Dominent Motion ###################
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = np.vstack((M,np.array([0.0, 0.0, 1.0], dtype=M.dtype)))
    # disp_img(image1, "input image")
    transfomrmed_img1 = affine_transform(image1.T,np.linalg.inv(M)).T
    # disp_img(transfomrmed_img1, "transformed image")
    # disp_img(image2, "It1 image")
    diff_img = image2 - transfomrmed_img1
    out_1, out_2 = np.where(diff_img > tolerance)

    for i in range(len(out_1)):
        x_coord = out_1[i]
        y_coord = out_2[i]
        mask[x_coord, y_coord] = 1
    # disp_img(mask.astype(np.float32), "mask image pre opening")
    # mask = binary_erosion(mask)
    mask = binary_dilation(mask, iterations=4)
    mask = binary_erosion(mask)
    # disp_img(mask.astype(np.float32), "mask image AFTER opening")

    # Ant sequence works best with 3 dilations, 1 erosion and tol = 0.07
    # Car sequence works best with 1 erosion, 3 dilations and tol = 0.02
    
    return mask.astype(bool)

def disp_img(img, heading):
    img = np.array(img)
    window_name = heading
    
    # Displaying the image 
    cv2.imshow(window_name, img)
    cv2.waitKey()
