import math
import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
WINDOW_SIZE = 3

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    """
    Find the epipolar correspondeces depending 
    Args:
        im1 : input image
        im2 : output image
        F   : Fundamental Matrix
        x1  : x_coords of keypoints found in im1
        y1  : y_coords of keypoints found in im1
    
    Returns:
        x2  : calculated correspondence point in im2
        y2  : calculated correspondence point in im2
    """
    
    # for a given x,y location in image 1, find the resulting line in second image
    sy, sx, _ = im2.shape

    # convert input point into homogenous coords
    v = np.array([x1, y1, 1])

    # transform the input point with fundamental matrix to get a line in output image
    l = F.dot(v)

    # this is norm of output line
    s = np.sqrt(l[0]**2+l[1]**2)

    # if norm is zero then vector is not scaled
    if s==0:
        print('Zero line vector in displayEpipolar')

    # distance of point from line formula
    l = l/s
    print("line is", l)

    # case when epipolar lines are running vertically through the image
    if l[0] != 0:
        # vary y and get corresp x values
        # ax + by + c = 0 (l = [a,b,c])
        # here we follow the eq: x = -(by +c)/a for multiple values of y
        x2, y2 = find_correspondences_vertical(im1, im2, x1, x2, l)

    # case when epipolar lines are running horizontally through image
    else:
        # vary x and get corresp y values
        # ax + by + c = 0 (l = [a,b,c])
        # here we follow the eq: y = -(ax + c)/a for multiple values of y
        x2, y2 = find_correspondences_horizontal(im1, im2, x1, x2, l)
    
    print("x1 and y1 is", x1, y1)
    print("x2 and y2 are", x2, y2)

    return x2, y2


def find_correspondences_vertical(im1, im2, x1, y1, l):
    """
    Vary the y value in eq. ax+by+c = 0 and get respective x value
    
    Args:
        im1 : image 1
        im2 : image 2
        x1  : keypoints in image 1 (given by user)
        y1  : keypoints in image 1 (given by user)
        l   : coefficients of line given by transforming x1,y1 with Fundamental matrix (F)

    Returns:
        x2  : calculated correspondence point in im2
        y2  : calculated correspondence point in im2
    """
    
    # run along y axis of image and find the best matching point
    sy, sx, _ = im2.shape

    # create a window of pixels about the keypoint for correspondence matching
    plain_window_im1 = im1[ 
                            (y2 - math.floor(WINDOW_SIZE/2)) : (y2 + math.floor(WINDOW_SIZE/2)),
                            (x2 - math.floor(WINDOW_SIZE/2)) : (x2 + math.floor(WINDOW_SIZE/2)),
                            :
                            ]
    
    gauss_window = create_gaussian_window(WINDOW_SIZE)

    intensity_error_min = 10000
    bestx2, besty2 = 0
    
    for y2 in range(0+WINDOW_SIZE,sy):
        # find the corresponding x at this y location
        x2 = -(l[1] * y2 + l[2])/l[0]

        # find window (eg. 3x3x3) around that pixel of y2 and x2
        plain_window_im2 = im2[ 
                            (y2 - math.floor(WINDOW_SIZE/2)) : (y2 + math.floor(WINDOW_SIZE/2)),
                            (x2 - math.floor(WINDOW_SIZE/2)) : (x2 + math.floor(WINDOW_SIZE/2)),
                            :
                            ]
        
        # find the difference between this window in im2 and it's respective window in im1
        diff_window = plain_window_im2 - plain_window_im1
        # weight the diff window according to our gaussian weights
        



def find_correspondences_horizontal(im1, im2, x1, y1, l):
    """
    Vary the x value in eq. ax+by+c = 0 and get respective y value
    
    Args:
        im1 : image 1
        im2 : image 2
        x1  : keypoints in image 1 (given by user)
        y1  : keypoints in image 1 (given by user)
        l   : coefficients of line given by transforming x1,y1 with Fundamental matrix (F)
    
    Returns:
        x2  : calculated correspondence point in im2
        y2  : calculated correspondence point in im2
    """
    pass

def create_gaussian_window(w_size):
    """
    Create a Gaussian Kernel which will be used to weight the window differently
    Args:
        w_size : window size (odd numbers above 3 only)
    Returns:
        gauss  : gaussian kernel of given window size
    """
    x, y = np.meshgrid(np.linspace(-1,1,w_size), np.linspace(-1,1,w_size))
    print(x)
    print(y)
    dst = np.sqrt(x*x+y*y)

    #Intializing sigma and muu (muu = mean at zero as normal dist)
    sigma = 1
    muu = 0.000
    
    #Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return gauss

if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    epipolarMatchGUI(im1, im2, F)
    
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)