import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        

        # TODO: Convert Images to GrayScale
        I1 = skimage.color.rgb2gray(I1)
        I2 = skimage.color.rgb2gray(I2)
        
        # TODO: Detect Features in Both Images
        # locs1 is just the detected corners of I1
        locs1 = corner_detection(I1, sigma)
        locs2 = corner_detection(I2, sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        # We use the breif descriptor to give the patch descriptions (patch of pixel width = 9)
        # for the corners(keypoints) which we obtained from corner_description

        # desc is the binary string (len(string)=256 and 256bits)
        # which serves as the feature descriptor
        desc1, locs1 = computeBrief(I1, locs1)
        desc2, locs2 = computeBrief(I2, locs2)

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)

        return matches, locs1, locs2
