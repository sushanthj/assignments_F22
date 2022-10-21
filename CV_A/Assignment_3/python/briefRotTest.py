import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

# Q2.1.6

def rotTest(opts):

    I1_orig = cv2.imread('../data/cv_cover.jpg')
    range_obj_1 = range(3)
    range_obj_2 = range(1,9,3)
    # compute_histogram_and_display(I1_orig, range_obj_1, display=False)
    compute_histogram_and_display(I1_orig, range_obj_2, display=True)


def compute_histogram_and_display(I1_orig, range_object, display):
    
    # container to store the count of matches for each orientation
    match_counts = np.array([])
    rotations = np.array([])
    
    for i in range_object:
        # store the rotations in each iteration
        rotations = np.append(rotations, i*10)
        
        # Rotate Image
        I1_rot = ndimage.rotate(I1_orig, i*10)

        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(I1_orig, I1_rot, opts)

        # store the count
        match_counts = np.append(match_counts, matches.shape[0])
        print(f'the orientation {i*10} has {matches.shape[0]} matches')

        if display is True:
            plotMatches(I1_orig, I1_rot, matches, locs1, locs2)
    
    # plt.hist(match_counts, bins=rotations, density = False, 
    #      histtype ='bar')

    plt.bar(rotations, match_counts)
    
    plt.title('Histogram of matches vs Rotation\n',
          fontweight ="bold")
    plt.xlabel("rotation angles (degrees)")
    plt.ylabel("match counts")
    plt.show()


if __name__ == "__main__":

    opts = get_opts()

    rotTest(opts)
