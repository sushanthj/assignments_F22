import sre_parse
import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
import HarryPotterize
from planarH import *

#Import necessary functions

from helper import loadVid

def main(opts):
    pass

def extract_frames(path):
    frames = loadVid(path)
    return frames

def extract_frames_2(path):
    frame_seq = []
    currentframe = 0

    # Read the video from specified path
    cam = cv2.VideoCapture(path)
    
    while(True):
        
        # reading from frame
        success,frame = cam.read()

        currentframe += 1
        if success and currentframe%10 == 0 :
            # if video is still left continue creating images
            print("frame no. is", currentframe)
            frame = np.array(frame)
            frame_seq.append(frame)
        if currentframe == 100:
            break
    
    # Release all space and windows once done
    cam.release()

    return frame_seq

def crop_frames(dst_frames, crop_locs):
    crop_frames = []
    for frame in dst_frames:
        crop_frames.append(frame[crop_locs[1]:crop_locs[3], crop_locs[0]:crop_locs[2]])
    return crop_frames

def warp_frames_and_composite(ar_frames, book_frames, homography_ref):
    for i in range(len(book_frames)):
        matches, locs1, locs2 = matchPics(homography_ref, book_frames[i], opts)

        # invert the columns of locs1 and locs2
        locs1[:, [1, 0]] = locs1[:, [0, 1]]
        locs2[:, [1, 0]] = locs2[:, [0, 1]]

        matched_points = create_matched_points(matches, locs1, locs2)
        h, inlier = computeH_ransac(matched_points[:,0:2], matched_points[:,2:], opts)

        print("homography matrix is \n", h)
        
        composite_img = compositeH(h, ar_frames[i], book_frames[i])

        # Display images
        cv2.imshow("Composite Image :)", composite_img)
        cv2.waitKey()

if __name__ == '__main__':
    opts = get_opts()
    
    # extract frames from videos
    video1_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/book.mov'
    video2_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/ar_source.mov'
    homography_ref = cv2.imread('../data/cv_cover.jpg')
    ar_frames = extract_frames_2(video2_path)
    book_frames = extract_frames_2(video1_path)

    # find where book is in frame 1
    book_loc = (134, 84, 414, 424)
    # calculate size of book in pixels
    book_shape = (280, 340)
    # crop the AR video frames according to the book size
    crop_locs_ar = (180, 10, 460, 350)
    # resize homography_ref book to same size as AR crop
    homography_ref = cv2.resize(homography_ref, book_shape)

    # crop ar_frames to act as our harry_potter template image
    ar_frames = crop_frames(ar_frames, crop_locs_ar)
    # warp_frames_and_composite(ar_frames, book_frames)

    warp_frames_and_composite(ar_frames, book_frames, homography_ref)

    cv2.imshow("ar_img", ar_frames[0])
    cv2.waitKey()
    cv2.imshow("book_img", book_frames[0])
    cv2.waitKey()
    cv2.imshow("template reshaped", homography_ref)
    cv2.waitKey()
    
    main(opts)
