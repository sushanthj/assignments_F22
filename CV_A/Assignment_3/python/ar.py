import sre_parse
import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
import HarryPotterize

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
        if success and currentframe%100 == 0 :
            # if video is still left continue creating images
            print("frame no. is", currentframe)
            frame = np.array(frame)
            frame_seq.append(frame)
        elif currentframe == 502:
            break
    
    # Release all space and windows once done
    cam.release()

    return frame_seq

def crop_frames(dst_frames, book_shape):
    # crop each dst_frame to match the src_frame's book
    pass

def warp_frames_and_composite(dst_frames, src_frames):

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    # invert the columns of locs1 and locs2
    locs1[:, [1, 0]] = locs1[:, [0, 1]]
    locs2[:, [1, 0]] = locs2[:, [0, 1]]

    matched_points = create_matched_points(matches, locs1, locs2)
    h, inlier = computeH_ransac(matched_points[:,0:2], matched_points[:,2:], opts)

    print("homography matrix is \n", h)
    
    composite_img = compositeH(h, template_img, image2)

    # Display images
    cv2.imshow("Composite Image :)", composite_img)
    cv2.waitKey()

if __name__ == '__main__':
    opts = get_opts()
    
    # extract frames from videos
    video1_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/book.mov'
    video2_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/ar_source.mov'
    dst_frames = extract_frames_2(video2_path)
    src_frames = extract_frames_2(video1_path)

    # crop frames of dst to fit src
    # dst_frames = crop_frames(dst_frames, src_frames[0])

    cv2.imshow("book_img", src_frames[0])
    cv2.waitKey()
    
    main(opts)
