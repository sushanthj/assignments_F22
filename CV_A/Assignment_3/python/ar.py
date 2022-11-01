import sre_parse
import numpy as np
import cv2
import skimage.io 
import skimage.color
import os
import warnings
import time
import HarryPotterize
from multiprocessing import Pool
from planarH import *
from opts import get_opts
from matchPics import matchPics
from itertools import repeat
from helper import loadVid

# find where book is in frame 1
BOOK_LOC = (134, 84, 414, 424)
# calculate size of book in pixels
BOOK_SHAPE = (280, 340)

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
        if success and currentframe%1 == 0 :
            # if video is still left continue creating images
            print("frame no. is", currentframe)
            frame = np.array(frame)
            frame_seq.append(frame)
        if currentframe == 1000:
            break
    
    # Release all space and windows once done
    cam.release()

    return frame_seq

def crop_frames_and_resize(dst_frames, crop_locs, scale_factor):
    crop_frames = []
    scale_percent = 1/scale_factor
    print("scale percent is", scale_percent)
    for frame in dst_frames:
        new_frame = frame[crop_locs[1]:crop_locs[3], crop_locs[0]:crop_locs[2]]
        width = int(new_frame.shape[1] * scale_percent)
        height = int(new_frame.shape[0] * scale_percent)
        dim = (width, height)
        new_frame_2 = cv2.resize(new_frame, dim, interpolation = cv2.INTER_LINEAR)
        crop_frames.append(new_frame_2)
    return crop_frames

def warp_frames_and_composite(ar_frames, book_frames, homography_ref, out_dir):
    # create out_dir if it doesn't exist
    check_and_create_directory(out_dir, create=1)
    
    # define threads for multi-threading
    pool = Pool(processes=15)
    
    print("No. of images to run filters on is", len(ar_frames))
    dataset_tracker = list(range(0, len(ar_frames)))

    print("len of dataset tracker is and of ar_frames is", len(dataset_tracker), len(ar_frames))

    start_time = time.time()
    with pool as p:
        p.starmap(
                ar_each_frame, 
                zip(
                    ar_frames, 
                    book_frames,
                    dataset_tracker,
                    repeat((opts, homography_ref))
                    ))
    end_time = time.time()
    print("***************************************")
    print("fps of video processing is", round((end_time-start_time)/len(ar_frames), 2))
    print("***************************************")

    compiled_video = compile_video(out_dir)
        

def compile_video(out_dir):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    """

    # create container for frames which will hold all images
    frames = []
    
    # hard set FPS to 30
    FPS = 30

    # output video path
    out_video_path = os.path.join(out_dir, 'ar_video.avi')

    # create dummny list to store each frame
    frames = list(range(0, len(os.listdir(out_dir))))
    
    # compile the frames
    for img_name in os.listdir(out_dir):
        if img_name.endswith(".jpeg"):
            frame_no = int(img_name.split(".")[0].split("_")[1])
            frames[frame_no]= (cv2.imread(os.path.join(out_dir, img_name)))

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_video_path, fourcc , FPS, (w, h))

    print("composing AR frames into video")
    for frame in tqdm(frames):
        writer.write(frame)

    writer.release()

def ar_each_frame(ar_frame, book_frame, i, pack):
    opts = pack[0]
    homography_ref = pack[1]
    matches, locs1, locs2 = matchPics(homography_ref, book_frame, opts)

    # invert the columns of locs1 and locs2
    locs1[:, [1, 0]] = locs1[:, [0, 1]]
    locs2[:, [1, 0]] = locs2[:, [0, 1]]
    
    matched_points = create_matched_points(matches, locs1, locs2)

    h, inlier = computeH_ransac(matched_points[:,0:2], matched_points[:,2:], opts)

    print("homography matrix is \n", h)
    
    composite_img = compositeH(h, ar_frame, book_frame)
    save_path = os.path.join(out_dir, 'frame' + '_' + str(i) + '.jpeg')
    cv2.imwrite(save_path, composite_img)

def check_and_create_directory(dir_path, create):
    """
    Checks for existing directories and creates if unavailable

    [input]
    * dir_path : path to be checked
    * create   : tag to specify only checking path or also creating path
    """
    if create == 1:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    else:
        if not os.path.exists(dir_path):
            warnings.warn(f'following path could not be found: {dir_path}')

if __name__ == '__main__':
    
    # receive user params
    opts = get_opts()
    
    # define directory where outputs of multi-proc will be saved
    out_dir = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/outputs/frames'
    
    # extract frames from videos
    video1_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/book.mov'
    video2_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_3/data/ar_source.mov'
    homography_ref = cv2.imread('../data/cv_cover.jpg')
    ar_frames = extract_frames_2(video2_path)
    book_frames = extract_frames_2(video1_path)

    # clip the processing to min_frames of both videos
    if len(ar_frames) > len(book_frames):
        ar_frames = ar_frames[0:len(book_frames)]
    else:
        book_frames = book_frames[0:len(ar_frames)]
    
    print("total frames to run AR on is", len(book_frames))
    
    # downscale the book shape to remove black regions from image
    scale_factor = 0.8
    book_shape_ds = (BOOK_SHAPE[0]*scale_factor, BOOK_SHAPE[1]*scale_factor)
    print("book shape ds is", book_shape_ds)

    # crop out central region of ar_frames
    c_y, c_x, _ = ar_frames[0].shape
    c_y, c_x = c_y/2, c_x/2
    crop_locs_ar = (int(c_x-book_shape_ds[0]/2),
                    int(c_y-book_shape_ds[1]/2),
                    int(c_x+book_shape_ds[0]/2),
                    int(c_y+book_shape_ds[1]/2))
    print("crops locs ar are ", crop_locs_ar)

    # resize homography_ref book to same size as AR crop
    homography_ref = cv2.resize(homography_ref, BOOK_SHAPE)

    # crop ar_frames to act as our harry_potter template image
    ar_frames = crop_frames_and_resize(ar_frames, crop_locs_ar, scale_factor)
    
    # warp_frames_and_composite(ar_frames, book_frames)
    warp_frames_and_composite(ar_frames, book_frames, homography_ref, out_dir)
