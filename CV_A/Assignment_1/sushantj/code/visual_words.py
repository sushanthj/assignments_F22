from distutils.log import warn
from logging import warning
import os
from os.path import join, isfile
import random
from tarfile import PAX_FIELDS
import warnings
import pickle

import numpy as np
import scipy.ndimage
import skimage.color
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# matplotlib only for local function tests
import matplotlib.pyplot as plt

def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    # get filter scales from user input
    filter_scales = opts.filter_scales

    # handle for different img shapes
    #print("input shape is", img.shape)
    if (len(img.shape) == 2):
        img_new = np.dstack((img, img, img))
    elif (len(img.shape) == 4):
        img_new = img[:,:,:,0]
    else:
        img_new = img

    #print(f'corrected shape is {img_new.shape}')
    
    # convert the image to lab space
    lab_img =  skimage.color.rgb2lab(img_new)

    no_of_filters = 4
    # split img to individual channels
    img_channels = [lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]]

    # construct the output image shape which holds filter resp
    try:    
        filter_resp = np.zeros(shape=(img_new.shape[0],img_new.shape[1],
                                img_new.shape[2]*no_of_filters*len(filter_scales)), 
                                dtype=img_new.dtype)
    except:
        print("one of the images had an issue")

    # iterate over each scale
    k = 0
    for i in range(len(filter_scales)):
        # run each filter for this scale
        for j in range(len(img_channels)):
            filter_resp[:,:,k] = scipy.ndimage.gaussian_filter(img_channels[j], sigma=filter_scales[i])
            k += 1
        for j in range(len(img_channels)):
            filter_resp[:,:,k] = scipy.ndimage.gaussian_laplace(img_channels[j], sigma=filter_scales[i])
            k += 1
        for j in range(len(img_channels)):
            filter_resp[:,:,k] = scipy.ndimage.gaussian_filter(img_channels[j], sigma=filter_scales[i], order=[0,1])
            k += 1
        for j in range(len(img_channels)):
            filter_resp[:,:,k] = scipy.ndimage.gaussian_filter(img_channels[j], sigma=filter_scales[i], order=[1,0])
            k += 1

    # pass final image of size array (3(scales) x 4(filters) x MxNx3)
    return filter_resp


def compute_dictionary_one_image(img_index, img_data):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * img_index       : index in the train_files which gives the image name
    * img_data        : user inputs and train_files

    [saved]
    * filter_response : saved image of shape height x width x (3*filter_scales*no_of_filters)
    """
    img_list = img_data[0]
    data_dir = img_data[1]
    opts = img_data[3]
    n_worker = img_data[4]

    img_name = img_list[img_index]
    img_input_path = os.path.join(data_dir, img_name)
    check_and_create_directory(img_input_path, create=0)

    feat_out_dir = os.path.join(img_data[2], os.path.split(os.path.splitext(img_name)[0])[1])

    img = Image.open(img_input_path)
    img = np.array(img).astype(np.float64) / 255
    filter_response = extract_filter_responses(opts, img)
    np.save(feat_out_dir, filter_response)

    if img_index == (len(img_list)-1):    
        print("Finished running filters over all images")


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = join(opts.feat_dir, 'filter_resps')
    out_dir = opts.out_dir
    check_and_create_directory(out_dir, create=1)
    check_and_create_directory(feat_dir, create=1)
    # K = no. of words 
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    print(f'saving final outputs to {out_dir} and filter_responses to {feat_dir}')

    # ----- TODO -----
    # Read train_files and subsample a smaller list of images
    # Open all images and extract the filter responses (maybe use multiproc)
    # Feed filter responses below

    # define threads for multi-threading
    pool = Pool(processes=15)
    
    print("No. of images to run filters on is", len(train_files))
    
    # track the items of the dataset
    dataset_tracker = list(range(0, len(train_files)))
    
    # check number of files in filter_response folder to avoid rerunning filters
    num_files = len([name for name in os.listdir(feat_dir)])
    if num_files != len(train_files):
        print("running filters on all images")
        # multi-threading for extracting filter responses 
        with pool as p:
            p.starmap(compute_dictionary_one_image, 
                        zip(dataset_tracker, 
                            repeat((train_files, 
                                    data_dir, 
                                    feat_dir, 
                                    opts, 
                                    n_worker))))
    else:
        print("filters have already been run on the training images")
    
    if not os.path.exists(join(out_dir,'dictionary.npy')):
        # merge all filter responses into one matrix (taking randomized pixels)
        print("merging filter responses")
        num_files = len([name for name in os.listdir(feat_dir)])
        merged_img = merge_filter_responses(feat_dir, num_files, opts)
        print("merged filter_responses shape is", merged_img.shape)
        dict_kmeans = KMeans(n_clusters=K).fit(merged_img)
        dictionary = dict_kmeans.cluster_centers_
        # print("dict of kmeans is", dict_kmeans.n_features_in_)

        # example code snippet to save the dictionary
        np.save(os.path.join(out_dir, 'dictionary.npy'), dictionary)
        # with open(os.path.join(out_dir, 'dictionary.pkl'), "wb") as f:
        #     pickle.dump(dict_kmeans, f)
    else:
        print("filter_responses have already been merged")

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

def merge_filter_responses(resp_dir, num_images, opts):
    """
    Iterates over saved filter_responses and combines them into
    one large numpy array

    [input]
    * resp_dir   : directory where all filter responses are stored
    * alpha      : user-defined crops of each image to take at random
    * num_images : number of images stored in the folder
    """
    alpha = opts.alpha
    final_img = None

    RANDOM_METHOD = 1
    
    # iterate over the filter responses in each folder
    count = 0
    for img_name in os.listdir(resp_dir):
        if img_name.endswith(".npy"):
            # read each image
            img_array = np.load(os.path.join(resp_dir, img_name))

            if RANDOM_METHOD == 1:
                # get random crops of size alpha x alpha from each image
                x_start = random.randrange(0, int(img_array.shape[0])-alpha)
                y_start = random.randrange(0, int(img_array.shape[1])-alpha)
                img_crop = img_array[x_start:x_start+alpha, y_start:y_start+alpha,:]
                #print("image crop shape", img_crop.shape)
                x,y,z = img_crop.shape
                img_resized = np.reshape(img_crop, (x*y, z))
                if count == 0:
                    final_img = img_resized
                else:
                    final_img = np.vstack((final_img, img_resized))
                count += 1
            
            elif RANDOM_METHOD == 2:
                x,y,z = img_array.shape
                ran_pixel_indices_x = np.random.choice(int(x), size=alpha)
                ran_pixel_indices_y = np.random.choice(int(y), size=alpha)
                resized_subsampled = img_array[ran_pixel_indices_x, ran_pixel_indices_y, :]
                print("the resized subsampled shape is", resized_subsampled.shape)
                if count == 0:
                    final_img = resized_subsampled
                else:
                    final_img = np.vstack((final_img, resized_subsampled))
                count += 1
    
    print("new merged image shape is", final_img.shape)

    return final_img



def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts       : options
    * img        : numpy.ndarray of shape (H,W) or (H,W,3)
    * dictionary : KMeans cluster data

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    PREDICTION_MECHANISM = 2

    # print("input image shape is", img.shape)
    filtered_image = extract_filter_responses(opts, img)
    x,y,z = filtered_image.shape
    img_scaled = np.reshape(filtered_image, (x*y, z))
    img_scaled = img_scaled.astype(np.float64)
    predictions = None
    
    if PREDICTION_MECHANISM == 1:
        img_scaled = img_scaled.astype(np.float64)
        predictions = dictionary.predict(img_scaled)
        predictions = predictions.reshape(x,y)
    elif PREDICTION_MECHANISM == 2:
        # dists = cdist(img_scaled,dictionary.cluster_centers_,metric='euclidean')
        dists = cdist(img_scaled, dictionary, metric='euclidean')
        num_channels = dists.shape[1]
        dists = dists.reshape(x,y,num_channels)

        predictions = np.zeros(shape=(x,y),dtype=dists.dtype)

        for i in range(x):
            for j in range(y):
                pixel_cluster = np.argmin(dists[i,j,:])
                predictions[i,j] = pixel_cluster
    
    if predictions.all != None:
        # print("successfully created the wordmap")
        pass
    else:
        warnings.warn("wordmap could not be created")

    return predictions
