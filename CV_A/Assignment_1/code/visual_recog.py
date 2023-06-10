from calendar import c
from cgi import test
from cmath import exp, sqrt
from concurrent.futures.thread import _worker
from email.errors import CloseBoundaryNotFoundDefect
import os
import math
import multiprocessing
from os.path import join
from copy import deepcopy, copy
from itertools import repeat

import pickle
import numpy as np
from PIL import Image

import util
import visual_words


def get_feature_from_wordmap(opts, wordmap, norm_check=1):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    K_list = list(range(0,K+1))
    x,y = wordmap.shape
    # flat_wordmap = wordmap.reshape(x*y) # makes no difference to flatten
    if norm_check == 1:
        histogram = np.histogram(wordmap, K_list, density=True)
    else:
        histogram = np.histogram(wordmap, K_list)

    histogram = histogram[0].astype(np.float64)
    return histogram


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * img   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    SPM_MODE = 2

    K = opts.K
    L = opts.L
    total_layers = L
    total_cells_last_layer = int(math.pow(2,(2*(total_layers))))
    
    # zero pad images to make the length along each axis divisible by the total_cells in last layer
    height, width = wordmap.shape
    pad_axis_0 = int((total_cells_last_layer * math.ceil(height/total_cells_last_layer)) - height)
    pad_axis_1 = int((total_cells_last_layer * math.ceil(width/total_cells_last_layer)) - width)
    npad = ((pad_axis_0,0),(pad_axis_1,0))
    # wordmap = np.pad(wordmap, pad_width=npad, mode='constant', constant_values=0)
    x,y = wordmap.shape
    
    if SPM_MODE == 2:
        """
        Efficient Computing
        """
        cell_width_last_layer = int(y/int(math.sqrt(total_cells_last_layer)))
        cell_height_last_layer = int(x/int(math.sqrt(total_cells_last_layer)))
        # print("cell height and width at last layer is", cell_height_last_layer, cell_width_last_layer)
        # print("                                                 ")
        
        # clumping all cells together as images for lowest layer
        image_matrix = np.zeros(
                                shape=(cell_height_last_layer,
                                        cell_width_last_layer,
                                        total_cells_last_layer),
                                dtype=wordmap.dtype)
        
        # create the image cells for the last layer
        image_matrix = create_img_cells(wordmap, image_matrix)
        
        # init a series of matrices to store the histogram values at each level
        normed_hists_matrices = [None]*(total_layers+1)
        
        # iterate to store the correct shape of hist at each level
        for i in range(total_layers+1):
            if i == 0:
                normed_hists_matrices[i] = np.zeros(shape=K, dtype=np.float64)    
            else:
                depth_of_hist = int(math.pow(2,(2*i)))
                length_of_hist = K
                normed_hists_matrices[i] = np.zeros(shape=(length_of_hist, depth_of_hist), dtype=np.float64)
        
        ### normed_hists_matrices = [layer][K, no._of_cells_in_this_layer] ###

        count = 0
        # compute histogram for last layer mosaic
        for i in range(int(math.sqrt(total_cells_last_layer))):
        # iterate along axis = 0
            for j in range(int(math.sqrt(total_cells_last_layer))):
                img_to_hist = image_matrix[:,:,count]
                cell_hist = get_feature_from_wordmap(opts, img_to_hist, norm_check=1)
                # print("cell hist is", cell_hist)
                normed_hists_matrices[total_layers][:,count] = cell_hist
                count += 1
        
        # merge the histograms of lower layers to give upper layer histograms
        # iterate over all layers from lowest to top order (skipping the smallest cells layer)
        for i in range(total_layers-1, -1, -1):
            if i == 0:
                blank_hist = np.zeros(shape=(K,4), dtype=np.float64)
                blank_hist[:,0] = normed_hists_matrices[i+1][:,0]
                blank_hist[:,1] = normed_hists_matrices[i+1][:,1]
                blank_hist[:,2] = normed_hists_matrices[i+1][:,2]
                blank_hist[:,3] = normed_hists_matrices[i+1][:,3]
                blank_hist_sum = np.sum(blank_hist, axis=1, dtype=np.float64)
                blank_hist_normed = normalize_1D(blank_hist_sum)
                normed_hists_matrices[i][:] = blank_hist_normed
            else:
                # define no. histograms for this layer
                num_hists = int(math.pow(2,(2*i)))
                num_hists_lower_layer = int(math.pow(2,(2*(i+1))))
                e_0 = int(math.sqrt(num_hists)) # defines length of edge in current layer (units=cells)
                e = int(math.sqrt(num_hists_lower_layer)) # defines length of edge in lower layer (units=cells)

                row_count = 0
                # iterate over each histogram in current layer
                for j in range(num_hists):

                    '''
                    Convert from row-major order to square block (stack or concat)
                    '''
                    # correction factor to ensure we go to next cell after we finish one row of cells
                    if (j/e_0 >= 1) and (j%e_0 == 0):
                        row_count += e_0
                    # create an empty image to store the 4 cells of the lower layer
                    # which feed into the 1 cell of the upper layer
                    blank_hist = np.zeros(shape=(K,4), dtype=np.float64)
                    blank_hist[:,0] = normed_hists_matrices[i+1][:,row_count*2]
                    blank_hist[:,1] = normed_hists_matrices[i+1][:,row_count*2+1]
                    blank_hist[:,2] = normed_hists_matrices[i+1][:,row_count*2+e]
                    blank_hist[:,3] = normed_hists_matrices[i+1][:,row_count*2+e+1]
                    
                    blank_hist_sum = np.sum(blank_hist, axis=1, dtype=np.float64)
                    blank_hist_sum = normalize_1D(blank_hist_sum)
                    normed_hists_matrices[i][:,j] = blank_hist_sum
                    row_count += 1
        
        # Above for-loop has created unnormalized histograms at each level

        hist_all = get_final_hist(normed_hists_matrices, K, total_layers, mode=1)
        # print("hist for one image shape is", hist_all.shape)
    
    elif SPM_MODE == 2:
        """
        Computing all histograms manually
        """
        # init a series of matrices to store the hist values at each level
        normed_hists_matrices = [None]*(total_layers+1)
        img_matrices = [None]*(total_layers+1)
        

        for i in range(total_layers+1):
            if i == 0:
                img_matrices[0] = wordmap
                # print("orig image shape is", wordmap.shape)
            else:
                cells_in_layer = int(math.pow(2,(2*(i))))
                cell_width = math.floor(y/int(math.sqrt(cells_in_layer)))
                cell_height = math.floor(x/int(math.sqrt(cells_in_layer)))

                # clumping all cells together as images for lowest layer
                image_mosaic = np.zeros(
                                        shape=(cell_height,
                                                cell_width,
                                                cells_in_layer),
                                        dtype=wordmap.dtype)
                # create the image cells for the last layer
                img_matrices[i] = create_img_cells(wordmap, image_mosaic)
        
        # iterate to store the correct shape of hist at each level
        for i in range(total_layers+1):
            if i == 0:
                normed_hists_matrices[i] = np.zeros(shape=K, dtype=np.float64)    
            else:
                depth_of_hist = int(math.pow(2,(2*i)))
                length_of_hist = K
                normed_hists_matrices[i] = np.array([], dtype=np.float64)
        
        ### normed_hists_matrices = [layer][K, no._of_cells_in_this_layer] ###

        for k in range(total_layers+1):
            if k == 0:
                cell_hist = get_feature_from_wordmap(opts, img_matrices[0], norm_check=1)
                normed_hists_matrices[k] = cell_hist
            # compute histogram for each layer mosaic
            else:
                for i in range(img_matrices[k].shape[2]):
                    img_to_hist = img_matrices[k][:,:,i]
                    cell_hist = get_feature_from_wordmap(opts, img_to_hist, norm_check=1)
                    normed_hists_matrices[k] = np.append(normed_hists_matrices[k], cell_hist)
        hist_all = get_final_hist(normed_hists_matrices, K, total_layers, mode=2)
            
    return hist_all


def normalize_1D(inp):
    inp = inp/np.sum(inp, dtype=np.float64)
    return inp

def get_final_hist(hist_matrix, K, total_layers, mode):
    # normalize matrices so that multiplying and adding weights will return 1
    for i in range(total_layers+1):
        if i == 0:
            layer_weight = math.pow(2,(-1*total_layers))
            hist_matrix[i] *= layer_weight
        else:
            hist_matrix[i] = hist_matrix[i]/np.sum(hist_matrix[i])
            layer_weight = math.pow(2,(i-total_layers-1))
            hist_matrix[i] *= layer_weight
    
    # combine all histograms into a one-dimensional array
    final_hist = np.array([], dtype=np.float64)
    
    if mode == 1:
        for i in range(total_layers+1):
            if i == 0:
                final_hist = np.append(final_hist, hist_matrix[i])
                # print(f'layer {i} has histogram \n {hist_matrix[i]}')
            else:
                x,y = hist_matrix[i].shape
                # print(f'layer {i} has hist matrix \n {hist_matrix[i]}')
                for column in range(y):
                    final_hist = np.append(final_hist, hist_matrix[i][:,column])
    elif mode == 2:
        for i in range(total_layers+1):
            final_hist = np.append(final_hist, hist_matrix[i])

    return final_hist

def get_overall_norm(matrix):
    np.sum(matrix)

def create_img_cells(img, image_matrix):
    img_mat = deepcopy(image_matrix)
    cell_height, cell_width, total_cells = img_mat.shape
    
    # extract image cells in ROW-MAJOR ORDER
    count = 0
    # iterate along axis = 1
    for i in range(int(math.sqrt(total_cells))):
        # iterate along axis = 0
        for j in range(int(math.sqrt(total_cells))):
            img_mat[:,:,count] = img[i*cell_height : i*cell_height + cell_height, j*cell_width : j*cell_width + cell_width]
            count += 1

    ### image of last layer is array([cell_1],[cell_2]....[cell_n]) and each cell is stacked along depth axis in row-major order ###
    
    return img_mat


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    img = Image.open(img_path)
    img = np.array(img).astype(np.float64)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    hist_single = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return hist_single


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))


    # Create Histogram of words from random input image
    img_path = join(opts.data_dir, 'kitchen/sun_aeazqdzaihtynisp.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float64)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    ### Q2.1 ###
    normal_hist = get_feature_from_wordmap(opts, wordmap, norm_check=1)
    print("histogram from wordmap using simple histogram is", normal_hist)

    ### Q2.2 ###
    single_image_hist = get_feature_from_wordmap_SPM(opts, wordmap)
    print("histogram from wordmap using Spatial Pyramid Histogram is", single_image_hist)
    print(" ")
    
    print("Now computing for all training images")

    data_dir = opts.data_dir
    hist_dir = join(opts.feat_dir, 'hist_resps')
    out_dir = opts.out_dir
    visual_words.check_and_create_directory(out_dir, create=1)
    visual_words.check_and_create_directory(hist_dir, create=1)
    
    num_files = len([name for name in os.listdir(hist_dir)])
    
    if num_files != len(train_files):
        print("running histograms on all images")
        pool = multiprocessing.Pool(processes=15)
        # track the items of the dataset
        dataset_tracker = list(range(0, len(train_files)))

        with pool as p:
            p.starmap(save_training_hists, 
                        zip(dataset_tracker, 
                            repeat((train_files, 
                                    data_dir, 
                                    hist_dir, 
                                    opts, 
                                    dictionary))))
    else:
        print("histograms have already been run on the training images")
    
    num_files = len([name for name in os.listdir(hist_dir)])
    print("number of histograms saved is", num_files)
    
    # Define variable which will hold histogram resps over all training images
    merged_hist = None

    if not os.path.exists(join(out_dir,'histograms.npy')):
        # merge all filter responses into one matrix (taking randomized pixels)
        merged_hist = merge_histograms(hist_dir, num_files, opts, train_files)

        np.save(os.path.join(out_dir, 'histograms.npy'), merged_hist)
    else:
        merged_hist = np.load(os.path.join(out_dir, 'histograms.npy'))
        print("Loading saved histogram model")

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=merged_hist,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    
    ###################################################################################################################

def merge_histograms(hist_dir, num_files, opts, train_files):
    """
    Iterates over saved histograms and combines them into
    one large numpy array
    """
    
    # final_hist = np.zeros(shape=(num_files, ((opts.K * int((math.pow(4,(opts.L+1))-1)/3)))))
    
    # iterate over all stored histograms
    count = 0
    for img_name in train_files:
            # read each image
            img_with_changed_ext = os.path.split(os.path.splitext(img_name)[0])[1] + ".npy"
            hist = np.load(os.path.join(hist_dir, img_with_changed_ext))
            hist = np.expand_dims(hist, axis=0)

            if count == 0:
                final_hist = hist
            else:
                final_hist = np.vstack((final_hist, hist))
            count += 1
    
    print("shape of new merged hist is", final_hist.shape)

    return final_hist

def save_training_hists(img_index, img_data):
    train_files = img_data[0]
    data_dir = img_data[1]
    hist_dir = img_data[2]
    opts = img_data[3]
    model = img_data[4]

    img_name = train_files[img_index]
    img_input_path = os.path.join(data_dir, img_name)
    visual_words.check_and_create_directory(img_input_path, create=0)

    img = Image.open(img_input_path)
    img = np.array(img).astype(np.float64) / 255
    wordmap = visual_words.get_visual_words(opts, img, model)
    hist_single = get_feature_from_wordmap_SPM(opts, wordmap)
    # print("hist single shape is", hist_single.shape)
    # print("shape of hist for this process is", hist_single.shape)

    hist_out_dir = os.path.join(hist_dir, os.path.split(os.path.splitext(img_name)[0])[1])
    np.save(hist_out_dir, hist_single)
    # with open(hist_out_dir, "wb") as f:
    #     pickle.dump(hist_single, f)


def similarity_to_set(word_hist, histograms, opts):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K) (test image histogram)
    * histograms: numpy.ndarray of shape (N,K) (trained histograms)
    * opts: user inputs

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    # print("histograms shape is", histograms.shape)
    out_dir = opts.out_dir
    ref_dict_path = join(opts.out_dir, 'histograms.npy')
    try:
        ref_dict = np.load(ref_dict_path)
    except:
        print("could not load ref dict")
        return None
    
    train_files_list = list(range(0, ref_dict.shape[0]))
    M = []

    # with multiprocessing.Pool(processes=15) as pool:
    #     M = pool.starmap(get_hist_similarities, zip(train_files_list, repeat((word_hist, histograms))))

    # itertae through every train file's histogram and get distance measure (inverse of similarity score)
    for i in train_files_list:
        M.append(get_hist_similarities(i, (word_hist, histograms)))
    
    return np.array(M)
    

def get_hist_similarities(index, data):
    '''
    Returns the sum of minimum values of histogram
    '''
    word_hist = data[0]
    histograms = data[1]
    if (word_hist.shape != histograms[index,:].shape):
        print("dictionary error")
        return None
    min_value_sum = 0.0
    #print("test image histogram shape is", word_hist.shape[0])

    for i in range(word_hist.shape[0]):
        #print(f'histogram of the {i}th training imgage has shape {histograms[index,:].shape}')
        # print(f'value of {index}th layer in word hist is at {i}th index is {word_hist[i]}')
        # print(f'value of {index}th layer in train_hist is at {i}th index is {histograms[index,i]}')
        min_value_sum += min(word_hist[i],histograms[index,i])
    
    dist_measure = 1 - min_value_sum

    return dist_measure


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]
    # with open((join(opts.out_dir, 'dictionary.pkl')), "rb") as f:
    #     dictionary = pickle.load(f)
    
    existing_hist_all = trained_system["features"]

    # using the stored options in the trained system instead of opts.py
    K = opts.K
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)

    print("computing confusion matrix")
    confusion_matrix = np.zeros(shape=(8,8), dtype=np.float64)
    # Itereate over train files and build confusion matrix
    for i in range(len(test_files)):
        img_path = join(data_dir, test_files[i])
        hist_test_img = get_image_feature(test_opts, img_path, dictionary)
        dist_to_all_hists = similarity_to_set(hist_test_img, existing_hist_all, test_opts)
        match_label = train_labels[np.argmin(dist_to_all_hists)]
        # print(f'matched label for image {test_files[i]} is {match_label} and the min value was {np.min(dist_to_all_hists)}')
        confusion_matrix[test_labels[i], match_label] += 1
    
    accuracy = confusion_matrix.trace()/np.sum(confusion_matrix)
    return confusion_matrix, accuracy


def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass