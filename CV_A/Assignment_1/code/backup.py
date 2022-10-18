'''
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
    print("input shape is", img.shape)
    # initialize a blank image which later stores filtered image
    fused_image = np.zeros(img.shape, img.dtype)
    # convert the image to lab space
    lab_img =  skimage.color.rgb2lab(img)
    img_split = [lab_img[:,:,0], lab_img[:,:,0], lab_img[:,:,0]]
    # ----- TODO -----
    # apply filters on each color channel seperately
    filter_resp = [None] * len(img_split)
    for i in range(len(img_split)):
        filter_resp[i] = scipy.ndimage.gaussian_filter(img_split[i], sigma=4)
    # fuse the inputs into one image again
    fused_image = np.dstack((filter_resp[0], filter_resp[1], filter_resp[2]))

    print("output shape is", fused_image.shape)

    plt.figure(figsize=(10,10))
    plt.imshow(skimage.color.lab2rgb((fused_image)))
    plt.show()

    # Finally make everythign modular to try 4 types of filters 
    # in one shot and compress all into one array (3(diff scales) x 4 x MxNx3)
    return filter_resp
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

# Load the Summer Palace photo
china_orig = Image.open('./test.jpg')

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china_orig).astype(np.float64) / 255
print("shape of image beforehand is", china.shape)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))
print("shape of image beforehand is", image_array.shape)


print("Fitting model on a small sub-sample of the data")
t0 = time()
print("image array shape is", image_array.shape)
image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
print("new sparse array shape is", image_array_sample.shape)

# kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print(f"done in {time() - t0:0.3f}s.")

# # Get labels for all points
# print("Predicting color indices on the full image (k-means)")
# t0 = time()
# labels = kmeans.predict(image_array)
# print(f"done in {time() - t0:0.3f}s.")


# codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
# print("Predicting color indices on the full image (random)")
# t0 = time()
# labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
# print(f"done in {time() - t0:0.3f}s.")

'''----------------------------------------------------------------------------------------------------'''

"""
# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Original image (96,615 colors)")
plt.imshow(china)

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, Random)")
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()

if __name__ == "__main__":
    make_dirs()
    load_cotton()
    pool = Pool(processes=10)
    result_list_tqdm = []
    for result in tqdm(pool.imap_unordered(func=main, iterable=img_list), total=len(img_list)):
       result_list_tqdm.append(result)

    with pool as p:
        p.starmap(compute_dictionary_one_image, zip(rgb_argument_list, repeat(orig_rgb_dict)))
    print("finished writing rgb images to disk")

    result_list_tqdm = []
    for result in tqdm(pool.imap_unordered(func=compute_dictionary_one_image, 
                                            iterable=train_files), total=sample_size):
       result_list_tqdm.append(result)
"""