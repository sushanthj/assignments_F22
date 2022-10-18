import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
from matplotlib import animation
from LucasKanade import LucasKanade
from LucasKanade import strat3
from tqdm import tqdm
from copy import deepcopy

def disp_img(img, heading):
    img = np.array(img)
    window_name = heading
    
    # Displaying the image 
    cv2.imshow(window_name, img)
    cv2.waitKey()

def updatefig(i, plot_idx = [0,99,199,299,399]):
    # This function will be called and save images at the above intervals
    
    rect = lk_res[i]
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    patch.set_width(pt_bottomright[0] - pt_topleft[0])
    patch.set_height(pt_bottomright[1] - pt_topleft[1])
    patch.set_xy((pt_topleft[0],pt_topleft[1]))
    im.set_array(seq[:,:,i])
    if i in plot_idx:
        plt.savefig("result/tracking_atharv_" + str(i) + ".png", bbox_inches='tight', pad_inches=0)
    return im,


def create_img_seq(in_dir, out_dir):
    temp_seq = []
  
    # Read the video from specified path
    cam = cv2.VideoCapture(in_dir)
    
    try:
        # creating a folder named data
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    currentframe = 0
    
    while(True):
        
        # reading from frame
        success,frame = cam.read()

        currentframe += 1
        if success and currentframe%1 ==0 :
            # if video is still left continue creating images
            # name = './data/frame' + str(currentframe) + '.jpg'
            # print ('Creating...' + name)
    
            # writing the extracted images
            # cv2.imwrite(name, frame)
            print("frame no. is", currentframe)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale_percent = 25 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_LINEAR)
            frame = np.array(frame)
            temp_seq.append(frame)
        elif currentframe == 739:
            break
    
    atharv_seq = np.stack(temp_seq, axis=2)
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    np.save(os.path.join(out_dir, 'atharvseq.npy'), atharv_seq)


# Create the image frames from given video file
in_dir = '../videos/test_1.mp4'
out_dir = '../data'
create_img_seq(in_dir, out_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/atharvseq.npy")
# seq = np.load("../data/carseq.npy")


rect_0 = np.array([336, 60, 357, 212]).T
rect = np.copy(rect_0)

threshold_drift = 3
####### Run LK Template correction #######
lk_res = []
lk_res.append(rect)

# T1 image we will use throughout
It0 = seq[:,:,0]
disp_img(It0, "test")

# return this after iteration in the loop to 
# save which img to save as template
template_idx = 0

# the rect we will pass to LK and will keep being saved for each frame
template_rect = deepcopy(rect)

# rect will save the position of the crop in the current frame
# rect is also calculated outside LK for clarity IDK

# rect_0 will not change and has to passed to LK everytime
p = np.array([0.0, 0.0])

for i in tqdm(range(1, seq.shape[2])):
    pt_topleft = rect[:2]
    pt_bottomright = rect[2:4]
    It1 = seq[:,:,i]
    print("template idx is", template_idx)
    print("template rect is", lk_res[template_idx])
    # applying strategy 3 in research paper (Lucas Kanade 20 years on)
    p, template_idx = strat3(
                                It0, 
                                It1, 
                                lk_res[template_idx], 
                                template_rect, 
                                threshold, 
                                template_threshold, 
                                seq[:,:,template_idx],
                                num_iters,
                                i,
                                template_idx
                                )
    rect = np.concatenate((pt_topleft + p, pt_bottomright + p))
    lk_res.append(rect)

lk_res = np.array(lk_res)
np.save("atharv_seq_rect.npy", lk_res)

##### Code for animating, debugging, and saving images. 
lk_res = np.load("atharv_seq_rect.npy")

fig,ax = plt.subplots(1)
It1 = seq[:,:,0]
rect = lk_res[0]
pt_topleft = rect[:2]
pt_bottomright = rect[2:4]
patch = patches.Rectangle((pt_topleft[0],pt_topleft[1]), pt_bottomright[0] - pt_topleft[0],pt_bottomright[1] - pt_topleft[1] ,linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(patch)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im = ax.imshow(It1, cmap='gray')
ani = animation.FuncAnimation(fig, updatefig, frames=range(lk_res.shape[0]), 
                            interval=50, blit=True)
os.makedirs("result", exist_ok=True)
plt.show() 

### Sample code for genearting output image grid
fig, axarr = plt.subplots(1, 5)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plot_idx = [0,99,199,299,399]
for i in range(5):
    axarr[i].imshow(plt.imread(f"result/tracking_atharv_" + str(plot_idx[i]) + ".png"))
    axarr[i].axis('off'); axarr[i].axis('tight'); axarr[i].axis('image'); 
plt.show()