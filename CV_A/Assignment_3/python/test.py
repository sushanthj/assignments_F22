import numpy as np
import cv2
import skimage.io 
import skimage.color
from planarH import *
from opts import get_opts
from matchPics import matchPics

H = np.ones((3,3))

x1 = np.array([1,2,3], dtype=np.float32)

x2 = np.array([1,1,1])
x3 = np.array([[0,0],[0,1],[1,1],[1,0]]).reshape(-1,1,2)

print(x3[:,:,1])
print(x3.shape)

x4 = H @ x2.T

print(x4)
print(np.max(x4, axis=0))
