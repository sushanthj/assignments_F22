import numpy as np
import copy

img1 = np.array([[0, 1, 2, 3, 0],
                [0, 1, 1, 2, 0],
                [0, 2, 3, 2, 0],
                [0, 0, 3, 4, 0]])

img2 = np.array([[2, 1, 2, 3, 1],
                [2, 1, 4, 4, 2],
                [2, 2, 4, 4, 2],
                [2, 2, 3, 4, 1]])

# out_1, out_2 = np.nonzero((img1!=0) & (img2!=0))
# out_1, out_2 = np.nonzero((img1!=0))
out_1, out_2 = np.where(img1==0)

# print(out_1)
# print(out_2)

x,y = img2.shape
# img2_mod = np.zeros(shape=(x,y), dtype=img2.dtype)
img2_mod = copy.deepcopy(img2)

for i in range(len(out_1)):
    x_coord = out_1[i]
    y_coord = out_2[i]
    img2_mod[x_coord, y_coord] = 0

# print("udpated img2 is\n", img2_mod)


x = 3
y = 2
inter = y
y = x
x = inter

print(x, y)