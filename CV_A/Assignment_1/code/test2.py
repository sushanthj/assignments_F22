import numpy as np

x,y,z = 4,4,4
b = np.arange(x*y*z).reshape((x*y,z))
print(b)

random_labels = [0,3,5]


new_subsampled = b[random_labels, :]
print("new subsampled shape is \n", new_subsampled)
