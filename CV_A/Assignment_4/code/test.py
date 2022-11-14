import math
import numpy as np

a = np.array([[1,2],[3,4]])

b = np.where(a[:,1] == 0)

# print(b[0].shape[0])
WINDOW_SIZE = 1.5

print(math.floor(WINDOW_SIZE))
print(math.ceil(WINDOW_SIZE))