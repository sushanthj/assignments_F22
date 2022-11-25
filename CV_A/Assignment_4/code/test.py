import math
import numpy as np

a = np.array([[1,2], [3,4]])

b = np.array([[5,6], [7,8]])

a = a.flatten()
b = b.flatten()

c = np.append(a,b)

print(c)