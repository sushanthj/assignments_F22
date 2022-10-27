import numpy as np

H = np.ones((3,3))

x1 = np.array([1,2,3])

x2 = np.array([1,1,1])

x3 = np.vstack((x1, x2))
print(x3.shape)

x4 = H @ x2.T

print(x4)
print(np.max(x4, axis=0))
