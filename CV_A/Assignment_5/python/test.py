import numpy as np

b = np.array([[1,2,3],[4,5,6],[9,8,7]])

c = np.array([[0,0,1],[0,0,1],[0,0,1]])

d = np.where(np.argmax(b, axis=1) == np.argmax(c, axis=1))

print(d[0].shape[0])