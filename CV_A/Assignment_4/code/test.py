import math
import numpy as np

a = np.ones(shape=(3,3,3))

gauss_window = np.zeros(shape=(3,3))

gauss_window[1,1] = 1
print(gauss_window)

gauss_window = np.stack((gauss_window, gauss_window, gauss_window), axis=2)

print(gauss_window[:,:,0])

print((a*gauss_window)[:,:,0])