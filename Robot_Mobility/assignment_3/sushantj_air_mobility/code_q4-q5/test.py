import numpy as np

a = np.array([1,2,3,4])

b = np.array([1,0,2,5])

c = np.where(a != b)
for i in c[0].tolist():
    print(i)