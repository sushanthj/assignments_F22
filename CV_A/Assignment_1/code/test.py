from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool(processes=15) as pool:
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
    N = np.array(M)
    print(N.shape)

if __name__=="__main__":
    main()

'''
Trial
import multiprocessing as mp
import random
import string

# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                    string.ascii_lowercase
                    + string.ascii_uppercase
                    + string.digits)
               for i in range(length))
    output.put(rand_str)

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)

# Output
# ['yzQfA', 'PQpqM', 'SHZYV', 'PSNkD']

'''