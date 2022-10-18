import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

timestep = 0.001
timespan = np.arange(0,10,timestep)
# timespan = np.linspace(0, 1, 1000, endpoint=True)
# print("timespan is", timespan)

steering_frequency = 1 # unit = hertz
steering_angle = 10*signal.square(2*np.pi*steering_frequency*timespan)
# steering_angle = 10*np.sin(2*np.pi*steering_frequency*timespan)

plt.plot(timespan, steering_angle)
 
# Give x,y,title axis label
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()