import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def main():
    STR_MODE = 2

    timestep = 0.1
    timespan = np.arange(0,100,timestep)

    x, y, psi = init_state_variables()
    lf, lr, steering_angle, velocity = init_inputs(timespan, steering_mode=STR_MODE)
    pos_track_x = np.array([])
    pos_track_y = np.array([])

    for i in range(len(timespan)):
        x_dot = velocity*math.cos(psi)
        y_dot = velocity*math.sin(psi)
        if STR_MODE == 1:
            psi_dot = (velocity*math.tan(steering_angle[0]))/(lf+lr)
        elif (STR_MODE == 2) or (STR_MODE == 3):
            steering_angle_inst = steering_angle[i]
            psi_dot = (velocity*math.tan(steering_angle_inst))/(lf+lr)

        x += x_dot*timestep
        y += y_dot*timestep
        psi += psi_dot*timestep

        pos_track_x = np.append(pos_track_x, x)
        pos_track_y = np.append(pos_track_y, y)

    plot(pos_track_x, pos_track_y, STR_MODE)

def plot(x,y, str_mode):
    plt.plot(x,y)
    if str_mode == 1:
        plt.ylabel('position in y')
        plt.title('Pepy Model with constant driving angle')
        plt.xlabel('position in x')
    elif str_mode == 2:
        plt.ylabel('position in y')
        plt.title('Pepy Model with sinusoidal driving angle')
        plt.xlabel('position in x')
    elif str_mode == 3:
        plt.ylabel('position in y')
        plt.title('Pepy Model with square wave driving angle')
        plt.xlabel('position in x')
    plt.show()

def init_state_variables():
    x, y, psi = 0.0, 0.0, 0.0
    return x ,y, psi

def init_inputs(timespan, steering_mode=0):
    steering_amplitude = 0.5
    steering_angle = np.array([])
    
    # constant steering input
    if steering_mode == 1:
        steering_angle = np.append(steering_angle,steering_amplitude) # units = degrees
    
    # sinsusoidal steering input
    elif steering_mode == 2:
        steering_frequency = 0.1
        steering_angle = steering_amplitude*np.sin(2*np.pi*steering_frequency*timespan)
        print("shape of steering angle is", steering_angle)
    
    # square wave steering input
    elif steering_mode == 3:
        steering_frequency = 2
        timestep_new = 0.001
        timespan_new = np.arange(0,timespan[-1], timestep_new)
        steering_angle = steering_amplitude*signal.square(2*np.pi*steering_frequency*timespan_new)
    
    velocity = 2 # units = metres/second
    lf = lr = 1.5 # units = metres

    return lf, lr, steering_angle, velocity

if __name__ == '__main__':
    main()