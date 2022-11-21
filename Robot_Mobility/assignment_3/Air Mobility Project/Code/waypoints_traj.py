import numpy as np
import math

def lookup_waypoints(question):
    '''
    Input parameters:

    question: which question of the project we are on 
    (Possible arguments for question: 2, 3, 5, 6.2, 6.3, 6.5, 7, 9, 10)

    Output parameters:

    waypoints: of the form [x, y, z, yaw]
 
    waypoint_times: vector of times where n is the number of waypoints, 
    represents the seconds you should be at each respective waypoint
    '''

    # TO DO:

    # sample waypoints for hover trajectory 
    if int(question) == 2:
        # waypoints = [x_vals],[y_vals],[z_vals],[yaw_vals]
        waypoints = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0.05, 0.1, 0.1], [0,0,0,0]])
        waypoint_times = np.array([0,1,4,8])
        const_acc = None
        return([waypoints, waypoint_times, const_acc])
    
    elif int(question) == 3:
        const_accel = 0.0784
        const_acc = None # ignore this
        waypoint_times = np.arange(0,10.1,0.05)
        waypoints = np.zeros(shape=(4,waypoint_times.shape[0]))
        
        # space out waypoints according to constant accel profile
        for i in range(math.floor(len(waypoint_times)/2)):
            waypoints[2,i] = 0.5*const_accel*waypoint_times[i]*waypoint_times[i]

        halfway = waypoint_times[(math.floor(len(waypoint_times)/2))]
        
        for i in range(math.floor(len(waypoint_times)/2) - 1): # 4 timesteps are left for robot to stablize in hover
            waypoints[2,i+math.floor(len(waypoint_times)/2)] = 1 - (0.5 * const_accel * (waypoint_times[(math.floor(len(waypoint_times)/2)) + 1 + i] - halfway)**2)

        return([waypoints, waypoint_times, const_acc])

def trajectory_planner(question, waypoints, max_iteration, waypoint_times, time_step, const_acc):
    '''
    Input parameters:

    question: Which question we are on in the assignment

    waypoints: Series of points in [x, y, z, yaw] format
 
    max_iter: Number of time steps
 
    waypoint_times: Time we should be at each waypoint
 
    time_step: Length of one time_step
 
    Output parameters:
 
    trajectory_sate: [15 x max_iter] output trajectory as a matrix of states:
    [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]   
    '''

    # TO DO:
    if int(question) == 2 or int(question) == 3:
        # sample code for hover trajectory
        trajectory_state = np.zeros((15, max_iteration))
        # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

        current_waypoint_number = 0
        for i in range(0,max_iteration):
            # update the curr_waypoint_number depending on simulation iteration time
            if current_waypoint_number < len(waypoint_times)-1:
                if (i*time_step) >= waypoint_times[current_waypoint_number+1]:
                    current_waypoint_number +=1

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8,i] = waypoints[3, current_waypoint_number]

    return (trajectory_state)