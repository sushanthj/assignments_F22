import numpy as np

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
        waypoints = np.array([[0, 0.1, 0.2, 0.3],[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0,0,0,0]])
        waypoint_times = np.array([0,2,4,6])
        const_acc = None
        return([waypoints, waypoint_times, const_acc])
    
    elif int(question) == 3:
        # space out waypoints to 1cm between each (final dest is 1m)
        z_vals_tkoff = np.arange(start=0, stop=1, step=0.01)
        z_vals_land = np.arange(start=1, stop=0, step=-0.01)
        z_vals = np.append(z_vals_tkoff, z_vals_land)

        waypoints = np.array([np.zeros(shape=z_vals.shape), np.zeros(shape=z_vals.shape), z_vals, np.zeros(shape=z_vals.shape)])
        waypoint_times = np.arange(start=0, stop=20, step=0.1)
        const_acc = 1
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
    if int(question) == 2:
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

    elif int(question) == 3:
        trajectory_state = np.zeros((15, max_iteration))
        current_waypoint_number = 0
        z_vel = 0

        # update the curr_waypoint_number depending on simulation iteration time
        for i in range(0,max_iteration):
            if current_waypoint_number < len(waypoint_times)-1:
                if (i*time_step) >= waypoint_times[current_waypoint_number+1]:
                    current_waypoint_number +=1

            # update the state's x,y,z values
            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            # update the state's yaw value
            trajectory_state[8,i] = waypoints[3, current_waypoint_number]

            # we create a velocity profile by fixing a constant acceleration
            # use const_acc from lookup_waypoints

            # const positive accel upwards
            if current_waypoint_number < int((len(waypoint_times)-1)/4):
                # v = u + at
                z_vel = z_vel + const_acc*time_step
            elif current_waypoint_number == int((len(waypoint_times)-1)/4):
                # v = u + 0t
                z_vel = z_vel
            elif (current_waypoint_number > int((len(waypoint_times)-1)/4)) and \
                    (current_waypoint_number < int((len(waypoint_times)-1)/2)):
                # v = u -at
                z_vel = z_vel -const_acc*time_step
            elif current_waypoint_number == int(len(waypoint_times)-1)/2:
                # v = u + 0t
                z_vel = z_vel
            elif (current_waypoint_number > int((len(waypoint_times)-1)/2)) and \
                    (current_waypoint_number < int((len(waypoint_times)-1)*0.75)):
                # v = u + at
                z_vel = z_vel + const_acc*time_step
            elif (current_waypoint_number == int((len(waypoint_times)-1)*0.75)):
                # v = u + 0t
                z_vel = z_vel
            elif (current_waypoint_number > int((len(waypoint_times)-1)*0.75)):
                # v = u -at
                z_vel = z_vel -const_acc*time_step
            
            trajectory_state[5,i] = z_vel

    print("final waypoint number is", current_waypoint_number)
    return (trajectory_state)