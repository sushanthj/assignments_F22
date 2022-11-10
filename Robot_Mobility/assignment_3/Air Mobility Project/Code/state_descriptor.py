import numpy as np

class StateDescriptor:
    """
    Runs a user defined state of the quad
    """
    def __init__(self, mode, params, start_time):
        """
        Accept the boundaries of the state
        Args:
            mode (int): specify which mode the quad is presently
            params (list): list of [ start_points[x,y,z,yaw], 
                                     end_points[x,y,z,yaw], 
                                     completion_duration, num_steps ]
        """
        self.mode = mode
        self.params = params
        self.final_state = 0
        self.final_time = 0
        self.start_time = start_time
        self.max_iters = 0
        self.time_step = 0.005 # seconds
        self.time_vec = 0

        # get the required sim times from the given params
        self.time_params()

    # class will contain method to generate trajs according to mode and params (traj start, end and time conditions)
    # class will have control loop to execute trajectory
    # class will keep track of actual and desired states
    # class will have function to fetch the above states
    # class will display the current state the user has put it in
    # class will have final vizualization step to simulate all state phases together

    def traj_generator(self):
        start_points, end_points, duration, num_steps = self.params

        if self.mode == 0:
            waypoints, waypoint_times = self.generate_null_waypoints()
            trajectory_state = np.zeros((15, self.max_iteration))
            # height of 15 for: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]
            current_waypoint_number = 0
            for i in range(0, self.max_iteration):
                # update the curr_waypoint_number depending on simulation iteration time
                if current_waypoint_number < len(waypoint_times)-1:
                    if (i*self.time_step) >= waypoint_times[current_waypoint_number+1]:
                        current_waypoint_number +=1

                trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
                trajectory_state[8,i] = waypoints[3, current_waypoint_number]
            return trajectory_state
            
        elif self.mode == 1:
            waypoints, waypoint_times = self.generate_null_waypoints()
            trajectory_state = np.zeros((15, self.max_iteration))
            current_waypoint_number = 0
            z_vel = 0
            z_dist = 0
            const_acc = 0.08 # m/s^2

            # update the curr_waypoint_number depending on simulation iteration time
            for i in range(0,self.max_iteration):
                if current_waypoint_number < len(waypoint_times)-1:
                    if (i*self.time_step) >= waypoint_times[current_waypoint_number+1]:
                        current_waypoint_number +=1

                # update the state's yaw value
                trajectory_state[8,i] = waypoints[3, current_waypoint_number]

                # we create a velocity profile by fixing a constant acceleration
                # use const_acc from lookup_waypoints

                # const positive accel upwards
                if current_waypoint_number < num_steps: #int((len(waypoint_times)-1)/2):
                    # v = u + a*t
                    z_vel = z_vel + const_acc*self.time_step
                    # d = s*t
                    z_dist = z_dist + z_vel*self.time_step
                    trajectory_state[-1,i] = const_acc
                
                trajectory_state[2, i] = z_dist
                trajectory_state[5, i] = z_vel

            return (trajectory_state)
        
        elif self.mode == 2:
            pass
    
    def time_params(self):
        # Set the simualation parameters
        self.final_time = self.start_time + self.params[2] # self.params[2] = duration
        self.time_vec = np.arange(self.start_time, self.final_time, self.time_step).tolist()
        self.max_iteration = len(self.time_vec)
        
    def state_storage(self, states_dict):
        pass

    def generate_null_waypoints(self):
        start_points, end_points, duration, num_steps = self.params
        waypoints = np.array([np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)])
        waypoint_times = np.arange(start=0, stop=duration, step=(duration/num_steps))
        return([waypoints, waypoint_times])