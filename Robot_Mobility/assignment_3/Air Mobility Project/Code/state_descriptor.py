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
        self.final_state = None
        self.final_time = 0
        self.start_time = start_time
        self.max_iters = 0
        self.time_step = 0.005 # seconds
        self.time_vec = 0
        
        self.start = self.params[0]
        self.end = self.params[1]

    # class will contain method to generate trajs according to mode and params 
    #  which are (traj start, end and time conditions)
    # class will have control loop to execute trajectory
    # class will keep track of actual and desired states
    # class will have function to fetch the above states
    # class will display the current state the user has put it in
    # class will have final vizualization step to simulate all state phases together

    def traj_generator(self):
        start_points, end_points, duration, num_steps = self.params

        if self.mode == 0:
            waypoints, waypoint_times = self.generate_waypoints(0)
            trajectory_state = np.zeros((15, self.max_iteration))
            # [x, y, z, xdot, ydot, zdot, phi, theta, psi, 
            #  phidot, thetadot, psidot, xacc, yacc, zacc]
            trajectory_state, waypoints, = self.generate_trajectory(waypoint_times, 
                                                                    waypoints, 
                                                                    trajectory_state)
            
            return trajectory_state, waypoints
            
        elif self.mode == 1 or self.mode == 4:
            
            dist_step = (self.end[2] - self.start[2])/num_steps

            z_vals = np.arange(start=self.start[2], stop=self.end[2]+dist_step, step=dist_step)

            waypoints = np.array([np.ones(shape=z_vals.shape)*self.start[0], 
                                    np.ones(shape=z_vals.shape)*self.start[1], 
                                    z_vals, 
                                    np.ones(shape=z_vals.shape)*self.start[3]])
            
            print("waypoints in state descp class are ", waypoints.shape)
            step_size = (duration/num_steps)
            waypoint_times = np.arange(start=self.start_time, 
                                        stop=self.final_time+step_size, 
                                        step=step_size)
            print("waypoint times is", waypoint_times.shape)
            
            trajectory_state = np.zeros((15, self.max_iteration))
            #[x, y, z, xdot, ydot, zdot, phi, theta, psi, 
            # phidot, thetadot, psidot, xacc, yacc, zacc]
            trajectory_state, waypoints, = self.generate_trajectory(waypoint_times, 
                                                                    waypoints, 
                                                                    trajectory_state)

            return trajectory_state, waypoints
        
        elif self.mode == 2:
            if end_points != start_points:
                print("Wrong conditions for hover given")
                return 0
            else:
                waypoints, waypoint_times = self.generate_waypoints(start_points[2])
                trajectory_state = np.zeros((15, self.max_iteration))
                #[x, y, z, xdot, ydot, zdot, phi, theta, psi, 
                # phidot, thetadot, psidot, xacc, yacc, zacc]
                trajectory_state, waypoints, = self.generate_trajectory(waypoint_times, 
                                                                        waypoints, 
                                                                        trajectory_state)
                return trajectory_state, waypoints
        
        elif self.mode == 3:
            print("start and end are", self.start, self.end)
            changed_states = np.where(np.array(self.start) != np.array(self.end))
            print("changed states are", changed_states)
            # create dummy values
            vals = [np.ones(shape=num_steps+1)*self.start[0],
                    np.ones(shape=num_steps+1)*self.start[1], 
                    np.ones(shape=num_steps+1)*self.start[2],  
                    np.ones(shape=num_steps+1)*self.start[3]]

            for i in changed_states[0].tolist():    
                dist_step = (self.end[i] - self.start[i])/num_steps
                vals[i] = np.arange(start=self.start[i], 
                                         stop=self.end[i]+dist_step, 
                                         step=dist_step)
            
            waypoints = np.array([vals[0], vals[1], vals[2], vals[3]])

            print("waypoints in state descp class are ", waypoints)
            step_size = duration/num_steps
            waypoint_times = np.arange(start=self.start_time, 
                                        stop=self.final_time+step_size, 
                                        step=step_size)
            print("waypoint times is", waypoint_times.shape)
            
            trajectory_state = np.zeros((15, self.max_iteration))
            #[x, y, z, xdot, ydot, zdot, phi, theta, psi, 
            # phidot, thetadot, psidot, xacc, yacc, zacc]
            trajectory_state, waypoints, = self.generate_trajectory(waypoint_times, 
                                                                    waypoints, 
                                                                    trajectory_state)

            return trajectory_state, waypoints
    
    def time_params(self):
        # Set the simualation parameters
        self.final_time = self.start_time + self.params[2] # self.params[2] = duration
        self.time_vec = np.arange(self.start_time, self.final_time, self.time_step).tolist()
        self.max_iteration = len(self.time_vec)
        # print("time vec start and stop is ", self.time_vec[0], "  ", self.time_vec[-1])

        return self.start_time, self.final_time, self.time_step, self.time_vec, self.max_iteration
        
    def state_storage(self, states_dict):
        self.final_time = states_dict["finish_time"]
        self.final_state = states_dict

    def generate_waypoints(self, const):
        start_points, end_points, duration, num_steps = self.params
        waypoints = np.array(
                            [np.zeros(num_steps), 
                             np.zeros(num_steps), 
                             np.ones(num_steps)*const, 
                             np.zeros(num_steps)
                            ])
        step_size = (duration/num_steps)
        waypoint_times = np.arange(start=self.start_time, 
                                   stop=self.final_time+step_size, 
                                   step=step_size)
        print("waypoints in state descp class are ", waypoints.shape)
        print("waypoint times is", waypoint_times.shape)
        return([waypoints, waypoint_times])

    def generate_trajectory(self, waypoint_times, waypoints, trajectory_state):
        current_waypoint_number = 0
        for i in range(0, self.max_iteration):
            # update the curr_waypoint_number depending on simulation iteration time
            if current_waypoint_number < len(waypoint_times)-1:
                if (i*self.time_step) >= (waypoint_times[current_waypoint_number+1] 
                                            - self.start_time):
                    current_waypoint_number +=1

            trajectory_state[0:3, i] = waypoints[0:3, current_waypoint_number]
            trajectory_state[8,i] = waypoints[3, current_waypoint_number]
        return trajectory_state, waypoints