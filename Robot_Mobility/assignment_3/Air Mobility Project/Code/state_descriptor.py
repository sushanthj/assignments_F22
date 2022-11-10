class StateDescriptor:
    """
    Runs a user defined state of the quad
    """
    def __init__(self, mode, params):
        """
        Accept the boundaries of the state
        Args:
            mode (int): specify which mode the quad is presently
            params (list): list of [ start_points[x,y,z,yaw], end_points[x,y,z,yaw], completion_time ]
        """
        self.mode = mode
        self.params = params
        self.final_state = 0

    # class will contain method to generate trajs according to mode and params (traj start, end and time conditions)
    # class will have control loop to execute trajectory
    # class will keep track of actual and desired states
    # class will have function to fetch the above states
    # class will display the current state the user has put it in
    # class will have final vizualization step to simulate all state phases together

    def traj_generator(self):
        if self.mode == 0:
            pass # init all states to zero