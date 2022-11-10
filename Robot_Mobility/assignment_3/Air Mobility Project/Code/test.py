import numpy as np

a = np.zeros(3)

print(a)

"""
for i in range(0,max_iteration):
            if current_waypoint_number < len(waypoint_times)-1:
                if (i*time_step) >= waypoint_times[current_waypoint_number+1]:
                    current_waypoint_number +=1

            # update the state's yaw value
            trajectory_state[8,i] = waypoints[3, current_waypoint_number]

            # we create a velocity profile by fixing a constant acceleration
            # use const_acc from lookup_waypoints
            trajectory_state[5,i] = waypoints[4, current_waypoint_number]

            # const positive accel upwards
            if current_waypoint_number < 100: #int((len(waypoint_times)-1)/2):
                # s = d*t
                z_dist = z_vel*waypoint_times[current_waypoint_number]
                trajectory_state[-1,i] = const_acc
            elif current_waypoint_number == 100: #int((len(waypoint_times)-1)/2):
                # s = d*t
                z_dist = z_dist
                trajectory_state[-1,i] = 0
            elif (current_waypoint_number > 100) and (current_waypoint_number < 201): #int((len(waypoint_times)-1)/2):
                # s = d*t
                z_dist = z_vel - const_acc*time_step
                trajectory_state[-1,i] = -const_acc
            else:
                z_vel = 0
                trajectory_state[-1,i] = 0

    return (trajectory_state)
"""