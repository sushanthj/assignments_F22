import numpy as np

a = np.zeros(3)

print(a)

"""
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
"""