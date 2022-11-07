import matplotlib.pyplot as plt

def plot_state_error(state,state_des,time_vector):

    # actual states
    pos = state[0:3,:]
    vel = state[3:6,:]
    rpy = state[6:9,:]
    ang_vel = state[9:12,:]
    acc = state[12:15,:]

    # desired states
    pos_des = state_des[0:3,:]
    vel_des = state_des[3:6,:]
    rpy_des = state_des[6:9,:]
    ang_vel_des = state_des[9:12,:]
    acc_des = state_des[12:15,:]

    # get error from des and act
    error_pos = pos - pos_des
    error_vel = vel - vel_des
    error_rpy = rpy - rpy_des
    error_ang_vel = ang_vel - ang_vel_des
    error_acc = acc- acc_des

    # plot erros
    fig = plt.figure(1)
    # plot position error
    axs = fig.subplots(5,3)
    axs[0,0].plot(time_vector,error_pos[0,:])
    axs[0,0].set_title("Error in x")
    axs[0,0].set(xlabel = 'time(s)', ylabel = 'x(m)')
    axs[0,1].plot(time_vector,error_pos[1,:])
    axs[0,1].set_title("Error in y")
    axs[0,1].set(xlabel = 'time(s)', ylabel = 'y(m)')
    axs[0,2].plot(time_vector,error_pos[2,:])
    axs[0,2].set_title("Error in z")
    axs[0,2].set(xlabel = 'time(s)', ylabel = 'z(m)')

    # plot orientation error
    axs[1,0].plot(time_vector,error_rpy[0,:])
    axs[1,0].set_title("Error in phi")
    axs[1,0].set(xlabel = 'time(s)', ylabel = 'phi')
    axs[1,1].plot(time_vector,error_rpy[1,:])
    axs[1,1].set_title("Error in theta")
    axs[1,1].set(xlabel = 'time(s)', ylabel = 'theta')
    axs[1,2].plot(time_vector,error_rpy[2,:])
    axs[1,2].set_title("Error in psi")
    axs[1,2].set(xlabel = 'time(s)', ylabel = 'psi')

    # plot velocity error
    axs[2,0].plot(time_vector,error_vel[0,:])
    axs[2,0].set_title("Error in vx")
    axs[2,0].set(xlabel = 'time(s)', ylabel = 'vx (m/s)')
    axs[2,1].plot(time_vector,error_vel[1,:])
    axs[2,1].set_title("Error in vy")
    axs[2,1].set(xlabel = 'time(s)', ylabel = 'vy (m/s)')
    axs[2,2].plot(time_vector,error_vel[2,:])
    axs[2,2].set_title("Error in vz")
    axs[2,2].set(xlabel = 'time(s)', ylabel = 'vz (m/s)')

    # plot angular velocity error
    axs[3,0].plot(time_vector,error_ang_vel[0,:])
    axs[3,0].set_title("Error in omega_x")
    axs[3,0].set(xlabel = 'time(s)', ylabel = 'omega_x (rad/s)')
    axs[3,1].plot(time_vector,error_ang_vel[1,:])
    axs[3,1].set_title("Error in omega_y")
    axs[3,1].set(xlabel = 'time(s)', ylabel = 'omega_y (rad/s)')
    axs[3,2].plot(time_vector,error_ang_vel[2,:])
    axs[3,2].set_title("Error in omega_z")
    axs[3,2].set(xlabel = 'time(s)', ylabel = 'omega_z (rad/s)')

    # plot acceleration error
    axs[4,0].plot(time_vector,error_acc[0,:])
    axs[4,0].set_title("Error in acc_x")
    axs[4,0].set(xlabel = 'time(s)', ylabel = 'acc_x (m/s2)')
    axs[4,1].plot(time_vector,error_acc[1,:])
    axs[4,1].set_title("Error in acc_y")
    axs[4,1].set(xlabel = 'time(s)',ylabel = 'acc_y (m/s2)')
    axs[4,2].plot(time_vector,error_acc[2,:])
    axs[4,2].set_title("Error in acc_z")
    axs[4,2].set(xlabel = 'time(s)', ylabel = 'acc_z (m/s2)')

    fig.tight_layout(pad = 0.005)

    # plot values
    fig1 = plt.figure(2)
    # plot position
    axs1 = fig1.subplots(5,3)
    axs1[0,0].plot(time_vector,pos[0,:])
    axs1[0,0].set_title("x")
    axs1[0,0].set(xlabel = 'time(s)', ylabel = 'x(m)')
    axs1[0,1].plot(time_vector,pos[1,:])
    axs1[0,1].set_title("y")
    axs1[0,1].set(xlabel = 'time(s)', ylabel = 'y(m)')
    axs1[0,2].plot(time_vector,pos[2,:])
    axs1[0,2].set_title("z")
    axs1[0,2].set(xlabel = 'time(s)', ylabel = 'z(m)')

    # plot orientation 
    axs1[1,0].plot(time_vector,rpy[0,:])
    axs1[1,0].set_title("phi")
    axs1[1,0].set(xlabel = 'time(s)', ylabel = 'phi')
    axs1[1,1].plot(time_vector,rpy[1,:])
    axs1[1,1].set_title("theta")
    axs1[1,1].set(xlabel = 'time(s)', ylabel = 'theta')
    axs1[1,2].plot(time_vector,rpy[2,:])
    axs1[1,2].set_title("psi")
    axs1[1,2].set(xlabel = 'time(s)', ylabel = 'psi')

    # plot velocity 
    axs1[2,0].plot(time_vector,vel[0,:])
    axs1[2,0].set_title("vx")
    axs1[2,0].set(xlabel = 'time(s)', ylabel = 'vx (m/s)')
    axs1[2,1].plot(time_vector,vel[1,:])
    axs1[2,1].set_title("vy")
    axs1[2,1].set(xlabel = 'time(s)', ylabel = 'vy (m/s)')
    axs1[2,2].plot(time_vector,vel[2,:])
    axs1[2,2].set_title("vz")
    axs1[2,2].set(xlabel = 'time(s)', ylabel = 'vz (m/s)')

    # plot angular velocity
    axs1[3,0].plot(time_vector,ang_vel[0,:])
    axs1[3,0].set_title("omega_x")
    axs1[3,0].set(xlabel = 'time(s)', ylabel = 'omega_x (rad/s)')
    axs1[3,1].plot(time_vector,ang_vel[1,:])
    axs1[3,1].set_title("omega_y")
    axs1[3,1].set(xlabel = 'time(s)', ylabel = 'omega_y (rad/s)')
    axs1[3,2].plot(time_vector,ang_vel[2,:])
    axs1[3,2].set_title("omega_z")
    axs1[3,2].set(xlabel = 'time(s)', ylabel = 'omega_z (rad/s)')

    # plot acceleration 
    axs1[4,0].plot(time_vector,acc[0,:])
    axs1[4,0].set_title("acc_x")
    axs1[4,0].set(xlabel = 'time(s)', ylabel = 'acc_x (m/s2)')
    axs1[4,1].plot(time_vector,acc[1,:])
    axs1[4,1].set_title("acc_y")
    axs1[4,1].set(xlabel = 'time(s)',ylabel = 'acc_y (m/s2)')
    axs1[4,2].plot(time_vector,acc[2,:])
    axs1[4,2].set_title("acc_z")
    axs1[4,2].set(xlabel = 'time(s)', ylabel = 'acc_z (m/s2)')

    fig1.tight_layout(pad=0.05)
    plt.show()


# Helper to visualize positions for the flight of the quadrotor
def plot_position_3d(state, state_des):
    pos = state[0:3,:]
    pos_des = state_des[0:3,:]
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos[0,:], pos[1,:], pos[2,:], color='blue', 
            label='Actual position')
    ax.plot(pos_des[0,:], pos_des[1,:], pos_des[2,:], color='red',
            label='Desired position')

    ax.set(xlabel = 'x (m)')
    ax.set(ylabel = 'y (m)')
    ax.set(zlabel = 'z (m)')
    ax.set_title('Position')
    ax.legend()

    ax.axes.set_xlim3d(left=-.5, right=.5)
    ax.axes.set_ylim3d(bottom=-.5, top=.5)
    ax.axes.set_zlim3d(bottom=0, top=.5)

    plt.show()
