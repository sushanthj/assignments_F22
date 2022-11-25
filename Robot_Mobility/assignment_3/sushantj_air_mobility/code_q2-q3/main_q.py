#!/usr/bin/env python3
"""
Air Mobility Project- 16665
Author: Shruti Gangopadhyay (sgangopa), Rathn Shah(rsshah)
"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

from distutils.log import error
from locale import currency

from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

from utils import plot_state_error, plot_position_3d, plot_des_vs_track
from waypoints_traj import lookup_waypoints, trajectory_planner

thrust_array = []

def position_controller(current_state,desired_state,params,question, time_step):
    '''
    Input parameters:
  
    current_state: The current state of the robot with the following fields:
    current_state["pos"] = [x, y, z],
    current_state["vel"] = [x_dot, y_dot, z_dot]
    current_state["rot"] = [phi, theta, psi]
    current_state["omega"] = [phidot, thetadot, psidot]
    current_state["rpm"] = [w1, w2, w3, w4]
 
    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z] 
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]
 
    params: Quadcopter parameters
 
    question: Question number
 
    Output parameters
 
    F: u1 or thrust
 
    acc: will be stored as desired_state["acc"] = [xdotdot, ydotdot, zdotdot]
    '''

    '''
    store the values of state params    
    '''
    x,y,z = current_state["pos"]
    x_dot, y_dot, z_dot = current_state["vel"]
    phi, thetha, psi = current_state["rot"]
    phi_dot, thetha_dot, psi_dot = current_state["omega"]
    w1, w2, w3, w4 = current_state["rpm"]

    x_des, y_des, z_des = desired_state["pos"]
    x_dot_des, y_dot_des, z_dot_des = desired_state["vel"]
    phi_des, thetha_des, psi_des = desired_state["rot"]
    phi_dot_des, thetha_dot_des, psi_dot_des = desired_state["omega"]
    x_dot_dot, y_dot_dot, z_dot_dot = desired_state["acc"]

    # Example PD gains
    # gains for x,y,z respectively
    Kp1 = 17
    Kd1 = 6.6

    Kp2 = 17
    Kd2 = 6.6

    Kp3 = 21
    Kd3 = 9

    # TO DO:
    Kp = np.diag((Kp1, Kp2, Kp3))
    Kd = np.diag((Kd1, Kd2, Kd3))

    err_xyz = (-Kp @ np.array(([[x-x_des],[y-y_des],[z-z_des]]))) \
            - (Kd @ np.array(([[x_dot-x_dot_des],[y_dot-y_dot_des],[z_dot-z_dot_des]])))
    
    des_acc = np.array(([x_dot_dot],[y_dot_dot],[z_dot_dot]))

    print("desired accels are \n", des_acc)
    print("error in xyz accels are \n", err_xyz)


    # find the rotation matrix to map thrust to body frame
    # Rx(phi), Ry(thetha), Rz(psi)
    Rx = np.array(([1,0,0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]))
    Ry = np.array(([np.cos(thetha), 0, np.sin(thetha)], [0,1,0], [-np.sin(thetha), 0, np.cos(thetha)]))
    Rz = np.array(([np.cos(psi), -np.sin(psi), 0],[np.sin(psi), np.cos(psi), 0], [0,0,1]))

    R_eb = Rz @ (Ry @ Rx)
    # R_be = R_eb.T

    gravity_vec = np.array(([0],[0],[params["gravity"]]))
    
    thrust = params["mass"]*(np.array(([0,0,1])) @ (gravity_vec + des_acc + err_xyz))
    print("thrust was", thrust[0])

    return thrust, des_acc + err_xyz, R_eb


def attitude_by_flatness(desired_state,params):
    '''
    Input parameters:
    
    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z]
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]

    params: Quadcopter parameters
    
    Output parameters:
    
    rot: will be stored as desired_state["rot"] = [phi, theta, psi]

    omega: will be stored as desired_state["omega"] = [phidot, thetadot, psidot]
    
    '''

    phi_des, thetha_des, psi_des = desired_state["rot"]
    g = params["gravity"]

    ############### Calculate "rot"
    # phi_thetha_des = np.array(([0],[0]))
    des_acc = desired_state["acc"]

    phi_thetha_des = 1/g*((np.array(([np.sin(psi_des), -np.cos(psi_des)],
                                    [np.cos(psi_des), np.sin(psi_des)]))) @ des_acc[0:2,:])
    phi_thetha_psi_des = np.vstack((phi_thetha_des, psi_des))

    print("phi_theta_psi_des is \n", phi_thetha_psi_des)
    ################

    ################ Calculate "omega"
    psi_dot_des = desired_state["omega"][2]

    # phi_dot_thetha_dot_des = np.array(([0],[0]))
    des_acc_2 = des_acc * psi_dot_des

    # ignoring the jerk terms since trajectory is not differentiable
    phi_dot_thetha_dot_des = 1/g*((np.array(([np.cos(psi_des), np.sin(psi_des)],
                                    [-np.sin(psi_des), np.cos(psi_des)]))) @ des_acc_2[0:2,:])
    phi_dot_thetha_dot_psi_dot_des = np.vstack((phi_dot_thetha_dot_des, psi_dot_des))

    print("phi_dot_thetha_dot_psi_dot_des is \n", phi_dot_thetha_dot_psi_dot_des)
    #################
    
    return [phi_thetha_psi_des[0,0], 
            phi_thetha_psi_des[1,0], 
            phi_thetha_psi_des[2,0]], \
            [phi_dot_thetha_dot_psi_dot_des[0,0],
             phi_dot_thetha_dot_psi_dot_des[1,0],
             phi_dot_thetha_dot_psi_dot_des[2,0]]


def attitude_controller(params, current_state,desired_state,question):
    '''
    Input parameters
 
    current_state: The current state of the robot with the following fields:
    current_state["pos"] = [x, y, z]
    current_state["vel"] = [x_dot, y_dot, z_dot]
    current_state["rot"] = [phi, theta, psi]
    current_state["omega"] = [phidot, thetadot, psidot]
    current_state["rpm"] = [w1, w2, w3, w4]

    desired_state: The desired states are:
    desired_state["pos"] = [x, y, z] 
    desired_state["vel"] = [x_dot, y_dot, z_dot]
    desired_state["rot"] = [phi, theta, psi]
    desired_state["omega"] = [phidot, thetadot, psidot]
    desired_state["acc"] = [xdotdot, ydotdot, zdotdot]

    params: Quadcopter parameters

    question: Question number

    Output parameters:

    M: u2 or moment [M1, M2, M3]
    '''
    # Example PD gains
    Kpphi = 190
    Kdphi = 30

    Kptheta = 198
    Kdtheta = 30

    Kppsi = 80
    Kdpsi = 17.88

    # TO DO:
    Kp_mat = np.diag((Kpphi, Kptheta, Kppsi))
    Kd_mat = np.diag((Kdphi, Kdtheta, Kdpsi))

    phi, thetha, psi = current_state["rot"]
    phi_des, thetha_des, psi_des = desired_state["rot"]

    phi_dot, thetha_dot, psi_dot = current_state["omega"]
    phi_dot_des, thetha_dot_des, psi_dot_des = desired_state["omega"]

    if int(question) == 2 or int(question) == 3:
        zeroth_order_error = np.array((
                                       [phi - phi_des],
                                       [thetha - thetha_des],
                                       [psi - psi_des]))
        first_order_error = np.array((
                                      [phi_dot - phi_dot_des], 
                                      [thetha_dot - thetha_dot_des], 
                                      [psi_dot - psi_dot_des]
                                    ))

        torques = params["inertia"] @ (
                                        (-Kp_mat @ (zeroth_order_error)) - 
                                        (Kd_mat @ first_order_error)
                                       )
        
        print("torques is", torques)
        return torques



def motor_model(F,M,current_state,params): 
    '''
    Input parameters"

    F,M: required force and moment

    current_state["rpm"]: current motor RPM

    params: Quadcopter parameters

    Output parameters:

    F_motor: Actual thrust generated by Quadcopter's Motors

    M_motor: Actual Moment generated by the Quadcopter's Motors

    rpm_dot: Derivative of the RPM
    '''

    # TO DO:
    # using current rpm find actual force and moments
    # using desired force and moments, find desired rpm
    # using desired rpm and current rpm find rpm_dot

    k_m = params["motor_constant"]
    c_t = params["thrust_coefficient"]
    c_q = params["moment_scale"]
    d = params["arm_length"]

    curr_rpm = np.expand_dims(np.array(current_state["rpm"]), axis=1)
    print("current rpm is", curr_rpm)
    mix = np.array((
                  [c_t, c_t, c_t, c_t],
                  [0, d*c_t, 0, -d*c_t],
                  [-d*c_t, 0, d*c_t, 0],
                  [-c_q, c_q, -c_q, c_q]
                ))
    
    mix_inv = np.linalg.inv(mix)

    #? Find actual force and moments
    rpm_squared = curr_rpm*curr_rpm
    act_force_moment = mix @ rpm_squared

    T_motor_act = act_force_moment[1:4,0]
    F_motor_act = act_force_moment[0,0]
    print("T actual is", T_motor_act)

    #? Find desired rpm
    force_torq_mat = np.vstack((np.expand_dims(F, axis=0), M))
    desired_rpm = np.sqrt(mix_inv @ force_torq_mat)

    fixed_desired_rpm = deepcopy(desired_rpm)

    # limit the desired rpm
    for i in range(desired_rpm.shape[0]):
        if desired_rpm[i,0] > 20000:
            fixed_desired_rpm[i,0] = 20000
        else:
            pass

    print("desired rpm is", fixed_desired_rpm)
    
    #? Find rpm_dot
    w_dot = k_m * (fixed_desired_rpm - curr_rpm)
    print("w dot is", w_dot)

    return F_motor_act, T_motor_act, w_dot


def dynamics(t,state,params,F_actual,M_actual,rpm_motor_dot, rot_matrix):

    '''
    Input parameters:
  
    state: current state, will be using RK45 to update
 
    F, M: actual force and moment from motor model
 
    rpm_motor_dot: actual change in RPM from motor model
  
    params: Quadcopter parameters
 
    question: Question number
 
    Output parameters:
 
    state_dot: change in state
    '''
    # TO DO:
    x,y,z,xdot,ydot,zdot,phi,theta,psi,phidot,thetadot,psidot,rpm1, rpm2, rpm3, rpm4 = state

    F_eb = (rot_matrix @ np.array(([[0],[0],[F_actual]]))) - (params["mass"] * np.array(([[0],[0],[params["gravity"]]])))
    # find translational accelerations
    xddot, yddot, zddot = (F_eb / params["mass"])[0,0], (F_eb / params["mass"])[1,0], (F_eb / params["mass"])[2,0]

    # find rotational accelerations
    rot_acc = np.linalg.inv(params["inertia"]) @ M_actual
    phiddot, thetaddot, psiddot = rot_acc[0], rot_acc[1], rot_acc[2]

    state_dot = [xdot, ydot, zdot, xddot, yddot, zddot, phidot, thetadot, psidot, phiddot, thetaddot, psiddot, rpm_motor_dot[0,0], rpm_motor_dot[1,0], rpm_motor_dot[2,0], rpm_motor_dot[3,0]]
    return state_dot



def main(question):

    # Set up quadrotor physical parameters
    params = {"mass": 0.770, "gravity": 9.80665, "arm_length": 0.1103, "motor_spread_angle": 0.925, \
        "thrust_coefficient": 8.07e-9, "moment_scale": 1.3719e-10, "motor_constant": 36.5, "rpm_min": 3000, \
            "rpm_max": 20000, "inertia": np.diag([0.0033,0.0033,0.005]), "COM_vertical_offset": 0.05}
    
    # Get the waypoints for this specific question
    [waypoints, waypoint_times, const_acc] = lookup_waypoints(question)
    # waypoints are of the form [x, y, z, yaw]
    # waypoint_times are the seconds you should be at each respective waypoint
    # make sure the simulation parameters below allow you to get to all points

    # Set the simualation parameters
    time_initial = 0
    time_final = 20
    time_step = 0.005 # in secs
    # 0.005 sec is a reasonable time step for this system
    
    # vector of timesteps
    time_vec = np.arange(time_initial, time_final, time_step).tolist()
    max_iteration = len(time_vec)

    #? Create the state vector (x,y,z,psi_des for 4 states)
    state = np.zeros((16,1))
    # state: [x,y,z,xdot,ydot,zdot,phi,theta,psi,phidot,thetadot,psidot,rpm]

    print(waypoints.shape)
    # Populate the state vector with the first waypoint 
    # (assumes robot is at first waypoint at the initial time)
    state[0] = waypoints[0,0]
    state[1] = waypoints[1,0]
    state[2] = waypoints[2,0]
    state[8] = waypoints[3,0]

    #Create a trajectory consisting of desired state at each time step
    # Some aspects of this state we can plan in advance, some will be filled during the loop
    trajectory_matrix = trajectory_planner(question, waypoints, max_iteration, waypoint_times, time_step, const_acc)
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

    # Create a matrix to hold the actual state at each time step
    actual_state_matrix = np.zeros((15,max_iteration))
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

    #? the quad is assumed to be at 0.5?
    actual_state_matrix[:,0] = np.vstack((state[0:12], np.array([[0],[0],[0]])))[:,0]
    
    # trial outputs
    print("\n waypoints are \n", waypoints)
    print("\n actual state matrix is \n", actual_state_matrix[:,0])
    
    #Create a matrix to hold the actual desired state at each time step

    # Need to store the actual desired state for acc, omega dot, omega as it will be updated by the controller
    actual_desired_state_matrix = np.zeros((15,max_iteration))
    
    # state list created for the RK45 solver
    state_list = state.T.tolist()
    state_list = state_list[0]

    # Loop through the timesteps and update quadrotor
    for i in range(max_iteration-1):
        # break
        # convert current state to stuct for control functions
        current_state = {"pos":state_list[0:3],"vel":state_list[3:6],"rot":state_list[6:9], \
            "omega":state_list[9:12],"rpm":state_list[12:16]}
        # print("current state is \n", current_state)
        
        # Get desired state from matrix, put into struct for control functions
        desired_state = {"pos":trajectory_matrix[0:3,i],"vel":trajectory_matrix[3:6,i],\
            "rot":trajectory_matrix[6:9,i],"omega":trajectory_matrix[9:12,i],"acc":trajectory_matrix[12:15,i]}
        # print("desired state is \n", desired_state)
        
        # Get desired acceleration from position controller
        [F, desired_state["acc"], rot_matrix] = position_controller(current_state,desired_state,params,question, time_step)
        print("")
        
        # Computes desired pitch and roll angles
        desired_state["rot"], desired_state["omega"] = attitude_by_flatness(desired_state,params)        
        
        # Get torques from attitude controller
        M = attitude_controller(params,current_state,desired_state,question)
        
        # Motor model
        F_actual, M_actual, rpm_motor_dot = motor_model(F,M,current_state,params)
        thrust_array.append(F_actual)
        
        # Get the change in state from the quadrotor dynamics
        time_int = tuple((time_vec[i],time_vec[i+1]))

        sol = solve_ivp(dynamics,time_int,state_list,args=(params,F_actual,M_actual,rpm_motor_dot, rot_matrix),t_eval=np.linspace(time_vec[i],time_vec[i+1],(int(time_step/0.00005))))
        
        state_list = sol.y[:,-1]
        acc = (sol.y[3:6,-1]-sol.y[3:6,-2])/(sol.t[-1]-sol.t[-2])
        
        # Update desired state matrix (15 x N numpy array)
        actual_desired_state_matrix[0:3,i+1] = desired_state["pos"]
        actual_desired_state_matrix[3:6,i+1] = desired_state["vel"]
        print("rot is", desired_state["omega"])
        actual_desired_state_matrix[6:9,i+1] = desired_state["rot"][:]
        actual_desired_state_matrix[9:12,i+1] = desired_state["omega"][:]
        actual_desired_state_matrix[12:15,i+1] = desired_state["acc"][:,0]

        # Update actual state matrix (16 x N numpy array)
        actual_state_matrix[0:12,i+1] = state_list[0:12]
        actual_state_matrix[12:15,i+1] = acc
    
    # plot for values and errors
    plot_state_error(actual_state_matrix,actual_desired_state_matrix,time_vec)

    # plot for 3d visualization
    plot_position_3d(actual_state_matrix,actual_desired_state_matrix)

    # plot desired pose vs actual pose
    plot_des_vs_track(actual_state_matrix, actual_desired_state_matrix, time_vec, thrust_array)
        
        
if __name__ == '__main__':
    '''
    Usage: main takes in a question number and executes all necessary code to
    construct a trajectory, plan a path, simulate a quadrotor, and control
    the model. Possible arguments: 2, 3, 5, 6.2, 6.3, 6.5, 7, 9. THE
    TAS WILL BE RUNNING YOUR CODE SO PLEASE KEEP THIS MAIN FUNCTION CALL 
    STRUCTURE AT A MINIMUM.
    '''
    # run the file with command "python3 main.py question_number" in the terminal
    question = sys.argv[1]
    main(question)