#!/usr/bin/env python3
"""
Air Mobility Project- 16665
Author: Shruti Gangopadhyay (sgangopa), Rathn Shah(rsshah)
"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import state_descriptor

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

    Kp3 = 20
    Kd3 = 9

    # TO DO:
    Kp = np.diag((Kp1, Kp2, Kp3))
    Kd = np.diag((Kd1, Kd2, Kd3))

    err_xyz = (-Kp @ np.array(([[x-x_des],[y-y_des],[z-z_des]]))) \
            - (Kd @ np.array(([[x_dot-x_dot_des],[y_dot-y_dot_des],[z_dot-z_dot_des]])))
    
    des_acc = np.array(([x_dot_dot],[y_dot_dot],[z_dot_dot]))

    # print("desired accels are \n", des_acc)
    # print("error in xyz accels are \n", err_xyz)


    # find the rotation matrix to map thrust to body frame
    # Rx(phi), Ry(thetha), Rz(psi)
    Rx = np.array(([1,0,0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]))
    Ry = np.array(([np.cos(thetha), 0, np.sin(thetha)], 
                   [0,1,0], 
                   [-np.sin(thetha), 0, np.cos(thetha)]))
    Rz = np.array(([np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0], 
                   [0,0,1]))

    R_eb = Rz @ (Ry @ Rx)
    # R_be = R_eb.T

    gravity_vec = np.array(([0],[0],[params["gravity"]]))
    
    thrust = params["mass"]*(np.array(([0,0,1])) @ (gravity_vec + des_acc + err_xyz))
    # print("thrust was", thrust[0])

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

    # print("phi_theta_psi_des is \n", phi_thetha_psi_des)
    ################

    ################ Calculate "omega"
    psi_dot_des = desired_state["omega"][2]

    # phi_dot_thetha_dot_des = np.array(([0],[0]))
    des_acc_2 = des_acc * psi_dot_des

    # ignoring the jerk terms since trajectory is not differentiable
    phi_dot_thetha_dot_des = 1/g*((np.array(([np.cos(psi_des), np.sin(psi_des)],
                                    [-np.sin(psi_des), np.cos(psi_des)]))) @ des_acc_2[0:2,:])
    phi_dot_thetha_dot_psi_dot_des = np.vstack((phi_dot_thetha_dot_des, psi_dot_des))

    # print("phi_dot_thetha_dot_psi_dot_des is \n", phi_dot_thetha_dot_psi_dot_des)
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

    Kptheta = 190
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

    if True:
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
        
        # print("torques is", torques)
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
    # print("current rpm is", curr_rpm)
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
    # print("T actual is", T_motor_act)

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

    # print("desired rpm is", fixed_desired_rpm)
    
    #? Find rpm_dot
    w_dot = k_m * (fixed_desired_rpm - curr_rpm)
    # print("w dot is", w_dot)

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

    F_eb = (rot_matrix @ np.array(([[0],[0],[F_actual]]))) - \
            (params["mass"] * np.array(([[0],[0],[params["gravity"]]])))
    
    # find translational accelerations
    xddot, yddot, zddot = (F_eb / params["mass"])[0,0], \
                          (F_eb / params["mass"])[1,0], \
                          (F_eb / params["mass"])[2,0]

    # find rotational accelerations
    rot_acc = np.linalg.inv(params["inertia"]) @ M_actual
    phiddot, thetaddot, psiddot = rot_acc[0], rot_acc[1], rot_acc[2]

    state_dot = [
                 xdot,
                 ydot,
                 zdot,
                 xddot,
                 yddot, 
                 zddot, 
                 phidot, 
                 thetadot, 
                 psidot, 
                 phiddot, 
                 thetaddot, 
                 psiddot, 
                 rpm_motor_dot[0,0], 
                 rpm_motor_dot[1,0], 
                 rpm_motor_dot[2,0], 
                 rpm_motor_dot[3,0]
                ]
    return state_dot



def main(question, state_descp):
    """
    Main control loop for given start and end states

    Args:
        question    : question number from assignment
        state_descp : object from state machine

    Returns:
        actual_and_desired_states: tracking history of states
    """
    # Set up quadrotor physical parameters
    params = {
              "mass": 0.770, 
              "gravity": 9.80665, 
              "arm_length": 0.1103, 
              "motor_spread_angle": 0.925,
              "thrust_coefficient": 8.07e-9, 
              "moment_scale": 1.3719e-10, 
              "motor_constant": 36.5, 
              "rpm_min": 3000,
              "rpm_max": 20000, 
              "inertia": np.diag([0.0033,0.0033,0.005]), 
              "COM_vertical_offset": 0.05
              }
    
    # Get the waypoints for this specific question
    [waypoints, waypoint_times, const_acc] = lookup_waypoints(question)
    # waypoints are of the form [x, y, z, yaw]
    # waypoint_times are the seconds you should be at each respective waypoint
    # make sure the simulation parameters below allow you to get to all points

    # Set the simualation parameters
    time_initial = 0
    time_final = 20.1
    finish_time = time_final
    time_step = 0.005 # in secs
    # 0.005 sec is a reasonable time step for this system
    
    # vector of timesteps
    time_vec = np.arange(time_initial, time_final, time_step).tolist()
    max_iteration = len(time_vec)
    finish_time_iteration = max_iteration-1

    #? Create the state vector (x,y,z,psi_des for 4 states)
    state = np.zeros((16,1))
    # state: [x,y,z,xdot,ydot,zdot,phi,theta,psi,phidot,thetadot,psidot,rpm]

    # Populate the state vector with the first waypoint 
    # (assumes robot is at first waypoint at the initial time)
    state[0] = waypoints[0,0]
    state[1] = waypoints[1,0]
    state[2] = waypoints[2,0]
    state[8] = waypoints[3,0]

    #Create a trajectory consisting of desired state at each time step
    # Some aspects of this state we can plan in advance, some will be filled during the loop
    trajectory_matrix = trajectory_planner(
                                            question, 
                                            waypoints, 
                                            max_iteration, 
                                            waypoint_times, 
                                            time_step, 
                                            const_acc
                                            )
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

    if int(question) > 3:
        time_initial, time_final, time_step, time_vec, max_iteration = state_descp.time_params()
        finish_time_iteration = max_iteration-1
        # get trajectory from the state_descriptor class
        trajectory_matrix, waypoints = state_descp.traj_generator()
        state[0] = waypoints[0,0]
        state[1] = waypoints[1,0]
        state[2] = waypoints[2,0]
        state[8] = waypoints[3,0]

    # Create a matrix to hold the actual state at each time step
    actual_state_matrix = np.zeros((15,max_iteration))
    # [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot, xacc, yacc, zacc]

    #? the quad is assumed to be at 0.5?
    actual_state_matrix[:,0] = np.vstack((state[0:12], np.array([[0],[0],[0]])))[:,0]
    
    #Create a matrix to hold the actual desired state at each time step

    # Need to store the actual desired state for acc, omega dot, 
    # omega as it will be updated by the controller
    actual_desired_state_matrix = np.zeros((15,max_iteration))
    actual_desired_state_matrix[:,0] = np.vstack((state[0:12], np.array([[0],[0],[0]])))[:,0]
    
    # state list created for the RK45 solver
    state_list = state.T.tolist()
    state_list = state_list[0]

    # Loop through the timesteps and update quadrotor
    for i in range(max_iteration-1):
        # break
        # convert current state to stuct for control functions
        current_state = {
                        "pos":state_list[0:3],
                        "vel":state_list[3:6],
                        "rot":state_list[6:9],
                        "omega":state_list[9:12],
                        "rpm":state_list[12:16]
                        }
        # print("current state is \n", current_state)
        
        # Get desired state from matrix, put into struct for control functions
        desired_state = {
                        "pos":trajectory_matrix[0:3,i],
                        "vel":trajectory_matrix[3:6,i],
                        "rot":trajectory_matrix[6:9,i],
                        "omega":trajectory_matrix[9:12,i],
                        "acc":trajectory_matrix[12:15,i]
                        }
        # print("desired state is \n", desired_state)
        
        # Get desired acceleration from position controller
        [F, desired_state["acc"], rot_matrix] = position_controller(
                                                                    current_state,
                                                                    desired_state,
                                                                    params,question,
                                                                    time_step
                                                                    )
        
        # Computes desired pitch and roll angles
        desired_state["rot"], desired_state["omega"] = attitude_by_flatness(desired_state,params)        
        
        # Get torques from attitude controller
        M = attitude_controller(params,current_state,desired_state,question)
        
        # Motor model
        F_actual, M_actual, rpm_motor_dot = motor_model(F,M,current_state,params)
        thrust_array.append(F_actual)
        
        # Get the change in state from the quadrotor dynamics
        time_int = tuple((time_vec[i],time_vec[i+1]))

        sol = solve_ivp(
                        dynamics,
                        time_int,
                        state_list,
                        args=(params,F_actual,M_actual,rpm_motor_dot, rot_matrix),
                        t_eval=np.linspace(time_vec[i],time_vec[i+1],(int(time_step/0.00005)))
                        )
        
        state_list = sol.y[:,-1]
        acc = (sol.y[3:6,-1]-sol.y[3:6,-2])/(sol.t[-1]-sol.t[-2])
        
        # Update desired state matrix (15 x N numpy array)
        actual_desired_state_matrix[0:3,i+1] = desired_state["pos"]
        actual_desired_state_matrix[3:6,i+1] = desired_state["vel"]
        actual_desired_state_matrix[6:9,i+1] = desired_state["rot"][:]
        actual_desired_state_matrix[9:12,i+1] = desired_state["omega"][:]
        actual_desired_state_matrix[12:15,i+1] = desired_state["acc"][:,0]

        # Update actual state matrix (16 x N numpy array)
        actual_state_matrix[0:12,i+1] = state_list[0:12]
        actual_state_matrix[12:15,i+1] = acc

        if int(question) > 3:
            convergence = check_convergence(state_descp, actual_state_matrix[:,i+1])
            if convergence is True:
                finish_time_iteration = (i+2)
                finish_time = (i+2)*time_step + time_initial
                actual_state_matrix = actual_state_matrix[:,0:i+2]
                actual_desired_state_matrix = actual_desired_state_matrix[:,0:i+2]
                print("breaking iteration was", i+2)
                break
            else:
                continue

    if int(question) < 4:
        # calculate rise time and overshoot
        calcuate_rise_and_overshoot_for_z(actual_state_matrix, time_vec)
        # calcuate_rise_and_overshoot_for_psi(actual_state_matrix, time_vec)

        # plot for values and errors
        plot_state_error(actual_state_matrix,actual_desired_state_matrix,time_vec)

        # plot for 3d visualization
        plot_position_3d(actual_state_matrix,actual_desired_state_matrix)

        # plot desired pose vs actual pose
        plot_des_vs_track(actual_state_matrix, actual_desired_state_matrix, time_vec, thrust_array)
    else:
        if convergence is False:
            print("did not converge")
            finish_time_iteration = max_iteration
            finish_time = (max_iteration)*time_step + time_initial
            actual_state_matrix = actual_state_matrix
            actual_desired_state_matrix = actual_desired_state_matrix
            print("breaking iteration was", max_iteration)
        return {
                "actual_states" : actual_state_matrix, 
                "desired_states" : actual_desired_state_matrix, 
                "time_vec": time_vec, 
                "finish_time_iter": finish_time_iteration, 
                "finish_time": finish_time,
                "time_initial": time_initial, 
                "time_step": time_step
                }

def state_machine(question, traj):
    """
    inherit the class which will define boundaries of the states
    params (list): list of [ start_points[x,y,z,yaw], 
                                     end_points[x,y,z,yaw], 
                                     completion_duration, num_steps ]
    """

    print("executing mode 0: Idle")
    mode_0_params = [ [0,0,0,0], [0,0,0,0], 2, 2]
    mode_0 = state_descriptor.StateDescriptor(0, mode_0_params, 0)
    # since quad is idle, save states directly
    states_saved = store_idle_pos(mode_0)
    mode_0.state_storage(states_saved)
    print("\n")

    print("executing mode 1: Takeoff")
    mode_1_params = [ [0,0,0,0], [0,0,1,0], 20, 100]
    mode_1 = state_descriptor.StateDescriptor(1, mode_1_params, mode_0.final_time)
    # track quad and save the actual vs desired states
    states_saved = main(question, mode_1)
    mode_1.state_storage(states_saved)
    print("\n")

    print("executing mode 2: Hover")
    mode_2_params = [ [0,0,1,0], [0,0,1,0], 5, 5]
    mode_2 = state_descriptor.StateDescriptor(2, mode_2_params, mode_1.final_time)
    # track quad and save the actual vs desired states
    states_saved = store_idle_pos(mode_2)
    mode_2.state_storage(states_saved)
    print("\n")

    print("executing mode 3: Trajectory")
    # Note: the start pos of this mode should always be same as the end pos of mode_2
    mode_3_params = traj
    mode_3 = state_descriptor.StateDescriptor(3, mode_3_params, mode_2.final_time)
    # track quad and save the actual vs desired states
    states_saved = main(question, mode_3)
    mode_3.state_storage(states_saved)
    print("\n")

    print("executing mode 4: Landing")
    mode_3_final_pos = mode_3_params[1]
    mode_4_start_pos = deepcopy(mode_3_final_pos)
    print("start pos is",mode_3_final_pos)
    mode_3_final_pos[2] = 0
    mode_4_final_pos = mode_3_final_pos
    print("start pos is",mode_4_final_pos)
    mode_4_params = [ mode_4_start_pos, mode_4_final_pos, 20, 100]
    mode_4 = state_descriptor.StateDescriptor(4, mode_4_params, mode_3.final_time)
    # track quad and save the actual vs desired states
    states_saved = main(question, mode_4)
    mode_4.state_storage(states_saved)
    print("\n")

    # calculate the overall time vector across all the states
    state_array = [ mode_0.final_state, 
                    mode_1.final_state, 
                    mode_2.final_state, 
                    mode_3.final_state, 
                    mode_4.final_state
                    ]
    time_vec_overall, actual_overall, desired_overall = calcuate_overall_time_and_pose(state_array)

    plot_state_error(actual_overall, desired_overall, time_vec_overall)

    # plot for 3d visualization
    plot_position_3d(actual_overall, desired_overall)

    # plot desired pose vs actual pose
    plot_des_vs_track(actual_overall, desired_overall, time_vec_overall)


def check_convergence(state_descp, actual_state):
    """
    Check if robot has reached the destination and record state and time
    Args:
        state_descp  : Object of state descriptor
        actual_state : live tracking of controller

    Returns:
        convergence_condition: 1 = converged, 0 = not_yet_converged
    """
    end_point_des = np.array(state_descp.params[1])
    x_actual, y_actual, z_actual, yaw_actual = actual_state[0], \
                                               actual_state[1], \
                                               actual_state[2], \
                                               actual_state[8]
    
    end_point_actual = np.array([x_actual, y_actual, z_actual, yaw_actual])
    error_in_pos = np.linalg.norm((end_point_des - end_point_actual))
    if error_in_pos < 0.01:
        print("converged")
        return True
    else:
        return False

def calcuate_rise_and_overshoot_for_z(state_vector, time_vec):
    print("len of state array original is", state_vector.shape)
    print("len of time vec is", len(time_vec))
    # find rise time
    rise_index = np.where(state_vector[2]>0.09)
    rise_index = rise_index[0][0]
    print("rise index is", rise_index)
    rise_start_time = 4
    rise_end_time = time_vec[rise_index]
    print("rise time is", rise_end_time-rise_start_time)

    # find max percentage overshoot
    max_desired_value = 0.1
    max_actual_value = np.max(state_vector[2,:])
    max_pos_index = np.where(state_vector[2,:] == max_actual_value)[0][0]
    print("max_pos_index is", max_pos_index)
    print("max percentage overshoot is", (max_actual_value-max_desired_value)*100)

    if ((max_actual_value-max_desired_value)*100 < 10):
        print("settling time is", time_vec[rise_index] - 4)
    
    print("steady state value is", state_vector[2,-1], state_vector[8,-1])


def calcuate_overall_time_and_pose(state_array):
    """
    Combines the recorded states and time vectors of each state
    Args:
        state_array (list): array of state tracking histories saved in dict form
    """
    overall_time_vec = []
    overall_actual_state_vec = np.zeros(shape=(15,1))
    overall_desired_state_vec = np.zeros(shape=(15,1))

    mode_num = 0

    print("combining the individual time vectors")
    for state in state_array:
        # combine the time vectors
        print("\n mode number is", mode_num)
        time_vec = state["time_vec"]
        time_finish_iter = state["finish_time_iter"]
        time_vec_cut = time_vec[:time_finish_iter]
        overall_time_vec = overall_time_vec + time_vec_cut

        # combine the actual states, combine desired states
        overall_actual_state_vec = np.concatenate((overall_actual_state_vec, 
                                                    state["actual_states"]), 
                                                    axis=1)
        overall_desired_state_vec = np.concatenate((overall_desired_state_vec, 
                                                    state["desired_states"]), 
                                                    axis=1)
        mode_num += 1

    # remove the null values in overall_state_arrays
    overall_actual_state_vec = overall_actual_state_vec[:,1:]
    overall_desired_state_vec = overall_desired_state_vec[:,1:]
    overall_time_vec = overall_time_vec

    return overall_time_vec, overall_actual_state_vec, overall_desired_state_vec

def store_idle_pos(state_descp):
    time_initial, _, time_step, time_vec, max_iteration = state_descp.time_params()
    finish_time_iteration = max_iteration
    finish_time = (max_iteration)*time_step + time_initial
    actual_state_matrix, _ = state_descp.traj_generator()
    actual_desired_state_matrix = deepcopy(actual_state_matrix)
    saved_state = {
                "actual_states" : actual_state_matrix, 
                "desired_states" : actual_desired_state_matrix, 
                "time_vec": time_vec, 
                "finish_time_iter": finish_time_iteration, 
                "finish_time": finish_time,
                "time_initial": time_initial, 
                "time_step": time_step
                  }
    
    return saved_state


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
    if int(question) < 4:
        main(question, 0)

    elif int(question) == 4:
        traj = [ [0,0,1,0], [2,1,1,0], 10, 100]
        state_machine(question, traj)

    elif int(question) == 5:
        traj = [ [0,0,1,0], [0,0,0.1,0], 10, 100]
        question = 4
        state_machine(question, traj)