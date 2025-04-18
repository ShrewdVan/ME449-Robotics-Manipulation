import numpy as np
import modern_robotics as mr
from numpy import cos, sin, tan, pi, sqrt, array, matrix

# ===========================================================================================
# PURPOSE OF THE CODE (MILESTONE 1)
# ===========================================================================================
"""
    The milestone1, the first part of the Capstone is created to calculate the robot actual 
       position configuration for the after passing the unit time. The wheel speed and the 
       arm joint speed are approximated by the linear, simple first order Euler step; while 
       the chassis configuration is calculated by Odometry, which is the majority of the 
       'NextState' Function. With the current position configuration and the requested speed,
        the code included in milestone1 is able to return the configuration for the next iteration,
        which is one of the key components for the feedforward + feedback control. 
"""
# ===========================================================================================

# ===========================================================================================
# HOW TO EXECUTE OR TEST THE CODE
# ===========================================================================================
""" 
#    1. 
#       This code requires lots of result from the other parts of the capstone, a constant 
#         wheel speed and arm speed is preferred for a quick test. IMPORTANT thing is unlike 
#         capstone 2, this part is one of the essential part for iteration. Therefore there 
#         should be no loop inside the function, but for testing, you should create a iteration
#         loop inside or outside the function to see the whole result for your test case. To create
#         a loop inside, you can just uncomment the line 72 and line 108.
"""
# ==========================================================================================

# ===========================================================================================
# FUNCTION DEFINING SECTION
# ===========================================================================================
def H0(gamma_list,beta_list,x_list,y_list,r):
    """
    Main used for generating H matrix which can convert the chassis twist to chassis speed, 
    or vice versa

    :param gamma_list: The free sliding direction for each wheel
    :param beta_list: The forward driving direction for each wheel
    :param x_list: The x position of each wheel represented in chassis frame
    :param y_list: The y position of each wheel represented in chassis frame
    :paran r: The radius of the wheel

    :return: The H matrix for the certain wheel robot
    """
    H_list = []
    for l in range(4):

        gamma = gamma_list[l]
        beta = beta_list[l]
        x = x_list[l]
        y = y_list[l]

        m1 = array([1/r,tan(gamma)/r])
        m2 = array([[cos(beta),sin(beta)],[-sin(beta),cos(beta)]])
        m3 = array([[-y,1,0],[x,0,1]])
        h = m1@m2@m3

        H_list.append(h)
        H = array(H_list)

    return H

def NextState(C_A_W_Config,W_A_Speed,H,dt,limit,Grasp_State):
    """
    Main used for generating robot configuration which includes chassis, arm
    and wheel configuration for the next unit time

    :param C_A_W_Config: The current 12-vector configuration 
    :param W_A_Speed: The requested speed given by the controller
    :param H: The H matrix created by Function 'H0'
    :param t: The total time for whole procedue, created for constant speed test case using
    :param dt: The unit time for whole procedue
    :param limit: The speed limit 
    :paran Grasp_State: The Gripper state for the current time

    :return: The 12 configuration for the next iteration
    """
    configuration = C_A_W_Config.copy()
    limit = int(limit)

    # Limit the velocity within the bound
    W_A_Speed = np.clip(W_A_Speed, -limit, limit)

        # Wheel Config updated
    wheel_config = configuration[8:] + W_A_Speed[0:4] * dt

        # Joint Config updated
    arm_config = configuration[3:8] + W_A_Speed[4:9] * dt

        # Chassis Config 
    V_b = np.linalg.pinv(H,rcond=1e-3)@ W_A_Speed[:4].reshape(-1, 1)
    V_b6 = array([0,0,V_b[0,0],V_b[1,0],V_b[2,0],0])
    se3_b6 = mr.VecTose3(V_b6)
    SE_b = mr.MatrixExp6(se3_b6*dt)
    delta_omega_b = np.arccos(SE_b[1][1])
    delta_x_b = SE_b[0][3]
    delta_y_b = SE_b[1][3]
    if np.isclose(delta_omega_b, 0):
        delta_q_b = array([[0],[delta_x_b],[delta_y_b]])
    else:
        delta_q_b = array([[delta_omega_b],
                          [(delta_x_b*sin(delta_omega_b)+delta_y_b*(cos(delta_omega_b)-1))/delta_omega_b],
                          [(delta_y_b*sin(delta_omega_b)+delta_x_b*(1-cos(delta_omega_b)))/delta_omega_b]])
    R_z = array([[1,0,0],
                 [0,cos(configuration[0]),-sin(configuration[0])],
                 [0,sin(configuration[0]),cos(configuration[0])]])
    delta_q = R_z @ delta_q_b
    chassis_config = configuration[0:3] + delta_q.flatten()

    configuration[:3] = chassis_config
    configuration[3:8] = arm_config
    configuration[8:12] = wheel_config

    return configuration
