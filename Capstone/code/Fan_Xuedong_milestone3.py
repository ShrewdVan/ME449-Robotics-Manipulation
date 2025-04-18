import numpy as np
import modern_robotics as mr
from numpy import cos, sin, tan, pi, sqrt, array, matrix

# ===========================================================================================
# PURPOSE OF THE CODE (MILESTONE 3)
# ===========================================================================================
""" The milestone3, the third part of the Capstone is created to calculate the requested speed
       through the feedforward and feedback control law. The milestone2 provides the desired 
       trajectory, and milstone3 provides the speed requested to move the current configuration
       to the desired configuration with unit time.
"""
# ===========================================================================================

# ===========================================================================================
# HOW TO EXECUTE OR TEST THE CODE
# ===========================================================================================
''' 1. 
#    The first step to test a feedforward plus feedback control is to set both the Ki and Kp 
#        to be zero. The ff+fd control will degrade to ba a ff contorl. The function should return
#        a speed exact the same with the desire speed.
# 2. 
#    Then add the integral error feedback and proportional error feedback to see whether the other
#        two parts also work well.        '''
# ==========================================================================================
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

def FeedbackControl(T_se,T_se_d,T_se_d_next,Kp,Ki,dt):# Ki and Kp are matrix
    """
    Main used for calculate the requested twist leading the robot to move its end-effector to the 
    desired configuration

    :param T_se: The current configuration of end-effector represented in fixed frame
    :param T_se_d: The desired configuration of end-effector represented in fixed frame
    :param T_se_d_next: The desired configuration of end-effector represented in fixed frame two unit time from now
    :param Kp: Proportinal error parameter
    :paran Ki: Integral error parameter
    :paran dt: unit time for whole procedure

    :return: The requested twist leading the robot to move its end-effector to the 
             desired configuration
    """

    # Feedforward term
    se3_d = (1/dt)*mr.MatrixLog6(mr.TransInv(T_se_d)@T_se_d_next)
    V_d = mr.se3ToVec(se3_d)
    feedforward = (mr.Adjoint(mr.TransInv(T_se)@T_se_d))@V_d
    
    X_err = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T_se)@T_se_d))
    # Proportional Gain Term
    Feedback_P = Kp @ X_err
    # Integral Gain Term
    Feedback_I = (Ki @ X_err) * dt

    V = feedforward + Feedback_P + Feedback_I

    return V, X_err


"""
    Now we have the desired twist leading the end-effector to the desired configuration within the unit time,
        we need the jacobian to convert the twist to speed, which is our input form. Because the end-effector 
        is mounted on the chassis, the configuration of the end-effector depends on both chassis jacobian and
        the arm jacobian. The following part is exactly how to construct the chassis jacobian and then end-effector
        jacobian by gathering them up.  
"""

gamma_list = [-pi/4,pi/4,-pi/4,pi/4]
beta_list = [0,0,0,0]
x_list = [0.235,0.235,-0.235,-0.235]
y_list = [0.15,-0.15,-0.15,0.15]
r = 0.0475
H = H0(gamma_list,beta_list,x_list,y_list,r)
Blist = np.array([
    [0, 0, 1,       0,  0.033, 0],      
    [0, -1, 0,     -0.5076, 0, 0],    
    [0, -1, 0,     -0.3526, 0, 0],    
    [0, -1, 0,     -0.2176, 0, 0],   
    [0, 0, 1,       0,      0, 0]]).T 
M_0e = np.array([
    [1, 0, 0, 0.033],
    [0, 1, 0, 0],
    [0, 0, 1, 0.6546],
    [0, 0, 0, 1]])
T_b0 = np.array([
    [1, 0, 0, 0.1662],
    [0, 1, 0, 0],
    [0, 0, 1, 0.0026],
    [0, 0, 0, 1]])
arm_thetalist = array([0,0,0.2,-1.6,0])
# Calculate the base Jacobian
H_pesudo = np.linalg.pinv(H,rcond=1e-3) 
top_zeros = np.zeros((2, H_pesudo.shape[1]))
bottom_zeros = np.zeros((1, H_pesudo.shape[1]))
F_6 = np.vstack((top_zeros, H_pesudo, bottom_zeros))
T_0e = mr.FKinBody(M_0e,Blist,arm_thetalist)
T_eb = mr.TransInv(T_0e) @ mr.TransInv(T_b0)
J_base = (mr.Adjoint(T_eb) @ F_6)

J_arm = mr.JacobianBody(Blist,arm_thetalist)
J_e = np.hstack((J_base,J_arm))
