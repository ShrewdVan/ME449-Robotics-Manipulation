import numpy as np
import modern_robotics as mr
from numpy import cos, sin, tan, pi, sqrt, array, matrix
from Fan_Xuedong_milestone2 import T_se
from Fan_Xuedong_milestone2 import GraspAndStandoff
from Fan_Xuedong_milestone2 import Array_Output
from Fan_Xuedong_milestone2 import TrajectoryGenerator
from Fan_Xuedong_milestone1 import H0
from Fan_Xuedong_milestone3 import FeedbackControl
from Fan_Xuedong_milestone1 import NextState

# ===========================================================================================
# INFO ABOUT THE CODE 
# ===========================================================================================
""" The capstone project code gathers the result from milestone1, milestone2 and milestone3 
        to generate a completed path showing how robot pick up the cube and drop it at the 
        destination position. The whole procedure includes the operation for desired trajectory
        generation, feedforward plus feedback control, joint limit test, and configuration update
        within an iteration repeating 1205 times. 
"""
# ===========================================================================================

# ===========================================================================================
# PRINCIPLE OF THE CODE 
# ===========================================================================================
""" The milestone2 generates the desired trajectory list, the milestone3 operates the current 
        configuration with the desired one and return the requested speed. The milestone1 accepts
        the request speed and send the configuration for the next step to the milestone3. The 
        index moveforwads and give the next desired configuration so that the milestone1 can 
        return the next requested speed. Repeat this procedure, we can get the whole trajectory
        of the actual robot configuration.
"""
# ===========================================================================================
#  ************** *************  HOW TO EXECUTE THE CODE  ************** *************
# ===========================================================================================
"""  The zone betwwen line 107 and line 164 is the Global Parameter Setting zone. You can change
        the paramter including initial configuration, the grasp angle with the cube, the initial
        and destination position of the cube .etc as you like. The line 159 is the setting code
        for the initial 12-elements configuration, change it to the initial configuration
        you want it to be at before running the whole file and then you are ready to go.

        The example initial configuration provided is:
        C_A_W_Config = array([0,0,0,-0.52359878,-0.78539816,-0.57079633,-0.78539816,0,0,0,0,0])
"""
#============================================================================================
#                                 Function Defining
#============================================================================================

def array_to_SE3(T_se_d):
    """
    Convert a 13-element array back to the 4x4 transformation matrix T_se.

    Parameters:
    - array: A list or numpy array with 13 elements:
             [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper_state]

    Returns:
    - T_se: A 4x4 numpy matrix representing the transformation matrix.
    - gripper_state: The 13th element of the array (gripper state: 0 or 1).
    """
    # Extract the 3x3 rotation matrix
    R = array([
        [T_se_d[0], T_se_d[1], T_se_d[2]],
        [T_se_d[3], T_se_d[4], T_se_d[5]],
        [T_se_d[6], T_se_d[7], T_se_d[8]]
    ])

    # Extract the translation vector
    P = array([T_se_d[9], T_se_d[10], T_se_d[11]])

    # Combine into a 4x4 homogeneous transformation matrix
    T_se = np.eye(4)
    T_se[:3, :3] = R
    T_se[:3, 3] = P

    return T_se



def TestJointLimit(joint_1,joint_2,joint_3,joint_4,joint_5):
    """
    Test each joint for the next state and check whether they're within the limit or
    outside the limit. The outsider will be marked as "1" in the return list

    :param joint_1: The theta value of joint_1 in next state
    :param joint_2: The theta value of joint_1 in next state
    :param joint_3: The theta value of joint_1 in next state
    :param joint_4: The theta value of joint_1 in next state
    :param joint_5: The theta value of joint_1 in next state

    :return: A list indicting which joint is out of limit, "0" regarded as "within the limit"
             and "1" regarded as "out of limit"
    """
    error_joints = [0,0,0,0,0]
    if joint_1<-2.93 or joint_1>2.93:
        error_joints[1] = 1
    if joint_2<-2.63 or joint_2>2.63:
        error_joints[1] = 1
    if joint_3<-3 or joint_3>-0.02:
        error_joints[2] = 1
    if joint_4<-3 or joint_4>-0.02:
        error_joints[3] =1
    if joint_5<-1 or joint_5>1:
        error_joints[4] = 1
    return error_joints

#=========================================================================================
#                           Global parameter setting
#=========================================================================================
Blist = array([
    [0, 0, 1,       0,  0.033, 0],      
    [0, -1, 0,     -0.5076, 0, 0],    
    [0, -1, 0,     -0.3526, 0, 0],    
    [0, -1, 0,     -0.2176, 0, 0],   
    [0, 0, 1,       0,      0, 0]]).T 
M_0e = array([
    [1, 0, 0, 0.033],
    [0, 1, 0, 0],
    [0, 0, 1, 0.6546],
    [0, 0, 0, 1]])
T_b0 = array([
    [1, 0, 0, 0.1662],
    [0, 1, 0, 0],
    [0, 0, 1, 0.0026],
    [0, 0, 0, 1]])
T_sc_initial = array([
    [1, 0, 0, 1.0],  
    [0, 1, 0, 0.0],  
    [0, 0, 1, 0.025], 
    [0, 0, 0, 1]])
T_sc_final = array([
    [0, 1, 0, 0.0],
    [-1, 0, 0, -1.0],  
    [0, 0, 1, 0.025],  
    [0, 0, 0, 1]])
T_se_initial = array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0.5],
    [0, 0, 0, 1]])

# Calculate the H(0) matrix
gamma_list = [-pi/4,pi/4,-pi/4,pi/4]
beta_list = [0,0,0,0]
x_list = [0.235,0.235,-0.235,-0.235]
y_list = [0.15,-0.15,-0.15,0.15]
r = 0.0475
H = H0(gamma_list,beta_list,x_list,y_list,r)
H_pesudo = np.linalg.pinv(H,rcond=1e-3) 
top_zeros = np.zeros((2, H_pesudo.shape[1]))
bottom_zeros = np.zeros((1, H_pesudo.shape[1]))
F_6 = np.vstack((top_zeros, H_pesudo, bottom_zeros))
# Generate the reference trajectory
T_ce_grasp, T_ce_standoff = GraspAndStandoff(3*pi/4,0.3)
Tf = 10.26
dt = 0.01
N = int(Tf/dt)
Reference_Trajectory = TrajectoryGenerator(T_se_initial, T_sc_initial,\
                     T_sc_final, T_ce_grasp, T_ce_standoff, 1)
Ki = np.identity(6)* 0.25
Kp = np.identity(6)* 1
limit = 25
Feedback_I = 0
# Initial configuration
C_A_W_Config = array([0,0,0,-0.52359878,-0.78539816,-0.57079633,-0.78539816,0,0,0,0,0])
#=========================================================================================
#                                  Iteration
#=========================================================================================
Configuration = np.hstack((C_A_W_Config,Reference_Trajectory[0][-1]))
Configuration_list = np.zeros(13)
X_err_list = np.zeros(6)
for step in range(1525):
    #Calculate the current Configuration of T_se
    T_se_current = T_se(C_A_W_Config[0],C_A_W_Config[1],C_A_W_Config[2],C_A_W_Config[3:8])
    # Send the current configuration to compare with reference configuration, 
    # return with the twist in end-effector
    T_se_d = array_to_SE3(Reference_Trajectory[step])
    T_se_d_next = array_to_SE3(Reference_Trajectory[step+1])
    Grasp_State = Reference_Trajectory[step][-1]
    V_fffb, X_err = FeedbackControl(T_se_current,T_se_d,T_se_d_next,Kp,Ki,dt)
    # Consturct the full form of the Jacobian_e
    arm_thetalist = C_A_W_Config[3:8]
    T_0e = mr.FKinBody(M_0e,Blist,arm_thetalist)
    T_eb = mr.TransInv(T_0e) @ mr.TransInv(T_b0)
    J_base = (mr.Adjoint(T_eb) @ F_6)
    J_arm = mr.JacobianBody(Blist,arm_thetalist)
    J_e = np.hstack((J_base,J_arm))
#===============================================================================================
    error_list = TestJointLimit(C_A_W_Config[3],C_A_W_Config[4],C_A_W_Config[5],C_A_W_Config[6],C_A_W_Config[7])
    if error_list[0] == 1:
        J_e[:,4] = 0
    if error_list[1] == 1:
        J_e[:,5] = 0
    if error_list[2] == 1:
        J_e[:,6] = 0 
    if error_list[3] == 1:
        J_e[:,7] = 0
    if error_list[4] == 1:
        J_e[:,8] = 0
        
    # Compute the W_A_Speed, which is one of the essential part of Function "NextState"
    W_A_Speed = W_A_Speed = np.linalg.pinv(J_e,rcond=1e-3) @ V_fffb
    # Store the current Configuration and X_err with the Grasp_State
    Configuration = np.hstack((C_A_W_Config,Grasp_State))
    if np.all(Configuration_list == np.zeros(13)):
        Configuration_list = Configuration
    else:
        Configuration_list = np.vstack((Configuration_list, Configuration)) 
    if np.all(X_err_list == np.zeros(6)):
        X_err_list = X_err.reshape(1, -1)
    else:
        X_err_list = np.vstack((X_err_list, X_err.reshape(1,-1))) 
    # Compute the Configuration for next step
    C_A_W_Config = NextState(C_A_W_Config,W_A_Speed,H,dt,limit,Grasp_State)

    np.savetxt("Trajectory_with_limit.csv", Configuration_list, delimiter=",", fmt="%.6f", comments="")
    np.savetxt("X_err_with_limit.csv", X_err_list, delimiter=",", fmt="%.6f", comments="")

