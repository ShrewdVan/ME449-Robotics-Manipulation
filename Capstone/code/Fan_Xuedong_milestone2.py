import numpy as np
import modern_robotics as mr 
from numpy import sin, cos, pi

# ===========================================================================================
# PURPOSE OF THE CODE (MILESTONE 2)
# ===========================================================================================
""" The milestone2, the second part of the Capstone is created to generate the reference 
       trajectory for the robot to follow. It's the ideal path, or desired trajectory. The whole 
       task contains 8 segment, by plugging in the starting and ending configuration to the 
       function below, we can get the desired configuration at every single second. The whole 
       configuration path can be obtained by gathering all 8 segments together, which is one of 
       the key components for the feedforward + feedback control. 
"""
# ===========================================================================================

# ===========================================================================================
# HOW TO EXECUTE OR TEST THE CODE
# ===========================================================================================
''' 1. 
#    Find the "Test Section" at the end of page, there are already
#         two gievn example cases, uncomment the code to use.
# 2. 
#    If you want to simulate the case with your own parameter, edit the
#         parameters of included in that two row.
# 3.
#    The specific meaniing and details of each parameter can be found in
#         comment of the function "T_se" and "GraspAndStandoff" in
#         "Function Defining Section".
# 4. 
#    After finishing setting the parameter, just hit the "run" to get the 
#         desired trajectory in array form.
# 5.
#    The function will also generate a csv file, "trajectory.csv"   '''
# ==========================================================================================


# ===========================================================================================
# BACKGROUND PARAMETER SETTING SECTION
# ===========================================================================================
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
T_sc_initial = np.array([
    [1, 0, 0, 1.0],  
    [0, 1, 0, 0.0],  
    [0, 0, 1, 0.025], 
    [0, 0, 0, 1]])
T_sc_final = np.array([
    [0, 1, 0, 0.0],
    [-1, 0, 0, -1.0],  
    [0, 0, 1, 0.025],  
    [0, 0, 0, 1]])

wheel_radius = 0.0475  
wheel_base = 0.235 
track_width = 0.15 
thetalist0 = np.array([0,0,0,0,0]) 
dt = 0.01


# ===========================================================================================
# FUNCTION DEFINING SECTION
# ===========================================================================================
def T_se(phi,x,y,thetalist):
    """
    Main used for generating the SE3 T_se_initial with the specific setting.

    :param phi: The rotational angle in degree reagding the z axis as the screw axis
    :param x: x-position of base in {s} frame
    :param y: y-position of base in {s} frame
    :param thetalist: A list of joint coordinates from 1 to 5

    :return: A homogeneous transformation matrix representing the end-effector 
             frame in {s} frame when the joints are at the specified coordinates 
    """
    T_sb = np.array([
    [cos(phi), -sin(phi), 0, x],
    [sin(phi),  cos(phi), 0, y],
    [0,      0,      1, 0.0963],
    [0,      0,      0,      1]])
    T_0e = mr.FKinBody(M_0e,Blist,thetalist)
    T_be = T_b0 @ T_0e
    T_se = T_sb @ T_be
    return T_se

def GraspAndStandoff(angle,height):
    """
    Compute the Position of the end-effector respresented in the {c} frame, cube
    frame, at arbitrary setting.

    :param angle: The rotational angle in degree from the y axis
    :param height: the z distance between the standoff position and grasp position

    :return: The homogeneous transformation matrix representing the end-effector
             frame in {c} frame when the gripper is at standoff and grasping position
    """
    theta_ce = angle
    T_ce_grasp = np.array([[ cos(theta_ce), 0, sin(theta_ce), 0],
					       [ 0,             1, 0,             0],
					       [-sin(theta_ce), 0, cos(theta_ce), 0],
					       [ 0,             0,             0, 1]])
    
    T_ce_standoff = np.array([[ cos(theta_ce), 0, sin(theta_ce), 0],
					       [ 0,             1, 0,             0],
					       [-sin(theta_ce), 0, cos(theta_ce), height],
					       [ 0,             0,             0, 1]])
    
    return T_ce_grasp, T_ce_standoff


def Array_Output(traj, N, Grasp_State):
    """
    Convert the result form from the SE3 to the array form with the last element
    indicating the state of the gripper.

    :param traj: The trajectory created by ScrewTrajectory at each stage
    :param N: Number of step

    :return: A array containning the information of SE3 and gripper state
    """
    output = np.zeros((N, 13))
    
    for i in range(N):
        # Rotational part to list form
        output[i][0] = traj[i][0][0]
        output[i][1] = traj[i][0][1]
        output[i][2] = traj[i][0][2]
        output[i][3] = traj[i][1][0]
        output[i][4] = traj[i][1][1]
        output[i][5] = traj[i][1][2]
        output[i][6] = traj[i][2][0]
        output[i][7] = traj[i][2][1]
        output[i][8] = traj[i][2][2]

        # Translational part to list form
        output[i][9] = traj[i][0][3]  # x component
        output[i][10] = traj[i][1][3]  # y component
        output[i][11] = traj[i][2][3]  # z component
        
        # Add the parameter showing gripper open or close at last
        output[i][12] = Grasp_State

    return output


def TrajectoryGenerator(T_se_initial, T_sc_initial, T_sc_final, T_ce_grasp, \
                        T_ce_standoff, k):
    '''
    Computes the trajectory for the whole path generates a matching csv file 

    :param: Tse_initial: The initial configuration of the end-effector represented in {s}
    :param: Tsc_initial: The initial cofiguration of the cube represented in {s}
    :param: Tsc_final: The final configuration of the cube represented in {s}
    :param: Tce_grasp: The configuration of the end-effector represented in {c} when doing grasping
    :param: Tce_standoff: The configuration of the end-effector represented in {c} when it's 
           at standoff position
    
    :return: A representation of the N configurations of the end-effector along the entire 
             concatenated eight-segment reference trajectory with its matching csv file. 
             Each of these N reference points represents a transformation matrix T_se of
             the end-effector frame {e} relative to {s} at an instant in time, plus the gripper state (0 or 1).
  
    '''
    
    T_se_grasp_initial = T_sc_initial @ T_ce_grasp
    T_se_grasp_final = T_sc_final @ T_ce_grasp
    T_se_standoff_initial = T_sc_initial @ T_ce_standoff
    T_se_standoff_final = T_sc_final @ T_ce_standoff
    
    Trajectory = np.zeros((0, 13))
    for index in list(range(1,9)):
        if index == 1:
            '''A trajectory to move the gripper from its initial configuration
              to a "standoff" configuration a few cm above the block'''
            Tf = 3
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_initial, T_se_standoff_initial,\
                               Tf,N,5)
            traj = Array_Output(traj,N,0)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 2:
            '''A trajectory to move the gripper down to the grasp position'''
            Tf = 2
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_standoff_initial, T_se_grasp_initial,\
                               Tf,N,5)
            traj = Array_Output(traj,N,0)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 3:
            '''Closingof the gripper'''
            Tf = 0.63
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_grasp_initial, T_se_grasp_initial,\
                               Tf,N,5)
            traj = Array_Output(traj,N,1)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 4:
            '''A trajectory to move the gripper back up to the "standoff" configuration'''
            Tf = 2
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_grasp_initial, T_se_standoff_initial,\
                               Tf,N,5)
            traj = Array_Output(traj,N,1)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 5:
            '''A trajectory to move the gripper to a "standoff" configuration above 
               the final configuration'''
            Tf = 3
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_standoff_initial, T_se_standoff_final,\
                               Tf,N,5)
            traj = Array_Output(traj,N,1)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 6:
            '''A trajectory to move the gripper to the final configuration of the object'''
            Tf = 2
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_standoff_final, T_se_grasp_final,\
                               Tf,N,5)
            traj = Array_Output(traj,N,1)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 7:
            '''Opening of the gripper'''
            Tf = 0.63
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_grasp_final, T_se_grasp_final,\
                               Tf,N,5)
            traj = Array_Output(traj,N,0)
            Trajectory = np.vstack((Trajectory, traj))
        elif index == 8:
            '''A trajectory to move the gripper back to the "standoff" configuration'''
            Tf = 2
            N = int(Tf*k/dt)
            traj = mr.ScrewTrajectory(T_se_grasp_final, T_se_standoff_final,\
                               Tf,N,5)
            traj = Array_Output(traj,N,0)
            Trajectory = np.vstack((Trajectory, traj))

    np.savetxt("Reference_Trajectory.csv", Trajectory, delimiter=",", fmt="%.6f", comments="")
    return Trajectory


# ===========================================================================================
# TESTING SECTION
# ===========================================================================================
''' Here the two cases to test the function
# 1.  Base aligned with {s}, the angle of the gripper and the cube is 135 degree (rotate along y-axis)
#     when grasping and a height of 0.3 m, just like the animation of the reference video.'''
########################## Uncomment the two row to use the test case ######################
T_se_initial = T_se(0,0,0,thetalist0)
T_ce_grasp, T_ce_standoff = GraspAndStandoff(3*pi/4,0.3)
########################## Uncomment the two row to use the test case ######################

''' 2.  Base rotate a degree of 30 alone the z-axis, the angle of the gripper and the cube is 180 degree (rotate along y-axis)
#     when grasping and a height of 0.3 m.'''
########################## Uncomment the two row to use the test case ######################
# T_se_initial = T_se(pi/6,0,0,thetalist0)
# T_ce_grasp, T_ce_standoff = GraspAndStandoff(pi,0.1)
########################## Uncomment the two row to use the test case ######################

Trajectory = TrajectoryGenerator(T_se_initial, T_sc_initial, T_sc_final, T_ce_grasp, T_ce_standoff, 1)
