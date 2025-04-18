import numpy as np
import modern_robotics as mr
from numpy import cos, sin, tan, pi, sqrt, array, matrix

def H0(gamma_list,beta_list,x_list,y_list,r):
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

def NextState(C_A_W_Config,W_A_Speed,H,dt,t,limit,Grasp_State):

    configuration = array(np.hstack((C_A_W_Config, Grasp_State)))
    Configuration = configuration.copy()
    N = int(t/dt)
    limit = int(limit)

    # Limit the velocity within the bound
    W_A_Speed = np.clip(W_A_Speed, -limit, limit)

    for i in range(N):
        # Wheel Config updated
        wheel_config = configuration[8:12] + W_A_Speed[0:4] * dt

        # Joint Config updated
        arm_config = configuration[3:8] + W_A_Speed[4:9] * dt

        # Chassis Config 
        V_b = np.linalg.pinv(H)@ W_A_Speed[:4].reshape(-1, 1)
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
        configuration[12] = Grasp_State 

        Configuration = np.vstack([Configuration, configuration])
    
    np.savetxt("Next_State_Config.csv", Configuration, delimiter=",", fmt="%.6f", comments="")

    return Configuration


gamma_list = [-pi/4,pi/4,-pi/4,pi/4]
beta_list = [0,0,0,0]
x_list = [0.235,0.235,-0.235,-0.235]
y_list = [0.15,-0.15,-0.15,0.15]
r = 0.0475