import numpy as np
import modern_robotics as mr
import csv

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    iteration_matrix = np.array(thetalist0).copy()
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T)))
    SE3 = mr.FKinBody(M, Blist, thetalist)
    eomg_i = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    ev_i = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err = eomg_i > eomg or ev_i > ev

    print(f'Iteration {i}:')
    print(f'joint vector:\n',thetalist)
    print(f'SE(3) end-effector config:\n',SE3)
    print(f'error twist V_b \n',Vb)
    print(f'angular error ||omega_b|| \n',eomg_i)
    print(f'linear error ||v_b|| \n',ev_i)

    while err and i < maxiterations:
        thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
        i = i + 1
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T)))
        SE3 = mr.FKinBody(M, Blist, thetalist)
        eomg_i = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        ev_i = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        iteration_matrix = np.vstack((iteration_matrix,thetalist))

        print(f'Iteration {i}:')
        print(f'joint vector:\n',thetalist)
        print(f'SE(3) end-effector config:\n',SE3)
        print(f'error twist V_b \n',Vb)
        print(f'angular error ||omega_b|| \n',eomg_i)
        print(f'linear error ||v_b|| \n',ev_i)
        err = eomg_i > eomg or ev_i > ev

    with open('iteration_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(iteration_matrix)

# W1=0.109
# W2=0.082
# L1=0.425
# L2=0.392
# H1=0.089
# H2=0.095
# M=np.matrix([[-1,0,0,L1+L2],[0,0,1,W1+W2],[0,1,0,H1-H2],[0,0,0,1]])
# BT = np.matrix([[0,1,0,W1+W2,0,L1+L2],[0,0,1,H2,-L1-L2,0],[0,0,1,H2,-L2,0],[0,0,1,H2,0,0],\
#                [0,-1,0,-W2,0,0],[0,0,1,0,0,0]])
# Blist = BT.T
# T = np.array([[1,0,0,0.3],[0,1,0,0.3],[0,0,1,0.4],[0,0,0,1]])
# eomg = 0.001
# ev = 0.0001

# thetalist_converge = np.array([0.52556898, 4.65381829, 2.15082136, 4.19093463, 1.57079632, 4.18682])

# #A Thetalist converging with 2 iteration
# thetalist0 = np.array([0.525, 4.6, 2.12, 4.2, -4.7, -2.1])

# #A Thetalist converging with 19 iteration
# thetalist0 = np.array([3.66716163, 1.51222564, 5.29241401, 1.04934198, 4.71238897, 1.04522735])

# #Angle for CoppeliaSim input
# equivalent_angles = np.remainder(thetalist, 2 * np.pi)

