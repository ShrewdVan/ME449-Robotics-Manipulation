from ME449_HW2_code import IKinBodyIterates
import modern_robotics as mr
import numpy as np

W1=0.109
W2=0.082
L1=0.425
L2=0.392
H1=0.089
H2=0.095
M=np.matrix([[-1,0,0,L1+L2],[0,0,1,W1+W2],[0,1,0,H1-H2],[0,0,0,1]])
BT = np.matrix([[0,1,0,W1+W2,0,L1+L2],[0,0,1,H2,-L1-L2,0],[0,0,1,H2,-L2,0],[0,0,1,H2,0,0],\
               [0,-1,0,-W2,0,0],[0,0,1,0,0,0]])
Blist = BT.T
T = np.array([[1,0,0,0.3],[0,1,0,0.3],[0,0,1,0.4],[0,0,0,1]])
eomg = 0.001
ev = 0.0001

#A Thetalist converging with 4 iteration
thetalist0 = np.array([0.4, 5, 2.5, 3.8, -4, -2.1])
print("IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)")
IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)

# #A Thetalist converging with 19 iteration
# thetalist0 = np.array([3.66716163, 1.51222564, 5.29241401, 1.04934198, 4.71238897, 1.04522735])
# print("IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)")
# IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)