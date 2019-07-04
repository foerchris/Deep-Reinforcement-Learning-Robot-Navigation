import matplotlib.pyplot as plt
import numpy as np
import math as m
from __builtin__ import pow
g1 = np.array([5.14,0, -3.54,0])
c = np.array([0, 0,0, 0])
g2 = np.array([3.84, 0 , 3.9,0])

a=0
b=1.5
#c=1

theta = m.acos(1/m.sqrt(m.pow(a, 2) + m.pow(b, 2) + 1))
q = np.array([m.cos(theta/2), b*m.sin(theta/2) , -a*m.sin(theta/2),0])
print('theta' + str(theta))
print('q' + str(q))

g1_dot = np.array([6.24, 0 , -0.7])
c_dot = np.array([0, 0, 0])
g2_dot = np.array([6.95, 0, 1.77])


def clc_quaternion(q1, q2):

    # Extract the vector part of the quaternion
    #cv::Vec3d u(q.getX(), q.getY(), q.getZ());
    v = np.array([q1[0],q1[1] , q1[2]])
    u = np.array([q2[1],q2[2] , q2[3]])
    print('v' + str(v))
    print('u' + str(u))

    # Extract the scalar part of the quaternion
    s = q2[0];

    # Do the math
    vprime = 2.0 * np.multiply(np.dot(u,v),u)  + (s*s - np.dot(u,u)) * v + 2.0 * s * np.cross(u,v);
    print('vprime' + str(vprime))

    return vprime;

#g1_dot = clc_quaternion(g1,q)
#c_dot = clc_quaternion(c,q)
#g2_dot = clc_quaternion(g2,q)
#print('g1_dot' + str(g1_dot))
#print('c_dot' + str(c_dot))
#print('g2_dot' + str(g2_dot))

norm = np.array([0 , 0 , 1])

ug = g2_dot - g1_dot

print('ug' + str(ug))

uc = c_dot - g1_dot

print('uc' + str(uc))
#uc = np.array([5.0 , 0 , 1.88])
#ug = np.array([10.0 , 0 , 3.32])





term1 = np.divide(np.linalg.norm( np.dot(uc,ug)), np.linalg.norm(ug))

term2 = np.divide(ug, np.linalg.norm(ug))

p_foot = g1_dot +  np.multiply(term1, term2)

print('p_foot: ' +str(p_foot))


u_highest = norm - np.multiply(np.divide(np.linalg.norm(np.dot(norm,ug)), np.linalg.norm(ug)), np.divide( ug, np.linalg.norm(ug)))

print('u_highest' +str(u_highest))

p_highest = p_foot +  np.multiply(np.linalg.norm(p_foot - c_dot) , np.divide(u_highest,np.linalg.norm(u_highest)))

print('p_highest' +str(p_highest))

SNE = np.dot(norm,p_highest) - np.dot(norm,c_dot)

print('SNE' +str(SNE))


