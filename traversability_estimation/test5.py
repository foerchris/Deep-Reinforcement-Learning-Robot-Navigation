import matplotlib.pyplot as plt
import numpy as np
import math as m
from __builtin__ import pow


g1_dot = np.array([4.5, 0 ,-1.72])
c_dot = np.array([0.0, 0, 0])
g2_dot = np.array([-4.65, 0, -0.25])



norm = np.array([0 , 0 , 1])

ug = g2_dot - g1_dot

print('ug' + str(ug))

uc = c_dot - g1_dot

print('uc' + str(uc))

term1 = np.divide(np.linalg.norm( np.dot(uc,ug)), np.linalg.norm(ug))

term2 = np.divide(ug, np.linalg.norm(ug))

p_foot = g1_dot +  np.multiply(term1, term2)

print('p_foot: ' +str(p_foot))


u_highest = norm - np.multiply(np.divide(np.linalg.norm(np.dot(norm,ug)), np.linalg.norm(ug)), np.divide( ug, np.linalg.norm(ug)))

print('u_highest' +str(u_highest))

p_highest = p_foot +  np.multiply(np.linalg.norm(p_foot - c_dot) , u_highest)#np.divide(u_highest,np.linalg.norm(u_highest)))

print('p_highest' +str(p_highest))

SNE = np.dot(norm,p_highest) - np.dot(norm,c_dot)

print('SNE' +str(SNE))


