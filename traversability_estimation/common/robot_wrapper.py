#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
import numpy as np
from collections import deque  # Ordered collection with ends

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from tf import TransformListener
from sensor_msgs.msg import Image

from std_msgs.msg import String

from nav_msgs.msg import Odometry
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
import matplotlib.pyplot as plt
from scipy.constants.constants import alpha
class image_converter():
    def __init__(self,number):
        self.number = number

        self.bridge = CvBridge()
        self.bridge2 = CvBridge()

        self.VERBOSE = True
        self.depthImage =  np.zeros((84, 84), dtype = np.float32)
        self.eleviationImage = np.zeros((200, 200), dtype = "uint16")
        self.currentPose = Odometry()
        self.goalPose = Odometry()

        self.velocities = Twist()
        self.velocities.linear.x = 0.2
        self.velocities.angular.z = 0.0
        self.hard_left = [1, 0, 0, 0, 0]
        self.left = [0, 1, 0, 0, 0]
        self.forward = [0, 0, 1, 0, 0]
        self.right = [0, 0, 0, 1, 0]
        self.hard_right = [0, 0, 0, 0, 1]
        self.countPub = 0
        self.robotAction = 7
        self.main()
        self.deltaDist = 1
        self.reach_the_goal = False

        self.mean_lin_vel = Memory(2)
        self.mean_angl_vel = Memory(2)
        self.mean_lin_vel.resetBuffer()
        self.mean_angl_vel.resetBuffer()

    '''
    stop all robots
    '''
    def stop(self):
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0

    '''
    set the action which should executed
    @ param action
    @ return linear and angular velocity
    '''
    def setAction(self, action):

       # action[0] += 1.0
        action[0] = action[0]
        action[1] = action[1]

        if(action[0] >= 1.0):
            action[0] = 1.0
        elif(action[0] <= 0.1):
            action[0] = 0.1

        action[1] = action[1]
        if (action[1] >= 1.0):
            action[1] = 1.0
        elif (action[1] <= -1.0):
            action[1] = -1.0


        self.velocities.linear.x = action[0]
        self.velocities.angular.z = action[1]
        return  self.velocities.linear.x, self.velocities.angular.z

    '''
    calculate mean values for linear and angular veloicity 
    @ return mean- linear and angular velocity and acceleration
    '''
    def clcMean(self):
        meanLinVel, meanLinAcc = self.mean_lin_vel.totalMean()
        meanAngVel, meanAngAcc = self.mean_angl_vel.totalMean()

        #print("meanLinVel = " + str(meanLinVel) + "\n")
        #print("meanAngVel = " + str(meanAngVel) + "\n")
        #print("meanLinAcc = " + str(meanLinAcc) + "\n")
        #print("meanAngAcc = " + str(meanAngAcc) + "\n")

       # file = open("Gazebo Script/measures.txt", "w")
       # file.write("meanLinVel = " + str(meanLinVel) + "\n")
       # file.write("meanAngVel = " + str(meanAngVel) + "\n")
       # file.write("meanLinAcc = " + str(meanLinAcc) + "\n")
       # file.write("meanAngAcc = " + str(meanAngAcc) + "\n")

        #file.close()
        return meanLinVel, meanAngVel, meanLinAcc, meanAngAcc

    '''
    get current robot pose
    @ param odom_data
    '''
    def robotPoseCallback(self,odom_data):
        self.currentRobotPose = odom_data
        self.robotMovementPub.publish( self.velocities)

        self.mean_lin_vel.addVelocity(odom_data.twist.twist.linear.x)
        self.mean_angl_vel.addVelocity(odom_data.twist.twist.angular.z)

    '''
    callback to get the goal-robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    @ param odom_data
    '''
    def goalCallback(self,odom_data):
        self.goalPose = odom_data

        currentdistance = self.clcDistance(self.goalPose.pose.pose.position)

        if currentdistance <= self.deltaDist:
            self.reach_the_goal = True

    '''
    calculate distance to goal
    @ param image; goal pose
    @ return distance 
    '''
    def clcDistance(self, goal):
        distance = math.sqrt(pow((goal.x), 2) + pow((goal.y), 2))
        return distance

    '''
    return the depth image with [x,x] Pixel and saves it to a global variable
    @ param depth_data
    '''
    def depthImageCallback(self, depth_data):
        #print("depthImageCallback");
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_data,"32FC1")
        except CvBridgeError as e:
            print(e)
        self.depthImage = cv2.resize(cv_image, (84, 84))

    '''
    return the eleviation map image with [200,200] Pixel and saves it to a global variable
    @ param map_data
    '''
    def elevImageCallback(self,map_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        try:
            cv_image = self.bridge2.imgmsg_to_cv2(map_data)
        except CvBridgeError as e:
            print(e)

        image = cv_image[:,:,0]
        alpha = cv_image[:,:,1]

        map = np.stack((cv2.resize(image, (200, 200)),cv2.resize(alpha, (200, 200))))

        self.eleviationImage = map

    '''
    return states
    @ return depth image, elevation map image, current robot pose, goal pose
    '''
    def returnData(self):
        return self.depthImage, self.eleviationImage, self.currentRobotPose, self.goalPose

    def main( self):
        '''Initializes and cleanup ros node'''
        print("rospy.init_node('GETjag_"+ str(self.number) +"_drl_gaz_robot_env_wrapper_worker')");
        rospy.init_node('GETjag_'+ str(self.number) +'_drl_gaz_robot_env_wrapper_worker')
        self.robotMovementPub = rospy.Publisher("/GETjag" + str(self.number) + "/cmd_vel", Twist, queue_size=10)
        self.currentRobotPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/current_pose", Odometry, self.robotPoseCallback)

        self.goalPoseSub = rospy.Subscriber("/GETjag" + str(self.number) + "/goal_pose", Odometry, self.goalCallback)
        self.depthImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/xtion/depth/image_raw", Image,
                                              self.depthImageCallback)
        self.elevImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_robot_ground_map", Image,
                                             self.elevImageCallback, queue_size=1)


    def shoutdown_node(self):
        rospy.signal_shutdown("because reason")

'''
class to calculate velocity acceleration and statistics
'''
class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.counter = 1
        self.returnMean = False
        self.reset_buffer = False

        self.lastPosition = 0
        self.lastVel = 0
        self.last_time = time.time()
        self.velovitys = []
        self.accelerations = []
        self.accel = 0


    def addVelocity(self, value):
        if(self.reset_buffer):
            self.resetBuffer()

        self.buffer.append(value)
        self.accel = 0
        if (self.counter < self.size):
            self.counter += 1
        else:
            self.returnMean = True
            self.clcAccFromVel()
        return self.accel

    def addPosition(self, value):
        if (self.reset_buffer):
            self.resetBuffer()

        self.buffer.append(value)
        self.accel = 0
        if (self.counter < self.size):
            self.counter += 1
        else:
            self.returnMean = True
            self.clcAccFromPose()
        return self.accel


    def resetBuffer(self):
        self.counter = 0
        self.returnMean = False
        self.buffer = deque(maxlen=self.size)

        self.lastPosition = 0
        self.lastVel = 0
        self.last_time = time.time()
        self.velovitys = []
        self.accelerations = []
        self.reset_buffer = False

    def clcAccFromPose(self):
        positon = self.mean()
        deltaTime = time.time() - self.last_time
        #if (abs(deltaTime - 0.023) < 0.02):
        vel = abs(positon - self.lastPosition) / deltaTime
        self.velovitys.append(vel)
        self.accel = abs(vel - self.lastVel) / deltaTime
        self.accelerations.append(abs(self.accel))
        self.lastPosition = positon
        self.lastVel = vel
        self.last_time = time.time()


    def clcAccFromVel(self):
        vel = self.mean()
        self.velovitys.append(abs(vel))
        deltaTime = time.time() - self.last_time
        if (abs(deltaTime - 0.023) < 0.02):
            self.accel = abs(vel - self.lastVel) / deltaTime
            self.accelerations.append(abs(self.accel))

        self.lastVel = vel
        self.last_time = time.time()

    def totalMean(self):
        self.reset_buffer = True
        return np.mean(self.velovitys), np.mean(self.accelerations)

    def mean(self):
        if self.returnMean:
            return np.mean(self.buffer)
        else:
            return 0.02

    def var(self):
        if self.returnMean:
            return np.var(self.buffer)
        else:
            return 0.02
    def getvelovitys(self):
        return self.velovitys

    def returnNumpyArray(self):
        return np.asarray(self.buffer)