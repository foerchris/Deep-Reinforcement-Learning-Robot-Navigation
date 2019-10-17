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

        #if(number==1):
        self.bridge = CvBridge()
        self.bridge2 = CvBridge()

       # self.tf = TransformListener()

        self.VERBOSE = True
        self.depthImage =  np.zeros((84, 84), dtype = np.float32)
        self.eleviationImage = np.zeros((200, 200), dtype = "uint16")
        self.currentPose = Odometry()
        self.goalPose = Odometry()
        # actions, 0: hard turn left, 1: soft turn left, 2: drive forward, 3: soft turn right, 4: hard turn right
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

        self.mean_lin_vel = Memory(5)
        self.mean_angl_vel = Memory(5)
        self.mean_lin_vel.resetBuffer()
        self.mean_angl_vel.resetBuffer()
    def stop(self):
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0

    def setAction(self, action):

        action[0] += 1.0
        action[0] = action[0] / 2
        action[1] = action[1] * 3

        if(action[0] >= 1.0):
            action[0] = 1.0
        elif(action[0] <= 0.1):
            action[0] = 0.1
       # elif(action[0] <= 0.05):
        #    action[0] = 0.05

        if (action[1] >= 1.0):
            action[1] = 1.0
        elif (action[1] <= -1.0):
            action[1] = -1.0
        #if(self.number == 1):
        #    print ("action_worker_" + str(self.number) + ": " + str(action) )

        self.velocities.linear.x = action[0]
        self.velocities.angular.z = action[1]
        return  self.velocities.linear.x, self.velocities.angular.z

    # callback to get the current robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    # also transmits the robot action as velocities
    def robotCallback(self,odom_data):
        self.currentPose = odom_data

        self.robotMovementPub.publish( self.velocities)
       # self.countPub +=1
       # if self.countPub % 1000 == 0:
         #   self.countPub = 0
         #   print("worker_" + str(self.number) + "self.velocities" + str(self.velocities))

    def clcMean(self):

        meanLinVel, meanLinAcc = self.mean_lin_vel.totalMean()
        meanAngVel, meanAngAcc = self.mean_angl_vel.totalMean()
       # print("meanLinVel = " + str(meanLinVel))
       # print("meanAngVel = " + str(meanAngVel))
       # print("meanLinAcc = " + str(meanLinAcc))
       # print("meanAngAcc = " + str(meanAngAcc))

        ##print("linearAccelerations max = " + str(np.max(self.linearAccelerations)))


        file = open("Gazebo Script/measures.txt", "w")
        file.write("meanLinVel = " + str(meanLinVel) + "\n")
        file.write("meanAngVel = " + str(meanAngVel) + "\n")
        file.write("meanLinAcc = " + str(meanLinAcc) + "\n")
        file.write("meanAngAcc = " + str(meanAngAcc) + "\n")

        file.close()
        return meanLinVel, meanAngVel, meanLinAcc, meanAngAcc

    def robotPoseCallback(self,odom_data):
        self.currentRobotPose = odom_data

        self.mean_lin_vel.add(odom_data.twist.twist.linear.x)
        self.mean_angl_vel.add(odom_data.twist.twist.angular.z)

    # callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    def goalCallback(self,odom_data):
        self.goalPose = odom_data

        currentdistance = self.clcDistance(self.goalPose.pose.pose.position)

        if currentdistance <= self.deltaDist:
            self.reach_the_goal = True

    def clcDistance(self, goal):
        distance = math.sqrt(pow((goal.x), 2) + pow((goal.y), 2))
        return distance
    # return the eleviation map image with [x,x] Pixel and saves it to a global variable
    def depthImageCallback(self, depth_data):
        #print("depthImageCallback");
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_data,"32FC1")
        except CvBridgeError as e:
            print(e)
        self.depthImage = cv2.resize(cv_image, (84, 84))


    # return the eleviation map image with [200,200] Pixel and saves it to a global variable
    def elevImageCallback(self,map_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        try:
            cv_image = self.bridge2.imgmsg_to_cv2(map_data)
        except CvBridgeError as e:
            print(e)

        #print(cv_image.shape)
        image = cv_image[:,:,0]
        alpha = cv_image[:,:,1]

        map = np.stack((cv2.resize(image, (200, 200)),cv2.resize(alpha, (200, 200))))
        self.eleviationImage = map
        #print(map.shape)

        #if(self.number== 2):
         #   plt.imshow(cv_image,cmap="gray")
          #  plt.show()

        #self.eleviationImage = cv2.resize(cv_image, (200, 200))



    def returnData(self):
        return self.depthImage, self.eleviationImage, self.currentPose, self.goalPose

    def main( self):
        '''Initializes and cleanup ros node'''
        #rospy.signal_shutdown("shut de fuck down")
        print("rospy.init_node('GETjag_"+ str(self.number) +"_drl_gaz_robot_env_wrapper_worker')");
        rospy.init_node('GETjag_'+ str(self.number) +'_drl_gaz_robot_env_wrapper_worker')
        self.robotMovementPub = rospy.Publisher("/GETjag" + str(self.number) + "/cmd_vel", Twist, queue_size=10)
        self.robotPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/odom", Odometry, self.robotCallback)
        self.currentRobotPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/current_pose", Odometry, self.robotPoseCallback)

        self.goalPoseSub = rospy.Subscriber("/GETjag" + str(self.number) + "/goal_pose", Odometry, self.goalCallback)
        self.depthImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/xtion/depth/image_raw", Image,
                                              self.depthImageCallback)
        # self.elevImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_map_image", Image, self.elevImageCallback)
        self.elevImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_robot_ground_map", Image,
                                             self.elevImageCallback, queue_size=1)

        #rospy.spin()

    def shoutdown_node(self):
        rospy.signal_shutdown("because reason")

class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.counter = 1
        self.returnMean = False
        self.reset_buffer = False

        self.lastVel = 0
        self.last_time = time.time()
        self.velovitys = []
        self.accelerations = []

    def add(self, value):
        if(self.reset_buffer):
            self.resetBuffer()

        self.buffer.append(value)
        if (self.counter < self.size):
            self.counter += 1
        else:
            self.returnMean = True
            self.clcAcc()

    def resetBuffer(self):
        self.counter = 0
        self.returnMean = False
        self.buffer = deque(maxlen=self.size)

        self.lastVel = 0
        self.last_time = time.time()
        self.velovitys = []
        self.accelerations = []
        self.reset_buffer = False


    def clcAcc(self):
        vel = self.mean()
        self.velovitys.append(abs(vel))
        deltaTime = time.time() - self.last_time
        if (abs(deltaTime - 0.023) < 0.02):
            accel = abs(vel - self.lastVel) / deltaTime
            self.accelerations.append(abs(accel))

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

    def returnNumpyArray(self):
        return np.asarray(self.buffer)