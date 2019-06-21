#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
#rospy.path.insert(0, ' /opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from collections import deque# Ordered collection with ends

import time
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image

from std_msgs.msg import String
import threading

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError

import math
from time import  sleep

class image_converter():
    def __init__(self):
        self.main()
        self.bridge = CvBridge()
        self.robotMovementPub = rospy.Publisher("/GETjag/cmd_vel",Twist,queue_size=10)
        self.robotPoseSub = rospy.Subscriber("GETjag/odom", Odometry , self.robotCallback)
        self.goalPoseSub = rospy.Subscriber("/GETjag/goal_pose", Odometry  , self.goalCallback)
        self.depthImageSub = rospy.Subscriber("/GETjag/xtion/depth/image_raw", Image, self.depthImageCallback)
        self.elevImageSub = rospy.Subscriber("/GETjag/elevation_map_image", Image, self.elevImageCallback)
        #self.depthImageSub = rospy.Subscriber("/GETjag/xtion/depth/image_raw", Image , self.depthImageCallback)
        #self.elevImageSub = rospy.Subscriber("/GETjag/elevation_map_image", Image , self.elevImageCallback)
        self.VERBOSE = True
        self.depthImage =  np.zeros((480, 640, 1), dtype = "float")
        self.eleviationImage = np.zeros((200, 200, 1), dtype = "uint8")
        self.currentPose = Odometry()
        self.goalPose = Odometry()
        # actions, 0: hard turn left, 1: soft turn left, 2: drive forward, 3: soft turn right, 4: hard turn right
        self.velocities = Twist()
        self.velocities.linear.x = 0.02
        self.velocities.angular.z = 0.3
        self.hard_left = [1, 0, 0, 0, 0]
        self.left = [0, 1, 0, 0, 0]
        self.forward = [0, 0, 1, 0, 0]
        self.right = [0, 0, 0, 1, 0]
        self.hard_right = [0, 0, 0, 0, 1]
        self.countPub = 0

    def setAction(self,action):
        if (isinstance(5, list)):
            #print("set Action")
            if action == self.hard_left:
                self.velocities.linear.x = 0.05
                self.velocities.angular.z = 0.8
            elif action == self.left:
                self.velocities.linear.x = 0.1
                self.velocities.angular.z = 0.4
            elif action == self.forward:
                self.velocities.linear.x = 0.4
                self.velocities.angular.z = 0.0
            elif action == self.right:
                self.velocities.linear.x = 0.1
                self.velocities.angular.z = -0.4
            elif action == self.hard_right:
                self.velocities.linear.x = 0.05
                self.velocities.angular.z = -0.8
            else:
                self.velocities.linear.x = 0.0
                self.velocities.angular.z = 0.0
        else:
            if action == 0:
                self.velocities.linear.x = 0.05
                self.velocities.angular.z = 0.8
            elif action == 1:
                self.velocities.linear.x = 0.1
                self.velocities.angular.z = 0.4
            elif action == 2:
                self.velocities.linear.x = 0.4
                self.velocities.angular.z = 0.0
            elif action == 3:
                self.velocities.linear.x = 0.1
                self.velocities.angular.z = -0.4
            elif action == 4:
                self.velocities.linear.x = 0.05
                self.velocities.angular.z = -0.8
            else:
                self.velocities.linear.x = 0.0
                self.velocities.angular.z = 0.0

    # callback to get the current robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    # also transmits the robot action as velocities
    def robotCallback(self,odom_data):
        self.currentPose = odom_data
        self.robotMovementPub.publish(self.velocities)
        self.countPub +=1
        if self.countPub % 400 == 0:
            self.countPub = 0
           # print("self.velocities" + str(self.velocities))


    # callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    def goalCallback(self,odom_data):
        self.goalPose = odom_data


    # return the eleviation map image with [x,x] Pixel and saves it to a global variable
    def depthImageCallback(self, ros_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data,"32FC1")
        except CvBridgeError as e:
            print(e)
        self.depthImage = cv2.resize(cv_image, (84, 84))


    # return the eleviation map image with [200,200] Pixel and saves it to a global variable
    def elevImageCallback(self,ros_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgra8")
        except CvBridgeError as e:
            print(e)
        self.eleviationImage = cv2.resize(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY), (200, 200))

    def returnData(self):
        return self.depthImage, self.eleviationImage, self.currentPose, self.goalPose

    def main( self):
        '''Initializes and cleanup ros node'''
        #rospy.signal_shutdown("shut de fuck down")
        #äprint("run ros node")
        rospy.init_node('GETjag_drl_gaz_robot_env_wrapper')
# This class proveds the interaction between the robot and the DQN Agent
class robotEnv():
    def __init__(self):
        self.actions = [1,2,3,4]
        self.maxDistanz=5
        self.currentPose = Odometry()
        self.goalPose = Odometry()
        self.ic = image_converter()
        self.availableActionsSize = 5
        self.EpisodeLength = 100
        self.stepCounter=0
        self.episodeFinished = False
        # Variables to calculate the reward
        self.deltaDist = 0.20
        self.discountFactorMue = 1
        self.closestDistance = 0
        self.startGoalDistance = 0
        self.lastDistance = 0
        self.startTime =0

        self.lastTime = 0

        self.velMemory = Memory(10)

    def set_episode_length(self,EpisodeLength):
        self.EpisodeLength = EpisodeLength

    # Takes a step inside the enviroment according to the proposed action and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel),
    # orientation (Euler notation [roll, pitch, yaw]) and the archived reward for the new state
    def takeStep(self, action):
        # here the action should be to the robot and the reward should return
        # überlegen wie ich die zeitverzögerung realisieren will
        #print("take a step")
        self.ic.setAction(action)
        self.stepCounter += 1

        sleep(0.3)

        reward, d =self.clcReward()
        return reward, d

    def getState(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()

        # creat np arrays as input of the drl agent
        roll, pitch, yaw = self.returnRollPitchYaw(self.goalPose.pose.pose.orientation)
        goalOrientation = np.asarray([roll, pitch, yaw])
        goalPosition = self.goalPose.pose.pose.position;
        goalPosition = np.array([goalPosition.x, goalPosition.y, goalPosition.z])

        depthData = np.asarray(depthImage)
        eleviationData = np.asarray(eleviationImage)
        depthData = depthData.astype('float32')
        eleviationData = eleviationData.astype('float32')

        # norrm input data between -1 and 1
        depthData = depthData/5
        eleviationData = eleviationData/255

        goalOrientation = goalOrientation/math.pi
        goalPosition = goalPosition / self.maxDistanz

        goalPose = np.concatenate((goalPosition, goalOrientation), axis=None)

        goalPose = np.asarray(goalPose)
        # replace nan values with 0
        depthData=np.nan_to_num(depthData)

        return depthData, eleviationData, goalPose

    def get_available_actions_size(self):
        return self.availableActionsSize

    # Restet the enviroment and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel) and
    # orientation (Euler notation [roll, pitch, yaw])of the oberservation in the new enviroment
    def reset(self):
        # reset the model and replace the robot at a random location
        valiedEnv = False

        # resest the step counter
        self.stepCounter=0
        # reset the buffer with the velocitys
        self.velMemory.resetBuffer()
        countEnv = 0

        print("reset")
        # repead until the robot is in a valite starting position
        while valiedEnv == False:
            rospy.set_param('/GETjag/End_of_episode', True)

            waitForReset = False
            # wait that the enviroment is build up and the agent is ready
            countReset = 0

            while waitForReset==False:
                sleep(0.2)
                Ready = rospy.get_param("/GETjag/Ready_to_Start_DRL_Agent")
                if Ready:
                    waitForReset = True
                countReset += 1
                if countReset%10 == 0:
                    print("countReset: " + str(countReset))


            depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()
            roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

            position_q = self.currentPose.pose.pose.position

            # cheakk if the roboter is in a valide starting position
            if roll <= math.pi/4 and roll >= -math.pi/4 and pitch <= math.pi/4 and pitch >= -math.pi/4 and position_q.z <0.7:
                valiedEnv = True
                sleep(0.2)

            countEnv += 1
            if countEnv%5 == 0:
                print("countEnv: " + str(countEnv))

        self.startGoalDistance = self.clcDistance(self.goalPose.pose.pose.position)
        self.closestDistance = self.startGoalDistance
        self.startTime = time.time()
        self.lastTime = self.startTime

        sleep(0.3)


    def clcReward(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()

        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

        currenVel = self.currentPose.twist.twist.linear.x

        currentdistance = self.clcDistance(self.goalPose.pose.pose.position)
        currentTime = time.time()
        currentTime = currentTime - self.startTime
        deltaTime = currentTime - self.lastTime


        self.velMemory.add(currenVel)
        meanVel = self.velMemory.mean()

        EndEpisode = False;
        reward=0

        #print("currentdistance: " + str(currentdistance))
        #print("self.startGoalDistance: " + str(self.startGoalDistance))
        #print("self.closestDistance: " + str(self.closestDistance))
        #print("currentTime: " + str(currentTime))
        #print("deltaTime: " + str(deltaTime))

#
        if rospy.get_param("/GETjag/Error_in_simulator"):
            EndEpisode = True
            rospy.set_param("/GETjag/Error_in_simulator",False)

        if currentdistance < self.closestDistance:
            reward = self.discountFactorMue*(self.closestDistance-currentdistance)/deltaTime
            self.closestDistance = currentdistance
        elif currentdistance <= self.startGoalDistance:
            reward = 0.5 + (self.startGoalDistance / currentTime)
        elif roll>=math.pi/4 or roll<=-math.pi/4:
            reward = -0.5
            EndEpisode = True

        if (meanVel <= 0.01):
            reward = -0.5
            EndEpisode = True


        if currentdistance <= self.deltaDist:
            reward = 1000
            print("reached Goal")
            EndEpisode = True

        if  self.stepCounter>=self.EpisodeLength:
            EndEpisode = True

        #print("reward",reward)

        self.lastDistance = currentdistance
        self.lastTime = currentTime

        self.episodeFinished = EndEpisode
        return reward, EndEpisode

    def stopSteps(self):
        self.ic.setAction(20)

    def is_episode_finished(self):
        return self.episodeFinished

    def endEnv(self):
        rospy.set_param("/GETjag/End_of_enviroment",True)

    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return  euler_from_quaternion(orientation_list)

    def clcDistance(self, goal):
        distance = math.sqrt(pow((goal.x),2)+pow((goal.y),2))
        return distance

class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.counter = 1
        self.returnMean = False
    def add(self, value):
        self.buffer.append(value)
        if(self.counter<self.size):
            self.counter += 1
        else:
            self.returnMean=True
    def resetBuffer(self):
        self.counter = 0
        self.returnMean = False
        self.buffer = deque(maxlen=self.size)
    def mean(self):
        if self.returnMean:
            return np.sum(self.buffer)/self.size
        else:
            return 0.02




