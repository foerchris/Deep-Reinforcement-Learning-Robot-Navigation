#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
import numpy as np

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image

from std_msgs.msg import String

from nav_msgs.msg import Odometry

import cv2
from cv_bridge import CvBridge, CvBridgeError

class image_converter():
    def __init__(self,number):
        self.number = number
        #if(number==1):
        self.bridge = CvBridge()
        self.bridge2 = CvBridge()


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

    def stop(self):
        self.velocities.linear.x = 0
        self.velocities.angular.z = 0

    def setAction(self, action):

        if(action[0] >= 1.0):
            action[0] = 1.0
        elif(action[0] <= -1.0):
            action[0] = -1.0

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


    # callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    def goalCallback(self,odom_data):
        self.goalPose = odom_data

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

        self.eleviationImage = cv2.resize(cv_image, (200, 200))



    def returnData(self):
        return self.depthImage, self.eleviationImage, self.currentPose, self.goalPose

    def main( self):
        '''Initializes and cleanup ros node'''
        #rospy.signal_shutdown("shut de fuck down")
        print("rospy.init_node('GETjag_"+ str(self.number) +"_drl_gaz_robot_env_wrapper_worker')");
        rospy.init_node('GETjag_'+ str(self.number) +'_drl_gaz_robot_env_wrapper_worker')
        self.robotMovementPub = rospy.Publisher("/GETjag" + str(self.number) + "/cmd_vel", Twist, queue_size=10)
        self.robotPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/odom", Odometry, self.robotCallback)
        self.goalPoseSub = rospy.Subscriber("/GETjag" + str(self.number) + "/goal_pose", Odometry, self.goalCallback)
        self.depthImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/xtion/depth/image_raw", Image,
                                              self.depthImageCallback)
        # self.elevImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_map_image", Image, self.elevImageCallback)
        self.elevImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_map_image", Image,
                                             self.elevImageCallback, queue_size=1)

        #rospy.spin()

    def shoutdown_node(self):
        rospy.signal_shutdown("because reason")
