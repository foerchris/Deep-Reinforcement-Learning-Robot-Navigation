#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
import numpy as np
import math as m
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from dynamixel_msgs.msg import JointState
from sensor_msgs.msg import Image

from std_msgs.msg import Float64
from std_msgs.msg import Float32

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Range

from tf.transformations import euler_from_quaternion
import time

import matplotlib.pyplot as plt
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from __builtin__ import True
from math import fabs

class image_converter():
    def __init__(self,number):
        self.number = number
        #if(number==1):
        self.bridge = CvBridge()
        self.bridge2 = CvBridge()

        self.flipperVelFront = Float64()
        self.flipperVelRear = Float64()

        self.frontUss = float()
        self.mid1Uss = float()
        self.mid2Uss = float()
        self.rearUss = float()

        self.startStopRobot = Bool()
        self.startStopRobot.data = False;
        self.VERBOSE = True
        self.robotGroundMap = np.zeros((28, 28), dtype = "uint16")
        self.currentPose = Odometry()
        self.imu_data = Imu()
        self.accelZ = 0

        self.acceleration_to_high = 0
        self.tip_over_angle = math.pi/4 + math.pi/6
        self.robot_flip_over = False
        self.biggestangularAccelz = 0
        self.goalPose = Odometry()
        self.flipperPoseFront = JointState()
        self.flipperPoseRear = JointState()
        self.countPub = 0
        self.robotAction = 7
        self.main()
        self.last_angular_velocity_y = 0
        self.last_nsecs = 0
        self.last_time = time.time()

    def stop(self):
        self.flipperVelFront = 0
        self.flipperVelRear = 0

    def setAction(self, action):
        bound = 1.22

        frontPose = self.flipperPoseFront.current_pos
        rearPose = self.flipperPoseRear.current_pos


        if(frontPose >= bound and action[0]>0):
            action[0] = -action[0]
        elif(frontPose <= -bound and action[0]<0):
            action[0] = -action[0]

        if (rearPose >= bound and action[1]>0):
            action[1] = -action[1]
        elif (rearPose <= -bound and action[1]<0):
            action[1] = -action[1]


        self.flipperVelFront = action[0]
        self.flipperVelRear = action[1]


    def frontUSS_Callback(self, uss_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        self.frontUss=uss_data.range

    def mid1USS_Callback(self, uss_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        self.mid1Uss=uss_data.range

    def mid2USS_Callback(self, uss_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        self.mid1Uss=uss_data.range

    def rearUSS_Callback(self, uss_data):
        '''Callback function of subscribed topic.
              Here images get converted and features detected'''
        self.rearUss=uss_data.range


    # callback to get the current robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    # also transmits the robot action as velocities
    def robotCallback(self,odom_data):
        self.currentPose = odom_data
        #if(self.number==1):
        #    self.startStopRobot.data = False
        #    self.flipperVelFront = 0
        #    self.flipperVelRear = 0

        self.startStopRobotPub.publish( self.startStopRobot)
        self.robotFlipperFrontPub.publish( self.flipperVelFront)
        self.robotFlipperRearPub.publish( self.flipperVelRear)

        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

        if roll>=self.tip_over_angle or roll<=-self.tip_over_angle or pitch>=self.tip_over_angle or pitch<=-self.tip_over_angle:
            self.robot_flip_over = True

    def flipperFrontPose(self, flipper_pose):
        self.flipperPoseFront = flipper_pose

    def flipperRearPose(self, flipper_pose):
        self.flipperPoseRear = flipper_pose


    # callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    def goalCallback(self, odom_data):
        self.goalPose = odom_data

    # callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    def imuCallback(self, imu_data):
        self.imu_data = imu_data

        deltaTime = time.time() - self.last_time


        if(abs(deltaTime - 0.023) < 0.02 ):
          ##  print("deltaTime2: " + str(deltaTime))

            angularAccely = abs(self.imu_data.angular_velocity.y - self.last_angular_velocity_y)/deltaTime
            maxAngularAccel = 65
            #if(self.number ==1):
              ##  print("angularAccely: " + str(angularAccely))

            #    if(angularAccely> self.biggestangularAccelz):
            #        print("self.imu_data.angular_velocity.y: " + str(self.imu_data.angular_velocity.y))
            #        print("self.last_angular_velocity_y: " + str(self.last_angular_velocity_y))

            #        print("angularAccelz: " + str(angularAccely))
             #       self.biggestangularAccelz = angularAccely

            if (angularAccely >= maxAngularAccel):
             #   if(self.number == 1):
             #       print("self.acceleration_to_high: " + str(angularAccely))

                self.acceleration_to_high = angularAccely
                self.accelZ = angularAccely

        self.last_time = time.time()

        self.last_angular_velocity_y = self.imu_data.angular_velocity.y
        self.last_nsecs = self.imu_data.header.stamp.nsecs

    def depthImageCallback(self, depth_data):
        #print("depthImageCallback");
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_data,"32FC1")
        except CvBridgeError as e:
            print(e)
        #plt.imshow(cv_image,cmap="gray")
        #plt.show()

        cv_image = np.asarray(cv_image, dtype=np.float32)
        cv_image = np.nan_to_num(cv_image)

        #if(self.number ==1):
           # print(cv_image.shape)
          #  plt.imshow(cv_image,cmap="gray")
          #  plt.show()
         #   print(cv_image)

        h=150
        w=640

        y=cv_image.shape[0]-h
        x=cv_image.shape[1]-w

        crop_img = cv_image[y:y+h, x:x+w]
        #if(self.number ==1):
           # plt.imshow(crop_img,cmap="gray")
          #  plt.show()

         #   print(crop_img.shape)

        #plt.imshow(crop_img,cmap="gray")
        #plt.show()
        #crop_img[crop_img > 5] =  5

        self.depthImage = cv2.resize(crop_img, (28, 28))
        #if(self.number ==1):
          #  plt.imshow(self.depthImage,cmap="gray")
          #  plt.show()

            #print(crop_img.shape)



    def returnData(self):
        return self.robotGroundMap, self.currentPose, self.goalPose, self.flipperPoseFront, self.flipperPoseRear

    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return  euler_from_quaternion(orientation_list)

    def main( self):
        '''Initializes and cleanup ros node'''
        #rospy.signal_shutdown("shut de fuck down")
        print("rospy.init_node('GETjag_"+ str(self.number) +"_drl_gaz_robot_env_wrapper_worker')");
        rospy.init_node('GETjag_'+ str(self.number) +'_drl_gaz_robot_env_wrapper_worker')

        self.startStopRobotPub = rospy.Publisher("/GETjag" + str(self.number) + "/start_stop_robot", Bool, queue_size=10)
        self.robotFlipperFrontPub = rospy.Publisher("/GETjag" + str(self.number) + "/flipper_front_controller/cmd_vel", Float64, queue_size=10)
        self.robotFlipperRearPub = rospy.Publisher("/GETjag" + str(self.number) + "/flipper_rear_controller/cmd_vel", Float64, queue_size=10)

        self.robotPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/odom", Odometry, self.robotCallback)
        self.flipperFrontPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/flipper_front_controller/state", JointState, self.flipperFrontPose)
        self.flipperRearPoseSub = rospy.Subscriber("GETjag" + str(self.number) + "/flipper_rear_controller/state", JointState, self.flipperRearPose)

        self.goalPoseSub = rospy.Subscriber("/GETjag" + str(self.number) + "/goal_pose", Odometry, self.goalCallback)

        frontUSS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_1", Range,
                                         self.frontUSS_Callback, queue_size=1)

        mid1USS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_2", Range,
                                         self.mid1USS_Callback, queue_size=1)
        mid2USS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_3", Range,
                                         self.mid2USS_Callback, queue_size=1)
        rearUSS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_4", Range,
                                         self.rearUSS_Callback, queue_size=1)

        self.depthImageSub = rospy.Subscriber("/GETjag" + str(self.number) + "/xtion/depth/image_raw", Image,
                                             self.depthImageCallback, queue_size=1)

        self.robotImuCallback = rospy.Subscriber("/GETjag" + str(self.number) + "/imu/data", Imu,
                                             self.imuCallback, queue_size=1)
    def shoutdown_node(self):
        rospy.signal_shutdown("because reasons")
