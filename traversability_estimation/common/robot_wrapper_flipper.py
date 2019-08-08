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

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

import matplotlib.pyplot as plt
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
from __builtin__ import True

class image_converter():
    def __init__(self,number):
        self.number = number
        #if(number==1):
        self.bridge = CvBridge()

        self.flipperVelFront = Float64()
        self.flipperVelRear = Float64()
        self.startStopRobot = Bool()
        self.startStopRobot.data = False;
        self.VERBOSE = True
        self.robotGroundMap = np.zeros((28, 28), dtype = "uint16")
        self.currentPose = Odometry()
        self.imu_data = Imu()
        self.accelZ = 0

        self.acceleration_to_high = False
        self.tip_over_angle = math.pi/4 + math.pi/6
        self.robot_flip_over = False
        self.biggestangularAccelz = 0
        self.goalPose = Odometry()
        self.flipperPoseFront = JointState()
        self.flipperPoseRear = JointState()
        self.countPub = 0
        self.robotAction = 7
        self.main()
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
        angularAccelz = self.imu_data.linear_acceleration.z
        maxAngularAccel = 10
        if(self.number ==1):
            if(angularAccelz> self.biggestangularAccelz):
                self.biggestangularAccelz = angularAccelz
            if (angularAccelz >= maxAngularAccel):
                #self.acceleration_to_high = True
                self.accelZ = angularAccelz


    # return the eleviation map image with [200,200] Pixel and saves it to a global variable
    def robotGroundMapCallback(self, map_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(map_data)
        except CvBridgeError as e:
            print(e)
        height, width = cv_image.shape
        if(height == 0 or width == 0):
           cv_image = np.zeros((28, 28), dtype = "uint16")
        self.robotGroundMap = np.asarray(cv2.resize(cv_image, (28, 28)))



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
        self.robotGroundMapSub = rospy.Subscriber("/GETjag" + str(self.number) + "/elevation_robot_ground_map", Image,
                                             self.robotGroundMapCallback, queue_size=1)

        self.robotImuCallback = rospy.Subscriber("/GETjag" + str(self.number) + "/imu/data", Imu,
                                             self.imuCallback, queue_size=1)
    def shoutdown_node(self):
        rospy.signal_shutdown("because reasons")
