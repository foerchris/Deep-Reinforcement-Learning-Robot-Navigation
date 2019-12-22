#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
import numpy as np
from collections import deque  # Ordered collection with ends
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

        self.mean_angl_vel_y = Memory(5)
        self.mean_angl_vel_y.resetBuffer()

        self.mean_flipper_front_angl_vel_y = Memory(5)
        self.mean_flipper_front_angl_vel_y.resetBuffer()

        self.mean_flipper_rear_angl_vel_y = Memory(5)
        self.mean_flipper_rear_angl_vel_y.resetBuffer()
        self.startStopRobotPub.publish(True)

    '''
    stop all robots
    '''
    def stop(self):
        #self.flipperVelFront = self.flipperPoseFront.current_pos
        #self.flipperVelRear = self.flipperPoseRear.current_pos

        self.flipperVelFront =0
        self.flipperVelRear = 0

    '''
    set the action which should executed
    @ param action
    '''
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

        #self.flipperVelFront = action[0]*(math.pi/2)
        #self.flipperVelRear = action[1]*(math.pi/2)

        self.flipperVelFront = action[0]
        self.flipperVelRear = action[1]


    '''
    get current robot pose
    @ param odom_data
    '''
    def robotCallback(self,odom_data):
        self.currentPose = odom_data

        self.robotFlipperFrontPub.publish( self.flipperVelFront)
        self.robotFlipperRearPub.publish( self.flipperVelRear)

        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

        if roll>=self.tip_over_angle or roll<=-self.tip_over_angle or pitch>=self.tip_over_angle or pitch<=-self.tip_over_angle:
            self.robot_flip_over = True

    '''
     get flipper front position callback 
     @ param odom_data
     '''
    def flipperFrontPose(self, flipper_pose):
        self.flipperPoseFront = flipper_pose
        self.mean_flipper_front_angl_vel_y.addPosition(flipper_pose.current_pos)

    '''
     get flipper rear position callback 
     @ param odom_data
     '''
    def flipperRearPose(self, flipper_pose):
        self.flipperPoseRear = flipper_pose
        self.mean_flipper_rear_angl_vel_y.addPosition(flipper_pose.current_pos)



    '''
    callback to get the goal-robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
    @ param odom_data
    '''
    def goalCallback(self, odom_data):
        self.goalPose = odom_data

    '''
    callback to get robot orientation
    @ param odom_data
    '''
    def imuCallback(self, imu_data):
        self.imu_data = imu_data
        #self.accelerations.append(abs(imu_data.))

        angularAccely = self.mean_angl_vel_y.addVelocity(imu_data.angular_velocity.y)

        maxAngularAccel = 65

        if (angularAccely >= maxAngularAccel):
            self.acceleration_to_high = angularAccely
            self.accelZ = angularAccely

    '''
    calculate mean values for linear and angular veloicity 
    @ return mean- linear and angular velocity and acceleration
    '''
    def clcMean(self):

        print(self.mean_angl_vel_y.getvelovitys())
        meanAgnleVelY, meanAngleAccY = self.mean_angl_vel_y.totalMean()

        meanFlipperFrontAgnleVelY, meanFlipperFrontAgnleAccY = self.mean_flipper_front_angl_vel_y.totalMean()

        meanFlipperRearAgnleVelY, meanFlipperRearAgnleAccY = self.mean_flipper_rear_angl_vel_y.totalMean()

        file = open("Gazebo Script/messungen/flipper_acc_measures.txt", "w")
        file.write("meanAgnleVelY = " + str(meanAgnleVelY) + "\n")
        file.write("meanAngleAccY = " + str(meanAngleAccY) + "\n")

        file.write("meanFlipperFrontAgnleVelY = " + str(meanFlipperFrontAgnleVelY) + "\n")
        file.write("meanFlipperFrontAgnleAccY = " + str(meanFlipperFrontAgnleAccY) + "\n")

        file.write("meanFlipperRearAgnleVelY = " + str(meanFlipperRearAgnleVelY) + "\n")
        file.write("meanFlipperRearAgnleAccY = " + str(meanFlipperRearAgnleAccY) + "\n")

        file.close()
        return meanLinVel, meanAngVel, meanLinAcc, meanAngAcc

    '''
    reset the the buffers
    '''
    def reset(self):
        self.mean_angl_vel_y.resetBuffer()
        self.mean_flipper_front_angl_vel_y.resetBuffer()
        self.mean_flipper_rear_angl_vel_y.resetBuffer()

    '''
    return the eleviation map image with [200,200] Pixel and saves it to a global variable
    @ param map_data
    '''
    def robotGroundMapCallback(self, map_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(map_data)
        except CvBridgeError as e:
            print(e)

        image = cv_image[:, :, 0]

        self.robotGroundMap = np.asarray(cv2.resize(image, (28, 28)))

    '''
    return states
    @ return depth image, elevation map image, current robot pose, goal pose
    '''
    def returnData(self):
        return self.robotGroundMap, self.currentPose, self.goalPose, self.flipperPoseFront, self.flipperPoseRear

    '''
    return roll pitch yaw robot orientation
    @ return roll pitch yaw orientation
    '''
    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return  euler_from_quaternion(orientation_list)

    def main( self):
        '''Initializes and cleanup ros node'''
        print("rospy.init_node('GETjag_"+ str(self.number) +"_drl_gaz_robot_env_wrapper_flipper_worker')");
        rospy.init_node('GETjag_'+ str(self.number) +'_drl_gaz_robot_env_wrapper_flipper_worker')

        self.startStopRobotPub = rospy.Publisher("/GETjag" + str(self.number) + "/start_stop_robot", Bool, queue_size=10)
        self.robotFlipperFrontPub = rospy.Publisher("/GETjag" + str(self.number) + "/flipper_front_controller/cmd_vel", Float64, queue_size=10)
        self.robotFlipperRearPub = rospy.Publisher("/GETjag" + str(self.number) + "/flipper_rear_controller/cmd_vel", Float64, queue_size=10)

       # self.robotFlipperFrontPub = rospy.Publisher("GETjag" + str(self.number) + "/flipper_front_controller/command", Float64, queue_size=10)
       # self.robotFlipperRearPub = rospy.Publisher("GETjag" + str(self.number) + "/flipper_rear_controller/command", Float64, queue_size=10)

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

