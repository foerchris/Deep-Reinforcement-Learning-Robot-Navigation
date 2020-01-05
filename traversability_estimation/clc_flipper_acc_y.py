#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import numpy as np
from collections import deque  # Ordered collection with ends
import time
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class image_converter():
    def __init__(self):

        self.main()

        self.mean_angl_vel_y = Memory(4)
        self.mean_angl_vel_y.resetBuffer()
        self.currentPose = Odometry()

        self.start_movement = False

    def stop(self):
        self.mean_angl_vel_y.linear.x = 0


    def clcMean(self):
        meanVel, meancc, maxVel, maxAcc = self.mean_angl_vel_y.totalMean()

        print("meanVel = " + str(meanVel) + "\n")
        print("meancc = " + str(meancc) + "\n")
        print("maxVel = " + str(maxVel) + "\n")
        print("maxAcc = " + str(maxAcc) + "\n")


    def imuCallback(self, imu_data):
        self.imu_data = imu_data
        # self.accelerations.append(abs(imu_data.))
        if (self.currentPose.twist.twist.linear.x >= 0.05):
            self.start_movement = True

        if(self.start_movement == True):
            print("imu_data.angular_velocity.y = " + str(imu_data.angular_velocity.y) + "\n")
            if(self.currentPose.twist.twist.linear.x<=0.01):
                self.start_movement = False
                self.clcMean()
                return
            angularAccely = self.mean_angl_vel_y.addVelocity(imu_data.angular_velocity.y)
            angularVel = self.mean_angl_vel_y.mean()

            print("angularVel = " + str(angularVel) + "\n")
            print("angularAccely = " + str(angularAccely) + "\n")


    def robotPoseCallback(self,odom_data):
        self.currentPose = odom_data
        #print("odom_data.twist.twist.linear.x" + str(odom_data.twist.twist.linear.x))

    def main( self):
        '''Initializes and cleanup ros node'''
        #rospy.signal_shutdown("shut de fuck down")
        print("rospy.init_node('clc_flipper_acc_node')");
        rospy.init_node("clc_flipper_acc_node")

        self.robotPoseSub = rospy.Subscriber("GETjag" + str(1) + "/odom", Odometry, self.robotPoseCallback)
        self.robotImuCallback = rospy.Subscriber("/GETjag" + str(1) + "/imu/data", Imu, self.imuCallback, queue_size=1)
        #self.robotPoseSub = rospy.Subscriber("GETjag/odom", Odometry, self.robotPoseCallback)
        #self.robotImuCallback = rospy.Subscriber("/GETjag/imu/data", Imu, self.imuCallback, queue_size=1)

        self.mean_angl_vel_y = Memory(2)
        self.mean_angl_vel_y.resetBuffer()
        self.start_movement = False

        self.currentPose = Odometry()

        rospy.spin()

    def shoutdown_node(self):
        rospy.signal_shutdown("because reason")

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
       # self.reset_buffer = True
        return np.mean(self.velovitys), np.mean(self.accelerations), np.max(self.velovitys), np.max(self.accelerations)

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

ic = image_converter()
