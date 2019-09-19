#!/usr/bin/env python2
# -*- coding: utf-8

import rospy
import roslib
import numpy as np

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from tf import TransformListener
from sensor_msgs.msg import Image

from std_msgs.msg import String

from nav_msgs.msg import Odometry

import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
import matplotlib.pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan
import numpy as np

bridge = CvBridge()


def elevImageCallback(map_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    try:
        cv_image = bridge.imgmsg_to_cv2(map_data)
    except CvBridgeError as e:
        print(e)
    # print(cv_image.shape)
    image = cv_image[:, :, 0]

    #alpha = cv_image[:, :, 1]


    plt.imshow(image, cmap="gray")
    plt.show()
    #plt.imshow(alpha, cmap="gray")
    #plt.show()

def frontUSS_Callback(uss_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    print("frontUSS_Callback" +str(uss_data.range))
def mid1USS_Callback(uss_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    print("mid1USS_Callback" +str(uss_data.range))

def mid2USS_Callback(uss_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    print("mid2USS_Callback" +str(uss_data.range))

def rearUSS_Callback(uss_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    print("rearUSS_Callback" +str(uss_data.range))

def laserScanCallback(laser_data):
    '''Callback function of subscribed topic.
          Here images get converted and features detected'''
    print("laser_data" +str(type(laser_data.ranges)))

    laser_data = np.array(laser_data.ranges)
    print("laser_data" +str(type(laser_data)))


    print("laser_data" +str(laser_data))

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    #frontUSS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_1", Range,
    #                                     frontUSS_Callback, queue_size=1)

    #mid1USS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_2", Range,
    #                                     mid1USS_Callback, queue_size=1)

   # mid2USS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_3", Range,
   #                                      mid2USS_Callback, queue_size=1)

 #   rearUSS_Sub = rospy.Subscriber("/GETjag" + "1" +"/ground_clearance_4", Range,
    #                                     rearUSS_Callback, queue_size=1)

   # laserScanSub = rospy.Subscriber("/GETjag" + "1" +"/laser_scan_mid", LaserScan,
   #                                      laserScanCallback, queue_size=1)


    elevImageSub = rospy.Subscriber("/GETjag" + "1" +"/elevation_robot_ground_map", Image,
                                         elevImageCallback, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()