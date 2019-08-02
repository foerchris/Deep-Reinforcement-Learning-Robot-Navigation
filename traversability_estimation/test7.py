import rospy
import roslib
import numpy as np

from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image

from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

from nav_msgs.msg import Odometry
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError

number = 1


def returnRollPitchYaw(orientation):
    orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
    return euler_from_quaternion(orientation_list)

count = 0
# also transmits the robot action as velocities
def robotCallback(odom_data):
    global count
    # if(count%15==0):
    #     roll, pitch, yaw = returnRollPitchYaw(odom_data.pose.pose.orientation)
    #     goalOrientation = np.asarray([roll, pitch, yaw], dtype=np.float32)
    #     goalPosition = odom_data.pose.pose.position;
    #     goalPosition = np.array([goalPosition.x, goalPosition.y, goalPosition.z], dtype=np.float32)
    #
    #     print("odomOrientation:\t roll=" + str(goalOrientation[0] / math.pi * 180.0) + "\t pitch=" + str(
    #     goalOrientation[1] / math.pi * 180.0) + "\t yaw=" + str(goalOrientation[2] / math.pi * 180.0) + "\n")
    #     print("odomPosition:\t x=" + str(goalPosition[0]) + "\t y=" + str(goalPosition[1]) + "\t z=" + str(
    #     goalPosition[2]) + "\n")

    count+=1


# callback to get the goal robot pose as position (x,y,z) and orientation as quaternion (x,y,z,w)
def goalCallback(odom_data):
    global count
    if(count%30==0):

        roll, pitch, yaw = returnRollPitchYaw(odom_data.pose.pose.orientation)
        goalOrientation = np.asarray([roll, pitch, yaw], dtype=np.float32)
        goalPosition = odom_data.pose.pose.position;
        goalPosition = np.array([goalPosition.x, goalPosition.y, goalPosition.z], dtype=np.float32)

        print("goalPosition:\t x="+str(goalPosition[0]) + "\t y=" + str(goalPosition[1]) + "\t z="+ str(goalPosition[2]) + "\n")

        print("yaw="+str(clcAngle(goalPosition[0],goalPosition[1])/math.pi*180))

def clcAngle(v1,v2):
    if(v1>0):
        return math.atan(v2/v1)
    elif(v1<0 and v2<0):
        return -math.pi/2.0-(math.pi/2.0-math.atan(v2/v1))
    elif(v1<0 and v2>0):
        return math.pi/2.0 + math.pi/2 + math.atan(v2/v1)
    elif(v2==0):
        return 0
    elif(v2<0):
        return math.pi/2
    else:
        return -math.pi/2



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)

    print("rospy.init_node('GETjag_" + str(number) + "_drl_gaz_robot_env_wrapper_worker')");
    robotPoseSub = rospy.Subscriber("GETjag" + str(number) + "/odom", Odometry, robotCallback)
    goalPoseSub = rospy.Subscriber("/GETjag" + str(number) + "/goal_pose", Odometry, goalCallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
