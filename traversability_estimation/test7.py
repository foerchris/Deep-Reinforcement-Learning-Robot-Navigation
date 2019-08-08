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

valueX = []
valueY = []
env_size = 10


for x in np.arange ( env_size/8.0, env_size,  env_size/4.0):
    for y in np.arange ( env_size/8.0, env_size,  env_size/4.0):
        valueX.append(x - env_size/2.0)
        valueY.append(y - env_size/2.0)

print(valueX)
print(valueY)