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

values = np.zeros([20,5])

print(values.shape)

next_value = np.array([1, 2, 3.0 ,4 ,3])
print(next_value.shape)

values = values + [next_value]
print(values.shape)
print(values)
