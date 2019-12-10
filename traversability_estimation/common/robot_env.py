# !/usr/bin/env python2
# -*- coding: utf-8

import numpy as np
from collections import deque  # Ordered collection with ends

import time

import math
from time import sleep
from multiprocessing import Process, Pipe

import rospy

from nav_msgs.msg import Odometry

from robot_wrapper import image_converter
from tf.transformations import euler_from_quaternion
import gym

import signal
import sys
import matplotlib.pyplot as plt

# skipped your comments for readability
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#from mock.mock import self
from __builtin__ import False


# This class proveds the interaction between the robot and the DQN Agent
class robotEnv():
    def __init__(self, number=1):
        self.actions = [1, 2, 3, 4]
        self.number = number
        self.maxDistanz = 10
        self.currentPose = Odometry()
        self.goalPose = Odometry()
        self.ic = image_converter(self.number)
        # thread = threading.Thread(target=self.ic.main())
        # thread.start()

        self.availableActionsSize = 5
        self.EpisodeLength = 100
        self.stepCounter = 0
        self.episodeFinished = False
        # Variables to calculate the reward
        self.deltaDist = 0.20
        self.discountFactorMue = 0.1
        self.closestDistance = 0
        self.startGoalDistance = 0
        self.lastDistance = 0
        self.startTime = 0

        self.lastTime = 0
        self.explored_last = 0
        self.delta_x_memory = Memory(10)
        self.delta_y_memory = Memory(10)
        self.delta_vel_memory = Memory(10)

        self.delta_set_vel_lin_memory = Memory(10)
        self.delta_set_vel_angl_memory = Memory(10)

        self.observation_space = []
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(200, 200), dtype=np.float32))
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(84, 84), dtype=np.float32))
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(9, 1), dtype=np.float32))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, 1), dtype=np.float32)
        self.total_reward = 0
        signal.signal(signal.SIGINT, self.signal_handler)
        self.number_of_epsiodes = 0
        self.number_reached_goal = 0
        self.me = "forpythontest2@gmail.com"
        self.my_password = r"rutWXjJkPItlmGwRYB9J"
        self.you = "chrf@notes.upb.de"

        self.msg = MIMEMultipart('alternative')
        self.msg['Subject'] = "Alert"
        self.msg['From'] = self.me
        self.msg['To'] = self.you

        html = '<html><body><p>Hi, Your programm has stop pleas start it again!</p></body></html>'
        part2 = MIMEText(html, 'html')

        self.msg.attach(part2)

        self.countResetZeroVel = 0
        self.newRewards = False

        self.cell_states = cell_checker(10.0)
        self.ic.stop()

    def set_episode_length(self, EpisodeLength):
        self.EpisodeLength = EpisodeLength

    # Takes a step inside the enviroment according to the proposed action and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel),
    # orientation (Euler notation [roll, pitch, yaw]) and the archived reward for the new state
    def step(self, action):
        # action[0] = abs(action[0])

        lin, angl = self.ic.setAction(action)
        self.delta_set_vel_lin_memory.add(lin);
        self.delta_set_vel_angl_memory.add(angl);
        self.stepCounter += 1

        sleep(0.3)

        reward, d = self.clcReward()

        if (d):
            self.total_reward = 0
        else:
            self.total_reward += reward

        eleviationData, depthData, goalPose = self.get_state()
        self.ic.stop()

        return eleviationData, depthData, goalPose, reward, d, info

    # transform messures to states and return them
    def get_state(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose = self.ic.returnData()

        # creat np arrays as input of the drl agent

        roll, pitch, yaw = self.clcRPYGoalPose(self.goalPose.pose.pose.position)

        goalOrientation = np.asarray([roll, pitch, yaw], dtype=np.float32)
        goalPosition = self.goalPose.pose.pose.position;
        goalPosition = np.array([goalPosition.x, goalPosition.y, goalPosition.z], dtype=np.float32)

        currentPosition = self.ic.currentRobotPose.pose.pose.position
        currentPosition = np.array([currentPosition.x, currentPosition.y, currentPosition.z], dtype=np.float32)

        eleviationData = np.asarray(eleviationImage, dtype=np.float32)
        depthData = np.asarray(depthImage, dtype=np.float32)

        # norrm input data between -1 and 1
        depthData = np.divide(depthData, 10)

        eleviationData = eleviationData[0]

        goalOrientation = np.divide(goalOrientation, math.pi)
        goalPosition = np.divide(goalPosition, self.maxDistanz)
        currentPosition =  np.divide(currentPosition, self.maxDistanz)

        goalPose = np.concatenate((goalPosition, goalOrientation), axis=None)
        goalPose = np.concatenate((goalPose, currentPosition), axis=None)
        depthData = np.nan_to_num(depthData)
        eleviationData = np.nan_to_num(eleviationData)

        eleviationData = np.asarray(eleviationData, dtype=np.float32)
        depthData = np.asarray(depthData, dtype=np.float32)
        goalPose = np.asarray(goalPose, dtype=np.float32)
        # replace nan values with 0

        eleviationData.reshape(1, 200, 200)
        depthData.reshape(1, 84, 84)
        goalPose.reshape(1, 9)

        return eleviationData, depthData, goalPose

    def get_available_actions_size(self):
        return self.availableActionsSize

    # Restet the enviroment and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel) and
    # orientation (Euler notation [roll, pitch, yaw])of the oberservation in the new enviroment
    def reset(self):
        # reset the model and replace the robot at a random location
        self.ic.stop()
        valiedEnv = False
        self.ic.clcMean()
        # resest the step counter

        self.stepCounter = 0
        # reset the buffer with the velocitys
        self.delta_x_memory.resetBuffer()
        self.delta_y_memory.resetBuffer()
        self.delta_vel_memory.resetBuffer()
        self.delta_set_vel_lin_memory.resetBuffer();
        self.delta_set_vel_angl_memory.resetBuffer();
        # repead until the robot is in a valite starting position
        count_valied_state = 0
        while valiedEnv == False:

            rospy.set_param("/GETjag" + str(self.number) + "/End_of_episode", True)
            rospy.set_param("/GETjag" + str(self.number) + "/Ready_to_Start_DRL_Agent", False)
            sleep(1)

            waitForReset = False
            # wait that the enviroment is build up and the agent is ready
            count_reset = 0
            while waitForReset == False:
                Ready = rospy.get_param("/GETjag" + str(self.number) + "/Ready_to_Start_DRL_Agent")
                if Ready:
                    waitForReset = True
                count_reset += 1
                if (count_reset % 1000 == 0):
                    s = smtplib.SMTP_SSL('smtp.gmail.com')
                    s.login(self.me, self.my_password)
                    s.sendmail(self.me, self.you, self.msg.as_string())
                    s.quit()
                    print("worker_" + str(self.number) + " stucked at waiting for Ready_to_Start_DRL_agent");
                sleep(0.2)

            depthImage, eleviationImage, self.currentPose, self.goalPose = self.ic.returnData()
            roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

            position_q = self.currentPose.pose.pose.position

            # cheakk if the roboter is in a valide starting position
            if roll <= math.pi / 4 and roll >= -math.pi / 4 and pitch <= math.pi / 4 and pitch >= -math.pi / 4 and position_q.z < 0.7:
                valiedEnv = True
                sleep(0.2)
            count_valied_state += 1
            if (count_valied_state % 1000 == 0):
                s = smtplib.SMTP_SSL('smtp.gmail.com')
                s.login(self.me, self.my_password)
                s.sendmail(self.me, self.you, self.msg.as_string())
                s.quit()
                print("worker_" + str(self.number) + " stucked at cheak for valid state");

        sleep(0.2)

        self.startGoalDistance = self.clcDistance(self.goalPose.pose.pose.position)
        self.closestDistance = self.startGoalDistance
        self.startTime = time.time()
        self.lastTime = self.startTime

        self.number_of_epsiodes += 1

        self.cell_states.reset_possible_cells()
        self.ic.reach_the_goal = False
        self.newRewards = False
        self.ic.clcMean()

        return self.get_state()

    def clcReward(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose = self.ic.returnData()
        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)
        if(self.number==1):
            print("roll" + str(abs(roll)))
            print("abs(pitch)" + str(abs(pitch)))

        explor = False

        currenVel = self.currentPose.twist.twist.linear.x

        currentdistance = self.clcDistance(self.goalPose.pose.pose.position)
        exploredNewArea = False
        goalAreaReached = False
        if (self.stepCounter == 1):
            self.cell_states.get_possible_cells(self.ic.currentRobotPose.pose.pose.position, self.goalPose.pose.pose.position)
            self.closestDistance = currentdistance
        else:
            exploredNewArea, goalAreaReached = self.cell_states.get_possible_cells(self.ic.currentRobotPose.pose.pose.position, self.goalPose.pose.pose.position)
            if(goalAreaReached):
                self.newRewards = True
                self.closestDistance = currentdistance



        currentTime = time.time()
        currentTime = currentTime - self.startTime
        deltaTime = currentTime - self.lastTime

        self.delta_x_memory.add(self.currentPose.pose.pose.position.x)
        self.delta_y_memory.add(self.currentPose.pose.pose.position.y)

        if(self.number==1):
            print("abs(currenVel)" + str(abs(currenVel)))
        self.delta_vel_memory.add(abs(currenVel))

        mean_delta_vel = self.delta_vel_memory.mean();

        EndEpisode = False;

        reward = 0

        if rospy.get_param("/GETjag" + str(self.number) + "/Error_in_simulator"):
            EndEpisode = True

            rospy.set_param("/GETjag" + str(self.number) + "/Error_in_simulator", False)

        if currentdistance < self.closestDistance: #and self.newRewards:

            reward = self.discountFactorMue * (self.closestDistance - currentdistance)

            self.closestDistance = currentdistance

        if explor == True:
            if exploredNewArea: #and not self.newRewards:
                reward += 0.06
            elif currentdistance <= self.lastDistance:
                reward = 0.5 + (self.startGoalDistance / currentTime)

        if roll >= math.pi / 4 or roll <= -math.pi / 4 or pitch >= math.pi / 4 or pitch <= -math.pi / 4:
            reward = -0.5
            EndEpisode = True
            self.countResetZeroVel = 0

        if (mean_delta_vel <= 0.015):
            reward = -0.5
            EndEpisode = True
            self.countResetZeroVel += 1


        if self.ic.reach_the_goal:
            reward = 0.5  + (self.startGoalDistance * 10 / self.stepCounter)

            print("reached Goal, reward: " + str(reward) + ", stepCounter: " + str(self.stepCounter) + ", EpisodeLength: " + str(self.EpisodeLength))
            EndEpisode = True
            self.number_reached_goal
            self.ic.reach_the_goal = False

        if self.stepCounter >= self.EpisodeLength:
            EndEpisode = True
            self.countResetZeroVel = 0

        if (np.isnan(currenVel)):
            rospy.set_param("/GETjag" + str(self.number) + "/Error_in_simulator", True)
            print("output_gazebo.txt" + "w")

            file = open("Gazebo Script/output_gazebo.txt", "w")
            file.write("Unable to set value")
            file.close()

        self.lastDistance = currentdistance
        self.lastTime = currentTime

        self.episodeFinished = EndEpisode
        reward = np.float32(reward)

        if (EndEpisode):
            self.explored_last = 0
        return reward, EndEpisode

    def stopSteps(self):
        self.ic.stop()

    def is_episode_finished(self):
        return self.episodeFinished

    def endEnv(self):
        rospy.set_param("/GETjag" + str(self.number) + "/End_of_enviroment", True)

    def clcRPYGoalPose(self, goalPose):
        yaw = self.clcAngle(goalPose.x, goalPose.y)
        roll = math.atan(goalPose.z / goalPose.x)
        pitch = math.atan(goalPose.z / goalPose.y)
        return roll, pitch, yaw

    def clcAngle(self, v1, v2):
        if (v1 > 0):
            return math.atan(v2 / v1)
        elif (v1 < 0 and v2 < 0):
            return -math.pi / 2.0 - (math.pi / 2.0 - math.atan(v2 / v1))
        elif (v1 < 0 and v2 > 0):
            return math.pi / 2.0 + math.pi / 2 + math.atan(v2 / v1)
        elif (v2 == 0):
            return 0
        elif (v2 < 0):
            return math.pi / 2
        else:
            return -math.pi / 2

    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return euler_from_quaternion(orientation_list)

    def return_times_reached_goal(self, start, goal):
        return self.number_of_epsiodes, self.number_reached_goal

    def clcDistance(self, goal):
        distance = math.sqrt(pow((goal.x), 2) + pow((goal.y), 2))
        return distance

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.ic.shoutdown_node();
        sys.exit(0)

# cheaks zells for exploring areas
class cell_checker():
    def __init__(self, size):
        self.valueX = []
        self.valueY = []
        self.env_size = size
        self.range = self.env_size / 8.0

    def get_possible_cells(self, pose, goalPose):
        xInRange = False
        goalInRange = False

        for i in range(0, len(self.valueX), 1):
            if (abs(pose.x - self.valueX[i]) < self.range and abs(pose.y - self.valueY[i]) < self.range):
                del self.valueX[i]
                del self.valueY[i]
                if (abs(goalPose.x) < self.range and abs(goalPose.y) < self.range):
                    goalInRange = True
                xInRange = True
                break
        return xInRange, goalInRange

    def reset_possible_cells(self):
        self.valueX = []
        self.valueY = []

        for x in np.arange(self.env_size / 8.0, self.env_size, self.env_size / 4.0):
            for y in np.arange(self.env_size / 8.0, self.env_size, self.env_size / 4.0):
                self.valueX.append(x - self.env_size / 2.0)
                self.valueY.append(y - self.env_size / 2.0)

## caclulate mean and std

class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.counter = 1
        self.returnMean = False

    def add(self, value):
        self.buffer.append(value)
        if (self.counter < self.size):
            self.counter += 1
        else:
            self.returnMean = True

    def resetBuffer(self):
        self.counter = 0
        self.returnMean = False
        self.buffer = deque(maxlen=self.size)

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

    def returnNumpyArray(self):
        return np.asarray(self.buffer)