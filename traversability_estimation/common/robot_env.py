#!/usr/bin/env python2
# -*- coding: utf-8

import numpy as np
from collections import deque# Ordered collection with ends

import time

import math
from time import  sleep
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

# This class proveds the interaction between the robot and the DQN Agent
class robotEnv():
    def __init__(self, number = 1):
        self.actions = [1,2,3,4]
        self.number = number
        self.maxDistanz=10
        self.currentPose = Odometry()
        self.goalPose = Odometry()
        self.ic = image_converter(self.number)
        #thread = threading.Thread(target=self.ic.main())
        #thread.start()

        self.availableActionsSize = 5
        self.EpisodeLength = 100
        self.stepCounter=0
        self.episodeFinished = False
        # Variables to calculate the reward
        self.deltaDist = 0.20
        self.discountFactorMue = 20
        self.closestDistance = 0
        self.startGoalDistance = 0
        self.lastDistance = 0
        self.startTime =0

        self.lastTime = 0
        self.explored_last = 0
        self.delta_x_memory = Memory(10)
        self.delta_y_memory = Memory(10)
        self.delta_vel_memory = Memory(10)

        self.observation_space = []
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(200,200),dtype = np.float16))
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(84,84),dtype = np.float16))
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(6,1),dtype = np.float16))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,1), dtype = np.float16)
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

    def set_episode_length(self,EpisodeLength):
        self.EpisodeLength = EpisodeLength

    # Takes a step inside the enviroment according to the proposed action and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel),
    # orientation (Euler notation [roll, pitch, yaw]) and the archived reward for the new state
    def step(self, action):
        # here the action should be to the robot and the reward should return
        # überlegen wie ich die zeitverzögerung realisieren will
        #action[0] = abs(action[0])

        #if(self.number ==1):
          #  print("action:" + str(action))

        self.ic.setAction(action)
        self.stepCounter += 1

        sleep(0.3)

        reward, d =self.clcReward()

        if(d):
            self.total_reward = 0
        else:
            self.total_reward += reward

        #if(self.number == 1):
        #    print('reward' + str(reward))
        #    print('total_reward' + str(self.total_reward))
        #if(action[0] >= 1.0 or action[0] <=  0.05):
        #    reward += -0.01

        #if(action[1] >= 1.0 or action[1] <= -1.0):
        #    reward += -0.01

        #if(self.number ==1):
          #  print("reward:" + str(reward))

        info = {}
        eleviationData, depthData, goalPose = self.get_state()
        self.ic.stop()

        return eleviationData, depthData, goalPose, reward, d, info


    def get_state(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()

        # creat np arrays as input of the drl agent
        roll, pitch, yaw = self.returnRollPitchYaw(self.goalPose.pose.pose.orientation)
        goalOrientation = np.asarray([roll, pitch, yaw])
        goalPosition = self.goalPose.pose.pose.position;
        goalPosition = np.array([goalPosition.x, goalPosition.y, goalPosition.z])



        depthData = np.asarray(depthImage)
        eleviationData = np.asarray(eleviationImage)
        depthData = depthData.astype('float16')
        eleviationData = eleviationData.astype('float16')



        # norrm input data between -1 and 1
        depthData =  np.divide(depthData,10)
        eleviationData =  np.divide(eleviationData, 255) #255
        goalOrientation = np.divide(goalOrientation,math.pi)
        goalPosition = np.divide(goalPosition, self.maxDistanz)

        goalPose = np.concatenate((goalPosition, goalOrientation), axis=None)

        depthData=np.nan_to_num(depthData)

        eleviationData = np.asarray(eleviationData, dtype=np.float16)
        depthData = np.asarray(depthData, dtype=np.float16)
        goalPose = np.asarray(goalPose, dtype=np.float16)
        # replace nan values with 0

        eleviationData.reshape(1,200,200)
        depthData.reshape(1,84,84)
        goalPose.reshape(1,6)

        return eleviationData, depthData, goalPose

    def get_available_actions_size(self):
        return self.availableActionsSize

    # Restet the enviroment and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel) and
    # orientation (Euler notation [roll, pitch, yaw])of the oberservation in the new enviroment
    def reset(self):
        # reset the model and replace the robot at a random location
        self.ic.stop()
        valiedEnv = False

        # resest the step counter
        self.stepCounter=0
        # reset the buffer with the velocitys
        self.delta_x_memory.resetBuffer()
        self.delta_y_memory.resetBuffer()
        self.delta_vel_memory.resetBuffer()

        # repead until the robot is in a valite starting position
        count_valied_state = 0
        while valiedEnv == False:

            rospy.set_param("/GETjag"+ str(self.number) + "/End_of_episode", True)
            rospy.set_param("/GETjag" + str(self.number) + "/Ready_to_Start_DRL_Agent",False)
            sleep(1)

            waitForReset = False
            # wait that the enviroment is build up and the agent is ready
            count_reset = 0
            while waitForReset==False:
                Ready = rospy.get_param("/GETjag"+ str(self.number) + "/Ready_to_Start_DRL_Agent")
                if Ready:
                    waitForReset = True
                count_reset += 1
                if(count_reset%100 == 0):
                    s = smtplib.SMTP_SSL('smtp.gmail.com')
                    s.login(self.me, self.my_password)
                    s.sendmail(self.me, self.you, self.msg.as_string())
                    s.quit()
                    print("worker_" + str(self.number) + " stucked at waiting for Ready_to_Start_DRL_agent");
                sleep(0.2)


            depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()
            roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

            position_q = self.currentPose.pose.pose.position
            # cheakk if the roboter is in a valide starting position
            if roll <= math.pi/4 and roll >= -math.pi/4 and pitch <= math.pi/4 and pitch >= -math.pi/4 and position_q.z <0.7:
                valiedEnv = True
                sleep(0.2)
            count_valied_state +=1
            if (count_valied_state % 100 == 0):
                s = smtplib.SMTP_SSL('smtp.gmail.com')
                s.login(self.me, self.my_password)
                s.sendmail(self.me, self.you, self.msg.as_string())
                s.quit()
                print("worker_" + str(self.number) + " stucked at cheak for valid state");

        self.startGoalDistance = self.clcDistance(self.currentPose.pose.pose.position,self.goalPose.pose.pose.position)
        self.closestDistance = self.startGoalDistance
        self.startTime = time.time()
        self.lastTime = self.startTime

        sleep(0.2)
        self.number_of_epsiodes += 1
        return self.get_state()


    def clcReward(self):
        depthImage, eleviationImage, self.currentPose, self.goalPose= self.ic.returnData()
        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

        currenVel = self.currentPose.twist.twist.linear.x

        currentdistance = self.clcDistance(self.currentPose.pose.pose.position,self.goalPose.pose.pose.position)
        currentTime = time.time()
        currentTime = currentTime - self.startTime
        deltaTime = currentTime - self.lastTime


        self.delta_x_memory.add(self.currentPose.pose.pose.position.x)
        self.delta_y_memory.add(self.currentPose.pose.pose.position.y)

        self.delta_vel_memory.add(currenVel)

        var_delta_x = self.delta_x_memory.var();
        var_delta_y = self.delta_y_memory.var();

        var_delta_vel = self.delta_vel_memory.var();

        EndEpisode = False;
        reward=0

        if rospy.get_param("/GETjag"+ str(self.number) + "/Error_in_simulator"):
            EndEpisode = True

            rospy.set_param("/GETjag"+ str(self.number) + "/Error_in_simulator",False)

        if currentdistance < self.closestDistance:
            reward = self.discountFactorMue*(self.closestDistance-currentdistance)/deltaTime
            self.closestDistance = currentdistance

        elif currentdistance <= self.lastDistance:
            reward = 0.5 + (self.startGoalDistance / currentTime)

#         explored = (eleviationImage > 100).sum()
#         if (explored > self.explored_last):
#             self.explored_last = explored
#             reward += 8
#             if explored< 1000:
#                 reward += explored /200
#             else:
#                 reward += 1

        #if explored< 1000:
        #    reward += explored /1000
        #else:
        #    reward += 1

        if roll>=math.pi/4 or roll<=-math.pi/4 or pitch>=math.pi/4 or pitch<=-math.pi/4:
            reward = -0.5
            EndEpisode = True

       # if(self.number == 1):
       #     print("var_delta_vel" + str(var_delta_vel))

       # if (var_delta_x <= 1e-2 and var_delta_y <= 1e-2):
        if (var_delta_vel <= 1e-2):
             reward = -0.5
             EndEpisode = True

        if currentdistance <= self.deltaDist:
            reward = 100
            text_file = open("a3c_results.txt", "a")
            text_file.write(str(self.number) + 'reached Goal\n')
            text_file.write(str(self.number) + 'reached Goal\n')
            text_file.close()
            print("reached Goal")
            EndEpisode = True
            self.number_reached_goal


        if  self.stepCounter>=self.EpisodeLength:
            EndEpisode = True

        #print("current reward: " + str(reward))
        reward = reward*0.01
        #reward = reward*0.001
        #if(self.number == 1):
            #print("reward for step: " + str(reward))
        self.lastDistance = currentdistance
        self.lastTime = currentTime

        self.episodeFinished = EndEpisode
        reward = np.float16(reward)
       # print("reward" + str(reward))
        if(EndEpisode):
            self.explored_last = 0
        return reward, EndEpisode

    def stopSteps(self):
        self.ic.stop()

    def is_episode_finished(self):
        return self.episodeFinished

    def endEnv(self):
        #rospy.set_param("/GETjag/End_of_enviroment",True)
        rospy.set_param("/GETjag"+ str(self.number) + "/End_of_enviroment",True)

    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return  euler_from_quaternion(orientation_list)

    def return_times_reached_goal(self, start, goal):
        return self.number_of_epsiodes, self.number_reached_goal

    def clcDistance(self, start, goal):
        distance = math.sqrt(pow((goal.x),2)+pow((goal.y),2))
        return distance

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.ic.shoutdown_node();
        sys.exit(0)

class Memory():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.counter = 1
        self.returnMean = False
    def add(self, value):
        self.buffer.append(value)
        if(self.counter<self.size):
            self.counter += 1
        else:
            self.returnMean=True
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

