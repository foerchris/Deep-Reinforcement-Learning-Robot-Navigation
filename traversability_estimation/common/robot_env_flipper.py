#!/usr/bin/env python2
# -*- coding: utf-8

import numpy as np
from collections import deque# Ordered collection with ends

import time

import math
from time import  sleep
from multiprocessing import Process, Pipe
import math as m

import rospy

from nav_msgs.msg import Odometry

from robot_wrapper_flipper import image_converter
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
        self.discountFactorMue = 0.1
        self.closestDistance = 0
        self.startGoalDistance = 0
        self.lastDistance = 0
        self.startTime =0
        self.delta_vel_memory = Memory(10)

        self.lastTime = 0

        self.observation_space = []
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(28,28),dtype = np.float32))
        self.observation_space.append(gym.spaces.Box(low=-1, high=1, shape=(7,1),dtype = np.float32))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,1), dtype = np.float32)
        self.total_reward = 0
        signal.signal(signal.SIGINT, self.signal_handler)

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
        self.startSteps()

        self.ic.setAction(action)
        self.stepCounter += 1

        sleep(0.3)

        reward, d =self.clcReward()

        if(d):
            self.total_reward = 0
        else:
            self.total_reward += reward

        info = {}
        robotGroundData, currentPose = self.get_state()
        self.stopSteps()

        return  robotGroundData, currentPose, reward, d, info


    def get_state(self):
        robotGroundMap, self.currentPose, self.goalPose, self.flipperPoseFront,  self.flipperPoseRear = self.ic.returnData()

        # creat np arrays as input of the drl agent
        roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

        currentOrientation = np.asarray([roll, pitch, yaw])

        currentOrientation = np.asarray(currentOrientation, dtype=np.float32  )

        robotGroundMap = np.asarray(robotGroundMap, dtype=np.float32  )


        # norrm input data between -1 and 1
        robotGroundMap =  np.divide(robotGroundMap, 65536) #255

        currentOrientation = np.divide(currentOrientation,math.pi)
        currentOrientation = np.append(currentOrientation, np.divide(self.flipperPoseFront.current_pos,m.pi))
        currentOrientation = np.append(currentOrientation, np.divide(self.flipperPoseRear.current_pos,m.pi))
        currentOrientation = np.append(currentOrientation, np.divide(self.ic.imu_data.linear_acceleration.x,20))
        currentOrientation = np.append(currentOrientation, np.divide(self.ic.imu_data.linear_acceleration.y,20))

        #print("currentOrientation" + str(currentOrientation))
        robotGroundMap = np.asarray(robotGroundMap, dtype=np.float32  )
        currentOrientation = np.asarray(currentOrientation, dtype=np.float32  )

        # replace nan values with 0
        robotGroundMap.reshape(1,28,28)
        currentOrientation.reshape(1,7)

        return robotGroundMap, currentOrientation

    def get_available_actions_size(self):
        return self.availableActionsSize

    # Restet the enviroment and return the depth image ([x,x] Pixel) the eleviation map ([200,200] Pixel) and
    # orientation (Euler notation [roll, pitch, yaw])of the oberservation in the new enviroment
    def reset(self):
        # reset the model and replace the robot at a random location
        self.ic.stop()
        valiedEnv = False
        self.delta_vel_memory.resetBuffer()

        # resest the step counter
        self.stepCounter=0


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
                if(count_reset%1000 == 0):
                    s = smtplib.SMTP_SSL('smtp.gmail.com')
                    s.login(self.me, self.my_password)
                    s.sendmail(self.me, self.you, self.msg.as_string())
                    s.quit()
                    print("worker_" + str(self.number) + " stucked at waiting for Ready_to_Start_DRL_agent");


                sleep(0.2)

            _, self.currentPose, self.goalPose, _, _ = self.ic.returnData()
            roll, pitch, yaw = self.returnRollPitchYaw(self.currentPose.pose.pose.orientation)

            position_q = self.currentPose.pose.pose.position
            # cheakk if the roboter is in a valide starting positionv
            if roll <= math.pi/4 and roll >= -math.pi/4 and pitch <= math.pi/4 and pitch >= -math.pi/4 and position_q.z <0.7:
                valiedEnv = True
                sleep(0.2)
            count_valied_state +=1
            if (count_valied_state % 1000 == 0):
                s = smtplib.SMTP_SSL('smtp.gmail.com')
                s.login(self.me, self.my_password)
                s.sendmail(self.me, self.you, self.msg.as_string())
                s.quit()
                print("worker_" + str(self.number) + " stucked at cheak for valid state");


        self.startTime = time.time()
        self.lastTime = self.startTime

        sleep(0.2)
        self.ic.acceleration_to_high = False
        self.ic.robot_flip_over = False

        return self.get_state()


    def clcReward(self):
        _, self.currentPose, self.goalPose, _, _ = self.ic.returnData()

        currenVel = self.currentPose.twist.twist.linear.x
        self.delta_vel_memory.add(currenVel)
        var_delta_vel = self.delta_vel_memory.var();
        mean_delta_vel = self.delta_vel_memory.mean();

        currentdistance = self.clcDistance(self.currentPose.pose.pose.position,self.goalPose.pose.pose.position)
        currentTime = time.time()
        currentTime = currentTime - self.startTime
        deltaTime = currentTime - self.lastTime


        EndEpisode = False;
        reward=0

        if rospy.get_param("/GETjag"+ str(self.number) + "/Error_in_simulator"):
            EndEpisode = True
            rospy.set_param("/GETjag"+ str(self.number) + "/Error_in_simulator",False)


        if currentdistance < self.closestDistance:
            reward = self.discountFactorMue*(self.closestDistance-currentdistance)
            self.closestDistance = currentdistance

        if (self.ic.acceleration_to_high):
            print(str(self.number) + "acceleration_to_high")
            print("accel z:" + str(self.ic.accelZ))

            reward = -1
            EndEpisode = True
            self.ic.acceleration_to_high = False


        if(self.ic.robot_flip_over):
            reward = -1
            EndEpisode = True
            self.ic.robot_flip_over = False


        #if (mean_delta_vel <= 1e-2 and self.number!=1):
        if (mean_delta_vel <= 1e-2):
            reward = -1
            EndEpisode = True

        if currentdistance <= self.deltaDist:
            reward = 0.5 +  (self.startGoalDistance*10 / self.stepCounter)
            print("reached Goal")
            EndEpisode = True

        if  self.stepCounter>=self.EpisodeLength:
            EndEpisode = True

        self.lastDistance = currentdistance
        self.lastTime = currentTime

        self.episodeFinished = EndEpisode
        reward = np.float32(reward)



        return reward, EndEpisode

    def stopSteps(self):
        self.ic.startStopRobot.data = False;
        self.ic.stop()

    def startSteps(self):
        self.ic.startStopRobot.data = True;

    def is_episode_finished(self):
        return self.episodeFinished

    def endEnv(self):
        #rospy.set_param("/GETjag/End_of_enviroment",True)
        rospy.set_param("/GETjag"+ str(self.number) + "/End_of_enviroment",True)

    def returnRollPitchYaw(self, orientation):
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        return  euler_from_quaternion(orientation_list)

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

