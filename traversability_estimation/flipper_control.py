#!/usr/bin/env python2
# -*- coding: utf-8

import math
import random

import numpy as np
from torchviz import make_dot

import torch
import torch.nn as nn
from numpy import number
from torch.distributions import Normal
import tensorflow as tf

import matplotlib.pyplot as plt

import rospy
from collections import deque
from time import  sleep
import gc
import os
dirname = os.path.dirname(__file__)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()
import sys
sys.path.append(os.path.join(dirname, 'common'))
from tensorboardX import SummaryWriter


from robot_env_flipper import robotEnv

from flipper_agents import Agent
from inspect import currentframe, getframeinfo


MODELPATH = os.path.join(dirname, 'train_getjag/ppo_flipper/Model')

load_model = True
last_number_of_frames = 0
frame_idx  = 0 + last_number_of_frames

num_envs = 1

writer = SummaryWriter("train_getjag/ppo_flipper/Tensorboard")

env = robotEnv(1)

state_size_map  = env.observation_space[0].shape[0] * env.observation_space[0].shape[1]
state_size_orientation   = env.observation_space[1].shape[0]


num_outputs = env.action_space.shape[0]

stack_size = 4
class image_stacker():
    def __init__(self, state_size, stack_size):
        self.stacked_frames = deque([np.zeros((state_size_map), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)
    def return_stacked_frame(self):
            return self.stacked_frames





stacked_map_frames = deque([np.zeros((num_envs,state_size_map/state_size_map,state_size_map/state_size_map), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)
stacked_orientation_frames = deque([np.zeros((num_envs,state_size_orientation), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)


def reset_single_frame(stacked_frames, state, stack_size, number):

    for i in range(0, stack_size, 1):
        #stacked_frames.append(state)
        stacked_frames[i][number] = state

    stacked_state = np.stack(stacked_frames, axis=1)

    return stacked_state, stacked_frames

def stack_frames(stacked_frames, state, stack_size, is_new_episode):

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((state.shape), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)


        # Because we're in a new episode, copy the same frame 4x
        for i in range(0, stack_size, 1):

            stacked_frames.append(state)


        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)


    else:

        stacked_frames.append(state)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)

    return stacked_state, stacked_frames


#Hyper params:
hidden_size      = 576
lr               = 1e-3
lr_decay_epoch   = 100.0
init_lr          = lr
epoch            = 0.0

max_num_steps    = 350
num_steps        = 2000
mini_batch_size  = 200
ppo_epochs       = 8
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
PPO_EPSILON      = 0.2
CRICIC_DISCOUNT  = 0.5
ENTROPY_BETA     = 0.01
eta              = 0.01
threshold_reward = 5
threshold_reached_goal = 2
#

agent = Agent(state_size_map, state_size_orientation, num_outputs, hidden_size, stack_size, load_model, MODELPATH, lr, mini_batch_size, num_envs, lr_decay_epoch, init_lr, writer, eta)

env.set_episode_length(100000)

map_state, orientation_state = env.get_state()

print("map_state shape:" + str(map_state.shape))
print("orientation_state shape:" + str(orientation_state.shape))

map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
orientation_state, stacked_orientation_frames = stack_frames(stacked_orientation_frames, orientation_state, stack_size, True)

print("map_state shape:" + str(map_state.shape))
print("orientation_state shape:" + str(orientation_state.shape))

agent.feature_net.hidden = agent.feature_net.init_hidden(num_envs)
(hidden_state_h, hidden_state_c) = agent.feature_net.hidden

done_cache = []
step_count = []
total_reward = []
total_total_reward = []
total_step_count = []
total_std = []
number_of_episodes = 0
number_reached_goal = 0
mean_flipp_over = 0
mean_stucked = 0
reach_goal = []
flipp_over= []
stucked = []

entropy = 0
total_angluar_acc = []

for i in range(0, num_envs):
    done_cache.append(False)
    step_count.append(0)
    total_reward.append(0)

agent.feature_net.eval()
agent.ac_model.eval()


while True :
    with torch.no_grad():
        map_state = torch.FloatTensor(map_state).to(device)

        orientation_state = torch.FloatTensor(orientation_state).to(device)

        map_state = map_state.view(1,-1,map_state.shape[1],map_state.shape[2])
        orientation_state = orientation_state.view(1,-1,orientation_state.shape[1])

        print("map_state shape:" + str(map_state.shape))
        print("orientation_state shape:" + str(orientation_state.shape))

        features, next_hidden_state_h, next_hidden_state_c = agent.feature_net(map_state, orientation_state, hidden_state_h, hidden_state_c)

        dist, value, std  = agent.ac_model( features)

        action = dist.mean.detach()

        # this is a x,1 tensor is kontains alle the possible actions
        #the cpu command move it from a gpu tensor to a cpu tensor
        action = action.view(action.shape[1])

        next_map_state, next_orientation_state, reward, done, angl_acc, _ = env.step(action.cpu().numpy())

        next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
        next_orientation_state, stacked_orientation_frames = stack_frames(stacked_orientation_frames,next_orientation_state,stack_size, False)



        map_state = next_map_state
        orientation_state = next_orientation_state

        next_map_state = torch.FloatTensor(next_map_state).to(device)
        next_orientation_state = torch.FloatTensor(next_orientation_state).to(device)

        hidden_state_h = next_hidden_state_h
        hidden_state_c = next_hidden_state_c