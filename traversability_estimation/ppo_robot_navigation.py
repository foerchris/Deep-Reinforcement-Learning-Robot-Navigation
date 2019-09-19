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
from __builtin__ import True
dirname = os.path.dirname(__file__)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()
import sys
sys.path.append(os.path.join(dirname, 'common'))
from tensorboardX import SummaryWriter


from multiprocessing_env import SubprocVecEnv
from robot_env import robotEnv

from agents import Agent


MODELPATH = os.path.join(dirname, 'train_getjag/ppo/Model')

load_model = False
last_number_of_frames = 0

frame_idx  = 0 + last_number_of_frames
num_envs_possible = 16;
num_envs = 0;

summary_writer = tf.summary.FileWriter("train_getjag/ppo/Tensorboard")

#test_writer = tf.summary.FileWriter("train_getjag/train_" + str(0), sess.graph)

writer = SummaryWriter("train_getjag/ppo/Tensorboard")
for i in range(num_envs_possible):
    if (rospy.has_param("/GETjag" + str(i) + "/worker_ready")):
        if (rospy.get_param("/GETjag" + str(i) + "/worker_ready")):
            num_envs += 1
            print("worker_", num_envs)

def make_env(i):
    def _thunk():
        #env = gym.make(env_name)
        env =  robotEnv(i)
        return env

    return _thunk

envs = [make_env(i+1) for i in range(num_envs)]

envs = SubprocVecEnv(envs)

#env = robotEnv(1)


state_size_map  = envs.observation_space[0].shape[0] * envs.observation_space[1].shape[1]
state_size_depth  = envs.observation_space[1].shape[0] * envs.observation_space[1].shape[1]
state_size_goal   = envs.observation_space[2].shape[0]


num_outputs = envs.action_space.shape[0]

stack_size = 1

class image_stacker():
    def __init__(self, state_size, stack_size):
        self.stacked_frames = deque([np.zeros((state_size_map), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)
    def return_stacked_frame(self):
            return self.stacked_frames





stacked_map_frames = deque([np.zeros((num_envs,state_size_map/state_size_map,state_size_map/state_size_map), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)
stacked_depth_frames = deque([np.zeros((num_envs,state_size_depth/state_size_depth,state_size_depth/state_size_depth), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)
stacked_goal_frames = deque([np.zeros((num_envs,state_size_goal), dtype=np.float32) for i in range(stack_size)], maxlen=stack_size)


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
        stacked_state = np.stack(stacked_frames, axis=1)


    else:

        stacked_frames.append(state)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=1)

    return stacked_state, stacked_frames


def plot(frame_idx, rewards):
    #plt.figure(figsize=(20, 5))
   # plt.subplot(131)
    #print("frame_idx" + str(frame_idx))
    #print("rewards" + str(rewards))

    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show(block=False)


#Hyper params:
hidden_size      = 585 #576*2
lstm_layers      = 1
lr               = 1e-3
lr_decay_epoch   = 100.0
init_lr          = lr
epoch            = 0.0

max_num_steps    = 300
num_steps        = 2000
mini_batch_size  = 200
ppo_epochs       = 6
max_grad_norm    = 0.5
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
PPO_EPSILON      = 0.25
CRICIC_DISCOUNT  = 0.5
ENTROPY_BETA     = 0.01
eta              = 0.01
threshold_reward = 5
threshold_reached_goal = 2
#


f= open("train_getjag/ppo/Tensorboard/Hyperparameters.txt","w+")

f.write("Navigation Control")
f.write("\n Workers: " + str(num_envs))
f.write("\n hidden_size: " + str(hidden_size))
f.write("\n lr: " + str(lr))
f.write("\n lr_decay_epoch: " + str(lr_decay_epoch))
f.write("\n init_lr: " + str(init_lr))
f.write("\n epoch: " + str(epoch))
f.write("\n max_num_steps: " + str(max_num_steps))
f.write("\n num_steps: " + str(num_steps))
f.write("\n mini_batch_size: " + str(mini_batch_size))
f.write("\n ppo_epochs: " + str(ppo_epochs))
f.write("\n GAMMA: " + str(GAMMA))
f.write("\n GAE_LAMBDA: " + str(GAE_LAMBDA))
f.write("\n PPO_EPSILON: " + str(PPO_EPSILON))
f.write("\n CRICIC_DISCOUNT: " + str(CRICIC_DISCOUNT))
f.write("\n ENTROPY_BETA: " + str(ENTROPY_BETA))
f.write("\n eta: " + str(eta))
f.write("\n LSTM: Yes")
f.write("\n Architecture: 1")

f.close()

agent = Agent(state_size_map, state_size_depth , state_size_goal, num_outputs, hidden_size, stack_size, lstm_layers,load_model, MODELPATH, lr, mini_batch_size, num_envs, lr_decay_epoch, init_lr, eta)
max_frames = 500000
test_rewards = []

#decay_ppo_epsilon = tf.train.polynomial_decay(PPO_EPSILON, frame_idx, max_num_steps, 1e-2, power=1.0)
#learning_rate = tf.train.polynomial_decay(lr, frame_idx, max_num_steps, 1e-5, power=1.0)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate )

episode_length = []
for i in range(0, num_envs):
    episode_length.append(max_num_steps)

envs.set_episode_length(episode_length)


early_stop = False

best_reward = 0

map_state,depth_state, goal_state = envs.reset()

map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)

agent.feature_net.hidden = agent.feature_net.init_hidden(num_envs)
(hidden_state_h, hidden_state_c) = agent.feature_net.hidden

done_cache = []
step_count = []
all_rewards = []
total_reward = []
total_total_reward = []
total_step_count = []
total_std = []
number_of_episodes = 0
number_reached_goal = 0
reach_goal = []
entropy = 0

for i in range(0, num_envs):
    done_cache.append(False)
    step_count.append(0)
    total_reward.append(0)
    bla = []
    all_rewards.append(bla)

while frame_idx < max_frames and not early_stop:

    log_probs = []
    values = []
    map_states = []
    depth_states = []
    goal_states = []
    hidden_states_h = []
    hidden_states_c = []
    actions = []
    rewards = []
    masks = []


    agent.feature_net.eval()
    agent.ac_model.eval()

    with torch.no_grad():
        for _ in range(num_steps):


            map_state = torch.FloatTensor(map_state).to(device)
            depth_state = torch.FloatTensor(depth_state).to(device)
            goal_state = torch.FloatTensor(goal_state).to(device)



            features, next_hidden_state_h, next_hidden_state_c = agent.feature_net(map_state, depth_state, goal_state, hidden_state_h, hidden_state_c)

            dist, value, std  = agent.ac_model( features)

            total_std.append(std.cpu().numpy())

            #print(dist.stddev())
            #

            action = dist.sample()
           # print("frame_idx: " + str(frame_idx))
           # print("dist.mean.detach()"+ str(dist.mean.detach().cpu()))
           # print("std.cpu(): " + str(std.cpu()))
            # this is a x,1 tensor is kontains alle the possible actions
            # the cpu command move it from a gpu tensor to a cpu tensor
            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step(action.cpu().numpy())


            for i in range(0, num_envs):
                if (done[i] == True):
                   # print("number_of_episodes" + str(number_of_episodes))
                   # print("number_reached_goal" + str(number_reached_goal))

                    number_of_episodes += 1
                    if (reward[i] >= 0.2):
                        number_reached_goal += 1
                        #print(str(number)+"reached goal")
                        reach_goal.append(1)
                    else:
                        reach_goal.append(0)

                    _, stacked_map_frames = reset_single_frame(stacked_map_frames, next_map_state[i], stack_size, i)
                    _, stacked_depth_frames = reset_single_frame(stacked_depth_frames, next_depth_state[i], stack_size,
                                                                 i)
                    _, stacked_goal_frames = reset_single_frame(stacked_goal_frames, next_goal_state[i], stack_size, i)

                    (single_hidden_state_h, single_hidden_state_c) = agent.feature_net.init_hidden(1)

                    for layer in range(0,lstm_layers):
                        next_hidden_state_h[layer][i] = single_hidden_state_h[layer][0].detach()
                        next_hidden_state_c[layer][i] = next_hidden_state_c[layer][0].detach()



                    #next_hidden_state_c[0][i] = single_hidden_state_c.detach()
                   # print("number_of_episodes" + str(number_of_episodes))
                   # print("number_reached_goal" + str(number_reached_goal))

            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames,next_depth_state,stack_size, False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames,next_goal_state,stack_size, False)


            next_features, _, _ = agent.feature_net(torch.FloatTensor(next_map_state).to(device), torch.FloatTensor(next_depth_state).to(device) , torch.FloatTensor(next_goal_state).to(device), next_hidden_state_h, next_hidden_state_h)

            # total reward = int reward
            #intrinsic_reward = agent.compute_intrinsic_reward(features, next_features, action)

            #reward +=  intrinsic_reward
            total_reward += reward

            for i in range(0, num_envs):
                #all_rewards[i].append(reward[i])

                step_count[i] += 1
                if (done[i] == True):
                    total_step_count.append(step_count[i])
                    step_count[i] = 0
                    total_total_reward.append(total_reward[i])
                    #print("all_rewards" + str(i + 1) + ": " + str(all_rewards[i]))
                    #print("mean_all_rewards" + str(i + 1) + ": " + str(np.sum(all_rewards[i])))
                    #print("total_step_count" + str(i + 1) + ": " + str(total_step_count))
                    #print("total_total_reward" + str(i + 1) + ": " + str(total_total_reward))
                    total_reward[i] = 0
                    #all_rewards[i] = []


            #sample_i_rall += intrinsic_reward[sample_env_idx]
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
            rewards.append(reward)
            done = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            masks.append(done)

            hidden_states_h.append(hidden_state_h)
            hidden_states_c.append(hidden_state_c)

            map_states.append(map_state)
            depth_states.append(depth_state)
            goal_states.append(goal_state)

            actions.append(action)

            map_state = next_map_state
            depth_state = next_depth_state
            goal_state = next_goal_state

            #torch.cuda.empty_cache()
            frame_idx += 1

            if frame_idx % 2000 == 0:

                mean_test_rewards = np.mean(total_total_reward)
                total_total_reward = []
                mean_test_lenghts = np.mean(total_step_count)
                total_step_count = []
                mean_total_std = np.mean(total_std)
                total_std = []
                mean_reach_goal = np.mean(reach_goal)
                reach_goal = []


                test_rewards.append(mean_test_rewards)
                print("save tensorboard")
                # plot(frame_idx, test_rewards)
                summary = tf.Summary()
                summary.value.add(tag='Mittelwert/Belohnungen', simple_value=float(mean_test_rewards))
                summary.value.add(tag='Mittelwert/Epsioden Länge', simple_value=float(mean_test_lenghts))
                summary.value.add(tag='Mittelwert/Std-Abweichung', simple_value=float(mean_total_std))
                summary.value.add(tag='Mittelwert/Ziel erreich', simple_value=float(mean_reach_goal))
                summary.value.add(tag='Mittelwert/anzahl Episoden', simple_value=float(number_of_episodes))
                number_of_episodes = 0
                summary.value.add(tag='Mittelwert/anzahl Ziel erreicht', simple_value=float(number_reached_goal))
                number_reached_goal = 0

                summary_writer.add_summary(summary, frame_idx)

                for name, param in agent.feature_net.named_parameters():
                    if param.requires_grad:
                        tensor = param.data
                        tensor = tensor.cpu().numpy()
                        writer.add_histogram(name, tensor, bins='doane')

                for name, param in agent.ac_model.named_parameters():
                    if param.requires_grad:
                        tensor = param.data
                        tensor = tensor.cpu().numpy()
                        writer.add_histogram(name, tensor, bins='doane')

                print("save weights")

                if best_reward is None or best_reward < mean_test_rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, mean_test_rewards))
                        name = "%s_best_%+.3f_%d.dat" % ('ppo_robot_nav', mean_test_rewards, frame_idx)
                        torch.save(agent.feature_net.state_dict(), MODELPATH + '/save_ppo_feature_net_best_reward.dat')
                        torch.save(agent.ac_model.state_dict(), MODELPATH + '/save_ppo_ac_model_best_reward.dat')
                    best_reward = mean_test_rewards

                if mean_test_rewards > threshold_reward: early_stop = True

                if mean_reach_goal > threshold_reached_goal: early_stop = True
                torch.save(agent.feature_net.state_dict(), MODELPATH + '/save_ppo_feature_net.dat')
                torch.save(agent.ac_model.state_dict(), MODELPATH + '/save_ppo_ac_model.dat')


            next_map_state = torch.FloatTensor(next_map_state).to(device)
            next_depth_state = torch.FloatTensor(next_depth_state).to(device)
            next_goal_state = torch.FloatTensor(next_goal_state).to(device)

            hidden_state_h = next_hidden_state_h
            hidden_state_c = next_hidden_state_c

    agent.feature_net.train()
    agent.ac_model.train()

    next_features, _, _= agent.feature_net(next_map_state, next_depth_state, next_goal_state, hidden_state_h, hidden_state_c)

    _, next_value, _ = agent.ac_model(next_features)
    returns = agent.compute_gae(next_value, rewards, masks, values, GAMMA, GAE_LAMBDA)

    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    map_states = torch.cat(map_states)
    depth_states = torch.cat(depth_states)
    goal_states = torch.cat(goal_states)
    hidden_states_h = torch.cat(hidden_states_h)
    hidden_states_c = torch.cat(hidden_states_c)

    hidden_states_h = hidden_states_h.view(-1, lstm_layers,hidden_states_h.shape[2])
    hidden_states_c = hidden_states_c.view(-1, lstm_layers, hidden_states_c.shape[2])


    actions = torch.cat(actions)
    advantages = returns - values

    epoch += 1.0
    agent.ppo_update(frame_idx, ppo_epochs,  map_states, depth_states, goal_states, hidden_states_h, hidden_states_c , actions, log_probs, returns, advantages, values, epoch, PPO_EPSILON, CRICIC_DISCOUNT, ENTROPY_BETA, max_grad_norm)



map_state,depth_state, goal_state = envs.reset()

map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)

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
reach_goal = []
entropy = 0

for i in range(0, num_envs):
    done_cache.append(False)
    step_count.append(0)
    total_reward.append(0)
    bla = []
    all_rewards.append(bla)

agent.feature_net.eval()
agent.ac_model.eval()
frame_idx = 0
eval_steps = 6000
steps_idx = 0

while frame_idx < eval_steps :
    with torch.no_grad():
        for _ in range(num_steps):
            print("frame_idx: " + str(frame_idx))
            frame_idx += 1

            map_state = torch.FloatTensor(map_state).to(device)
            depth_state = torch.FloatTensor(depth_state).to(device)
            goal_state = torch.FloatTensor(goal_state).to(device)



            features, next_hidden_state_h, next_hidden_state_c = agent.feature_net(map_state, depth_state, goal_state, hidden_state_h, hidden_state_c)

            dist, value, std  = agent.ac_model( features)

            total_std.append(std.cpu().numpy())


            action = dist.mean.detach()
            #print("action" + str(action))
            #action = dist.mean.detach()
            #print("action" + str(action))

            # this is a x,1 tensor is kontains alle the possible actions
            # the cpu command move it from a gpu tensor to a cpu tensor
            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step(action.cpu().numpy())


            for i in range(0, num_envs):
                if (done[i] == True):

                    number_of_episodes += 1
                    if (reward[i] >= 0.2):
                        number_reached_goal += 1
                        reach_goal.append(1)
                    else:
                        reach_goal.append(0)

                    _, stacked_map_frames = reset_single_frame(stacked_map_frames, next_map_state[i], stack_size, i)
                    _, stacked_depth_frames = reset_single_frame(stacked_depth_frames, next_depth_state[i], stack_size,
                                                                 i)
                    _, stacked_goal_frames = reset_single_frame(stacked_goal_frames, next_goal_state[i], stack_size, i)

                    (single_hidden_state_h, single_hidden_state_c) = agent.feature_net.init_hidden(1)

                    for layer in range(0,lstm_layers):
                        next_hidden_state_h[layer][i] = single_hidden_state_h[layer][0].detach()
                        next_hidden_state_c[layer][i] = next_hidden_state_c[layer][0].detach()


            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames,next_depth_state,stack_size, False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames,next_goal_state,stack_size, False)


            next_features, _, _ = agent.feature_net(torch.FloatTensor(next_map_state).to(device), torch.FloatTensor(next_depth_state).to(device) , torch.FloatTensor(next_goal_state).to(device), next_hidden_state_h, next_hidden_state_h)

            total_reward += reward

            for i in range(0, num_envs):
                step_count[i] += 1
                if (done[i] == True):
                    total_step_count.append(step_count[i])
                    step_count[i] = 0
                    total_total_reward.append(total_reward[i])
                    total_reward[i] = 0

            map_state = next_map_state
            depth_state = next_depth_state
            goal_state = next_goal_state

            #torch.cuda.empty_cache()
            steps_idx += 1


            next_map_state = torch.FloatTensor(next_map_state).to(device)
            next_depth_state = torch.FloatTensor(next_depth_state).to(device)
            next_goal_state = torch.FloatTensor(next_goal_state).to(device)

            hidden_state_h = next_hidden_state_h
            hidden_state_c = next_hidden_state_c

mean_test_rewards = np.mean(total_total_reward)
total_total_reward = []
mean_test_lenghts = np.mean(total_step_count)
total_step_count = []
mean_total_std = np.mean(total_std)
total_std = []
mean_reach_goal = np.mean(reach_goal)
reach_goal = []

test_rewards.append(mean_test_rewards)
print("save tensorboard")
# plot(frame_idx, test_rewards)
summary = tf.Summary()
summary.value.add(tag='Mittelwert/Belohnungen', simple_value=float(mean_test_rewards))
summary.value.add(tag='Mittelwert/Epsioden Länge', simple_value=float(mean_test_lenghts))
summary.value.add(tag='Mittelwert/Std-Abweichung', simple_value=float(mean_total_std))
summary.value.add(tag='Mittelwert/Ziel erreich', simple_value=float(mean_reach_goal))
summary.value.add(tag='Mittelwert/anzahl Episoden', simple_value=float(number_of_episodes))
number_of_episodes = 0
summary.value.add(tag='Mittelwert/anzahl Ziel erreicht', simple_value=float(number_reached_goal))
number_reached_goal = 0

summary_writer.add_summary(summary, frame_idx + steps_idx)