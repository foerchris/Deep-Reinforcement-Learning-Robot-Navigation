#!/usr/bin/env python2
# -*- coding: utf-8

import math
import random

import numpy as np
from torchviz import make_dot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

MODELPATH = os.path.join(dirname, 'train_getjag/ppo/Model')

load_model = True
last_number_of_frames = 34000
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


state_size_map  = envs.observation_space[0].shape[0] * envs.observation_space[0].shape[1]
state_size_depth  = envs.observation_space[1].shape[0] * envs.observation_space[1].shape[1]
state_size_goal   = envs.observation_space[2].shape[0]


num_outputs = envs.action_space.shape[0]

stack_size = 4

class image_stacker():
    def __init__(self, state_size, stack_size):
        self.stacked_frames = deque([np.zeros((state_size_map), dtype=np.float16) for i in range(stack_size)], maxlen=stack_size)
    def return_stacked_frame(self):
            return self.stacked_frames





stacked_map_frames = deque([np.zeros((num_envs,state_size_map/state_size_map,state_size_map/state_size_map), dtype=np.float16) for i in range(stack_size)], maxlen=stack_size)
stacked_depth_frames = deque([np.zeros((num_envs,state_size_depth/state_size_depth,state_size_depth/state_size_depth), dtype=np.float16) for i in range(stack_size)], maxlen=stack_size)
stacked_goal_frames = deque([np.zeros((num_envs,state_size_goal), dtype=np.float16) for i in range(stack_size)], maxlen=stack_size)


def reset_single_frame(stacked_frames, state, stack_size, number):

    for i in range(0, stack_size, 1):
        #stacked_frames.append(state)
        stacked_frames[i][number] = state

    stacked_state = np.stack(stacked_frames, axis=1)

    return stacked_state, stacked_frames


def stack_frames(stacked_frames, state, stack_size, is_new_episode):

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((state.shape), dtype=np.float16) for i in range(stack_size)], maxlen=stack_size)


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

def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.normal_(m.weight, mean=0., std=0.05)
        torch.nn.init.orthogonal_(m.weight.data)
        #nn.init.constant_(m.bias, 0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


class ActorCritic(nn.Module):
    def __init__(self,state_size_map, state_size_depth , state_size_goal, num_outputs, hidden_size, stack_size, std=0.0):
        self.state_size_map = state_size_map
        self.state_size_depth = state_size_depth
        self.state_size_goal = state_size_goal
        self.stack_size = stack_size
        self.hidden_dim = hidden_size
        super(ActorCritic, self).__init__()


        # This is the ElevMap part
        self.cnn_map = nn.Sequential(
            nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # This is the depth Image part
        self.cnn_depth = nn.Sequential(
            nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # This is the goal pose part
        self.cnn_goal = nn.Sequential(
            nn.Linear(6*stack_size, 5184),
            nn.ReLU()
        )

        self.cnn_map_goal = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1)

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, 1)

        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()

        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        #self.apply(init_weights)

        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim
                            , device=device))

    def forward(self, map_state, depth_state ,goal_state, hidden_h, hidden_c):

        self.hidden = (hidden_h, hidden_c)

        map_state = map_state.view(-1, map_state.shape[1], map_state.shape[2], map_state.shape[3])

        map = self.cnn_map(map_state)


        goal_state = goal_state.view(-1, goal_state.shape[1]* goal_state.shape[2])

        goal = self.cnn_goal(goal_state)

        goal = goal.view(-1, map.shape[1], map.shape[2], map.shape[3])


        map_and_goal = map.add(goal)

        map_goal_out = self.cnn_map_goal(map_and_goal)

        map_goal_out = map_goal_out.view(-1, map_goal_out.shape[1] * map_goal_out.shape[2] * map_goal_out.shape[3])

        depth_state = depth_state.view(-1, depth_state.shape[1], depth_state.shape[2], depth_state.shape[2])

        depth_out = self.cnn_depth(depth_state)
        depth_out = depth_out.view(-1, depth_out.shape[1] * depth_out.shape[2] * depth_out.shape[3])

        map_goal_depth = torch.cat((map_goal_out, depth_out), 1)

        map_goal_depth = map_goal_depth.view(1, -1, map_goal_depth.shape[1])

        lstm_out, self.hidden = self.lstm(map_goal_depth, self.hidden)

        lstm_out = lstm_out.view(-1, lstm_out.shape[2])

        value = self.critic(lstm_out)
        #print('value' + str(value))

        mu = self.actor(lstm_out)

        std = self.log_std.exp().expand_as(mu)
        #print('mu' + str(mu))
        #print('std' + str(std))
        dist = Normal(mu, std)

        hidden_h = self.hidden[0]
        hidden_c = self.hidden[1]

        return dist, value, hidden_h, hidden_c


def plot(frame_idx, rewards):
    #plt.figure(figsize=(20, 5))
   # plt.subplot(131)
    #print("frame_idx" + str(frame_idx))
    #print("rewards" + str(rewards))

    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show(block=False)


def test_env(vis=False):
    global stacked_map_frames
    global stacked_depth_frames
    global stacked_goal_frames

    map_state, depth_state,goal_state  = envs.reset_test(1)

    map_state = map_state[0]
    depth_state = depth_state[0]
    goal_state = goal_state[0]

    map_state = np.reshape(map_state, (-1, map_state.shape[0], map_state.shape[1]))
    depth_state = np.reshape(depth_state, (-1, depth_state.shape[0], depth_state.shape[1]))
    goal_state = np.reshape(goal_state, (-1, goal_state.shape[0]))

    map_state = np.concatenate([map_state for x in range(num_envs)])
    depth_state = np.concatenate([depth_state for x in range(num_envs)])
    goal_state = np.concatenate([goal_state for x in range(num_envs)])

    map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
    depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
    goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)


    model.hidden = model.init_hidden(1)
    (hidden_state_h, hidden_state_c) = model.hidden

    #if vis: env.render()
    done = False
    total_reward = 0
    model.eval()
    step_count = 0
    with torch.no_grad():
        while not done:


            map_state = torch.FloatTensor(map_state[0]).unsqueeze(0).to(device)
            depth_state = torch.FloatTensor(depth_state[0]).unsqueeze(0).to(device)
            goal_state = torch.FloatTensor(goal_state[0]).unsqueeze(0).to(device)


            dist, _, hidden_state_h, hidden_state_c  = model(map_state, depth_state, goal_state, hidden_state_h, hidden_state_c)

            action = dist.sample().cpu().numpy()

            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step_test(action,1)


            next_map_state = next_map_state[0]
            next_depth_state = next_depth_state[0]
            next_goal_state = next_goal_state[0]

            next_map_state = np.reshape(next_map_state, (-1, next_map_state.shape[0], next_map_state.shape[1]))
            next_depth_state = np.reshape(next_depth_state, (-1, next_depth_state.shape[0], next_depth_state.shape[1]))
            next_goal_state = np.reshape(next_goal_state, (-1, next_goal_state.shape[0]))

            next_map_state = np.concatenate([next_map_state for x in range(num_envs)])
            next_depth_state = np.concatenate([next_depth_state for x in range(num_envs)])
            next_goal_state = np.concatenate([next_goal_state for x in range(num_envs)])

            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames,next_depth_state,stack_size, False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames,next_goal_state,stack_size, False)

            map_state = next_map_state
            depth_state = next_depth_state
            goal_state = next_goal_state
            #if vis: env.render()
            step_count += 1
            total_reward += reward
    return total_reward, step_count


def multi_test_env():
    global stacked_map_frames
    global stacked_depth_frames
    global stacked_goal_frames

    map_state, depth_state,goal_state  = envs.reset()

    map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
    depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
    goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)


    model.hidden = model.init_hidden(num_envs)
    (hidden_state_h, hidden_state_c) = model.hidden

    #if vis: env.render()
    model.eval()
    done_cache = []
    step_count = []
    total_reward = []
    log_probs = []
    values = []
    entropy = 0
    for i in range(0, num_envs):
        done_cache.append(False)
        step_count.append(0)
        total_reward.append(0)

    with torch.no_grad():
        while not all(is_done == True for is_done in done_cache):

            map_state = torch.FloatTensor(map_state).to(device)
            depth_state = torch.FloatTensor(depth_state).to(device)
            goal_state = torch.FloatTensor(goal_state).to(device)

            dist, value, hidden_state_h, hidden_state_c  = model(map_state, depth_state, goal_state, hidden_state_h, hidden_state_c)



            #action = dist.sample()
            action = dist.mean.detach()

            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step(action.cpu().numpy())

            for i in range(0, num_envs):
                if(done[i]==True):
                    done_cache[i] = True



            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames,next_depth_state,stack_size, False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames,next_goal_state,stack_size, False)

            map_state = next_map_state
            depth_state = next_depth_state
            goal_state = next_goal_state

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean().cpu().numpy()

            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())

            for i in range(0, num_envs):
                if(done_cache[i]==False):
                    step_count[i] += +1
                    total_reward[i] += reward[i]

    return total_reward, step_count, log_probs, values, entropy

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, map_states, depth_states, goal_states, hidden_states_h, hidden_states_c, actions, log_probs, returns, advantage):
    batch_size = map_states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield map_states[rand_ids, :], depth_states[rand_ids, :], goal_states[rand_ids, :], hidden_states_h[rand_ids, :], hidden_states_c[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, map_states, depth_states, goal_states, hidden_states_h, hidden_states_c, actions, log_probs, returns, advantages, clip_param=0.2, discount=0.5, beta=0.001):
    count_steps = 0
    sum_returns = 0.0
    sum_advantage = 0.0
    sum_loss_actor = 0.0
    sum_loss_critic = 0.0
    sum_entropy = 0.0
    sum_loss_total = 0.0

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    for _ in range(ppo_epochs):
        for  map_state, depth_state, goal_state, hidden_state_h, hidden_state_c, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size,  map_states, depth_states, goal_states, hidden_states_h, hidden_states_c, actions, log_probs,
                                                                        returns, advantages):

            hidden_state_h = hidden_state_h.view(1, -1, hidden_state_h.shape[2])
            hidden_state_c = hidden_state_c.view(1, -1, hidden_state_c.shape[2])


            dist, value, hidden_state_h, hidden_state_c = model( map_state, depth_state, goal_state,  hidden_state_h, hidden_state_c)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            #print('actor_loss' + str(actor_loss))
            critic_loss = (return_ - value).pow(2).mean()
            #print('critic_loss' + str(critic_loss))

            loss = discount * critic_loss + actor_loss - beta * entropy
            #print('loss' + str(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track statistics
            sum_returns += return_.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
    summary = tf.Summary()
    summary.value.add(tag='Perf/sum_returns', simple_value=float(sum_returns))
    summary.value.add(tag='Perf/sum_advantage', simple_value=float(sum_advantage))
    summary.value.add(tag='Perf/sum_loss_actor', simple_value=float(sum_loss_actor))
    summary.value.add(tag='Perf/sum_loss_critic', simple_value=float(sum_loss_critic))
    summary.value.add(tag='Perf/sum_loss_total', simple_value=float(sum_loss_total))
    summary.value.add(tag='Perf/sum_entropy', simple_value=float(sum_entropy))
    summary_writer.add_summary(summary, frame_idx)

#Hyper params:
hidden_size      = 1152
lr               = 1e-5
lr_decay_epoch   = 120.0
init_lr          = lr
epoch            = 0.0

max_num_steps    = 200
num_steps        = 400
mini_batch_size  = 10
ppo_epochs       = 4
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
PPO_EPSILON      = 0.2
CRICIC_DISCOUNT  = 0.5
ENTROPY_BETA     = 0.01
threshold_reward = 5
#
model = ActorCritic(state_size_map*stack_size, state_size_depth * stack_size, state_size_goal * stack_size , num_outputs, hidden_size, stack_size).to(device)

if(load_model):
    model.load_state_dict(torch.load(MODELPATH + '/save_ppo_model'))
else:
    model.apply(init_weights)




max_frames = 1000000
test_rewards = []

#decay_ppo_epsilon = tf.train.polynomial_decay(PPO_EPSILON, frame_idx, max_num_steps, 1e-2, power=1.0)
learning_rate = tf.train.polynomial_decay(lr, frame_idx, max_num_steps, 1e-5, power=1.0)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate )
optimizer = optim.Adam(model.parameters(), lr=lr )

episode_length = []
for i in range(0, num_envs):
    episode_length.append(max_num_steps)

envs.set_episode_length(episode_length)





early_stop = False

best_reward = 0

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
    entropy = 0

    map_state,depth_state, goal_state = envs.reset()

    map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
    depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
    goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)

    model.hidden = model.init_hidden(num_envs)
    (hidden_state_h, hidden_state_c) = model.hidden

    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):


            map_state = torch.FloatTensor(map_state).to(device)
            depth_state = torch.FloatTensor(depth_state).to(device)
            goal_state = torch.FloatTensor(goal_state).to(device)

            hidden_states_h.append(hidden_state_h)
            hidden_states_c.append(hidden_state_c)

            dist, value, hidden_state_h, hidden_state_c = model(map_state, depth_state, goal_state, hidden_state_h, hidden_state_c)


            action = dist.sample()

            # this is a x,1 tensor is kontains alle the possible actions
            # the cpu command move it from a gpu tensor to a cpu tensor
            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step( action.cpu().numpy())

            for i in range(0, num_envs):
                if (done[i] == True):
                    _, stacked_map_frames = reset_single_frame(stacked_map_frames, next_map_state[i], stack_size, i)
                    _, stacked_depth_frames = reset_single_frame(stacked_depth_frames, next_depth_state[i], stack_size,
                                                                 i)
                    _, stacked_goal_frames = reset_single_frame(stacked_goal_frames, next_goal_state[i], stack_size, i)

                    (single_hidden_state_h, single_hidden_state_c) = model.init_hidden(1)
                    hidden_state_h[0][i] = single_hidden_state_h
                    hidden_state_c[0][i] = single_hidden_state_c

            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames,next_map_state,stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames,next_depth_state,stack_size, False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames,next_goal_state,stack_size, False)


            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
            rewards.append(reward)
            done = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            masks.append(done)

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
                mean_test_rewards = []
                mean_test_lenghts = []
                mean_test_log_probs = []
                mean_test_values = []
                mean_test_entropy = []
                print("test env")

                for _ in range(1):
                    test_reward, test_lenght, test_log_probs, test_values, test_entropy = multi_test_env()

                    mean_test_rewards.append( np.mean(test_reward))
                    mean_test_lenghts.append( np.mean(test_lenght))
                    mean_test_log_probs.append(np.mean(test_log_probs))
                    mean_test_values.append(np.mean(test_values))
                    mean_test_entropy.append(test_entropy)

                mean_test_rewards = np.mean(mean_test_rewards)
                mean_test_lenghts = np.mean(mean_test_lenghts)
                mean_test_log_probs = np.mean(mean_test_log_probs)
                mean_test_values = np.mean(mean_test_values)
                mean_test_entropy = np.mean(mean_test_entropy)

                test_rewards.append(mean_test_rewards)
                print("update tensorboard")
                #plot(frame_idx, test_rewards)
                summary = tf.Summary()
                summary.value.add(tag='Perf/mean_test_rewards', simple_value=float(mean_test_rewards))
                summary.value.add(tag='Perf/mean_test_lenghts', simple_value=float(mean_test_lenghts))
                summary.value.add(tag='Perf/mean_test_log_probs', simple_value=float(mean_test_log_probs))
                summary.value.add(tag='Perf/mean_test_values', simple_value=float(mean_test_values))
                summary.value.add(tag='Perf/mean_test_entropy', simple_value=float(mean_test_entropy))
                summary_writer.add_summary(summary, frame_idx)

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        tensor = param.data
                        tensor = tensor.cpu().numpy()
                        writer.add_histogram(name, tensor, bins='doane')

                print("updated tensorboard")

                if best_reward is None or best_reward < mean_test_rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, mean_test_rewards))
                        name = "%s_best_%+.3f_%d.dat" % ('ppo_robot_nav', mean_test_rewards, frame_idx)
                        #fname = os.path.join('.', 'checkpoints', name)
                        torch.save(model.state_dict(), MODELPATH + name)
                    best_reward = mean_test_rewards

                if mean_test_rewards > threshold_reward: early_stop = True
                torch.save(model.state_dict(), MODELPATH + '/save_ppo_model')

            next_map_state = torch.FloatTensor(next_map_state).to(device)
            next_depth_state = torch.FloatTensor(next_depth_state).to(device)
            next_goal_state = torch.FloatTensor(next_goal_state).to(device)

    model.train()

    _, next_value, hidden_state_h, hidden_state_c = model(next_map_state, next_depth_state, next_goal_state, hidden_state_h, hidden_state_c)
    returns = compute_gae(next_value, rewards, masks, values, GAMMA, GAE_LAMBDA)

    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    map_states = torch.cat(map_states)
    depth_states = torch.cat(depth_states)
    goal_states = torch.cat(goal_states)
    hidden_states_h = torch.cat(hidden_states_h)
    hidden_states_c = torch.cat(hidden_states_c)

    hidden_states_h = hidden_states_h.view(-1, 1, hidden_states_h.shape[2])
    hidden_states_c = hidden_states_c.view(-1, 1, hidden_states_c.shape[2])


    actions = torch.cat(actions)
    advantage = returns - values

    ppo_update(ppo_epochs, mini_batch_size, map_states, depth_states, goal_states, hidden_states_h, hidden_states_c , actions, log_probs, returns, advantage, PPO_EPSILON, CRICIC_DISCOUNT, ENTROPY_BETA)
    epoch += 1.0
    #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #if(lr>1e-4):
     #   lr=1e-4
    print('learning rate: ' + str(lr))
    optimizer = optim.Adam(model.parameters(), lr=lr )


from itertools import count

print("test the values")
max_expert_num = 50000
num_steps = 0
expert_traj = []

model.eval()

with torch.no_grad():
    for i_episode in count():
        map_state, depth_state, goal_state = envs.reset_test(1)

        map_state = map_state[0]
        depth_state = depth_state[0]
        goal_state = goal_state[0]

        map_state = np.reshape(map_state, (-1, map_state.shape[0], map_state.shape[1]))
        depth_state = np.reshape(depth_state, (-1, depth_state.shape[0], depth_state.shape[1]))
        goal_state = np.reshape(goal_state, (-1, goal_state.shape[0]))

        map_state = np.concatenate([map_state for x in range(num_envs)])
        depth_state = np.concatenate([depth_state for x in range(num_envs)])
        goal_state = np.concatenate([goal_state for x in range(num_envs)])

        map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, stack_size, True)
        depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, stack_size, True)
        goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, stack_size, True)

        model.hidden = model.init_hidden(1)
        (hidden_state_h, hidden_state_c) = model.hidden

        done = False
        total_reward = 0

        while not done:
            map_state = torch.FloatTensor(map_state[0]).unsqueeze(0).to(device)
            depth_state = torch.FloatTensor(depth_state[0]).unsqueeze(0).to(device)
            goal_state = torch.FloatTensor(goal_state[0]).unsqueeze(0).to(device)

            dist, _, hidden_state_h, hidden_state_c = model(map_state, depth_state, goal_state, hidden_state_h,
                                                            hidden_state_c)

            #action = dist.sample().cpu().numpy()
            print("dist.mean.detach()" + str(dist.mean.detach()))
            print("dist.mean" + str(dist.mean))
            action = dist.mean.detach().cpu().numpy()

            next_map_state, next_depth_state, next_goal_state, reward, done, _ = envs.step_test(action,1)

            next_map_state = next_map_state[0]
            next_depth_state = next_depth_state[0]
            next_goal_state = next_goal_state[0]

            next_map_state = np.reshape(next_map_state, (-1, next_map_state.shape[0], next_map_state.shape[1]))
            next_depth_state = np.reshape(next_depth_state, (-1, next_depth_state.shape[0], next_depth_state.shape[0]))
            next_goal_state = np.reshape(next_goal_state, (-1, next_goal_state.shape[0]))

            next_map_state = np.concatenate([next_map_state for x in range(num_envs)])
            next_depth_state = np.concatenate([next_depth_state for x in range(num_envs)])
            next_goal_state = np.concatenate([next_goal_state for x in range(num_envs)])

            next_map_state, stacked_map_frames = stack_frames(stacked_map_frames, next_map_state, stack_size, False)
            next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, next_depth_state, stack_size,
                                                                  False)
            next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, next_goal_state, stack_size, False)

            map_state = next_map_state
            depth_state = next_depth_state
            goal_state = next_goal_state

            total_reward += reward
            #expert_traj.append(np.hstack([map_state, depth_state, goal_state, action]))
            num_steps += 1

        print("episode:", i_episode, "reward:", total_reward)

        if num_steps >= max_expert_num:
            break

#expert_traj = np.stack(expert_traj)
print()
#print(expert_traj.shape)
print()
#np.save("expert_traj.npy", expert_traj)

#plt.show()

