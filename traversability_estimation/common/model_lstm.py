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
from torch.nn import init
import matplotlib.pyplot as plt


class FeatureNetwork(nn.Module):
    def __init__(self,state_size_map, state_size_depth , state_size_goal, hidden_size, stack_size, lstm_layers, std=0.0):
        self.state_size_map = state_size_map
        self.state_size_depth = state_size_depth
        self.state_size_goal = state_size_goal
        self.stack_size = stack_size
        self.hidden_dim = hidden_size
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(FeatureNetwork, self).__init__()

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
            nn.Linear(state_size_goal, 5184),
            nn.ReLU()
        )

        self.cnn_map_goal = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            #nn.init.normal_(m.weight, mean=0., std=0.05)
            torch.nn.init.orthogonal_(m.weight.data)
            #nn.init.constant_(m.bias, 0.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            #torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in self.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(p)
                elif 'bias' in name:
                    init.constant_(p, 0)
        elif isinstance(m, nn.LSTMCell):
            for name, p in self.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(p)
                elif 'bias' in name:
                    init.constant_(p, 0)

    def forward(self, map_state, depth_state ,goal_state):

        map = self.cnn_map(map_state)

        goal_state = goal_state.view(-1, goal_state.shape[1]* goal_state.shape[2])

        goal = self.cnn_goal(goal_state)

        goal = goal.view(-1, map.shape[1], map.shape[2], map.shape[3])


        map_and_goal = map.add(goal)
        map_goal_out = self.cnn_map_goal(map_and_goal)

        map_goal_out = map_goal_out.view(-1, map_goal_out.shape[1] * map_goal_out.shape[2] * map_goal_out.shape[3])

        depth_out = self.cnn_depth(depth_state)

        depth_out = depth_out.view(-1, depth_out.shape[1] * depth_out.shape[2] * depth_out.shape[3])

        map_goal_depth = torch.cat((map_goal_out, depth_out), 1)

        map_goal_depth = map_goal_depth.view(1, -1, map_goal_depth.shape[1])

        return map_goal_depth


class ActorCritic(nn.Module):
    def __init__(self, num_outputs,hidden_size, std=0.0):
        self.hidden_dim = hidden_size/2
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size/2),
            nn.ReLU()
        )


        self.critic_lstm = nn.LSTM(hidden_size/2,hidden_size/2, num_layers=1)

        self.critic2 = nn.Sequential(
            nn.Linear(hidden_size/2, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size/2),
            nn.ReLU()
        )

        self.actor_lstm = nn.LSTM(hidden_size/2, hidden_size/2, num_layers=1)

        self.actor2 = nn.Sequential(
            nn.Linear(hidden_size/2, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in self.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(p)
                elif 'bias' in name:
                    init.constant_(p, 0)
        elif isinstance(m, nn.LSTMCell):
            for name, p in self.named_parameters():
                if 'weight' in name:
                    init.orthogonal_(p)
                elif 'bias' in name:
                    init.constant_(p, 0)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device))


    def forward(self, out,  actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c):


        out = (out - out.mean()) / (out.std() + 1e-8)


        out = out.view(-1, self.hidden_dim*2)

        value = self.critic(out)


        value = value.view(1, -1, self.hidden_dim)

        critic_hidden = (critic_hidden_h, critic_hidden_c)


        value, critic_hidden = self.critic_lstm(value, critic_hidden)
        #lstm_out, self.hidden = self.lstm(map_goal_depth, self.hidden)



        value = value.view(-1, value.shape[2])

        value = self.critic2(value)


        critic_hidden_h = critic_hidden[0]
        critic_hidden_c = critic_hidden[1]

        mu = self.actor(out)

        mu = mu.view(1, -1, self.hidden_dim)

        actor_hidden = (actor_hidden_h, actor_hidden_c)

        mu, actor_hidden  = self.actor_lstm(mu, actor_hidden)

      #  mu = nn.functional.tanh(mu)


        mu = mu.view(-1, mu.shape[2])

        actor_hidden_h = actor_hidden[0]
        actor_hidden_c = actor_hidden[1]

        mu = self.actor2(mu)


        std = self.log_std.exp().expand_as(mu)

        dist = Normal(mu, std)

        return dist, value, std, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c
