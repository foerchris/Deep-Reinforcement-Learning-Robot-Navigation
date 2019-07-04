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




class ActorCritic(nn.Module):
    def __init__(self,state_size_map, state_size_depth , state_size_goal, num_outputs, hidden_size, stack_size, std=0.0):
        self.state_size_map = state_size_map
        self.state_size_depth = state_size_depth
        self.state_size_goal = state_size_goal
        self.stack_size = stack_size
        self.hidden_dim = hidden_size
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

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
            nn.Conv2d(in_channels=stack_size*4, out_channels=32, kernel_size=5, stride=1, padding=2),
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

        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=1)

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
    def init_weights(self,m):
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

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device))

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

        return dist, value, hidden_h.detach(), hidden_c.detach()

class ICMModel(nn.Module):
    def __init__(self,state_size_map, state_size_depth , state_size_goal, num_outputs, hidden_size, stack_size, std=0.0):
        self.state_size_map = state_size_map
        self.state_size_depth = state_size_depth
        self.state_size_goal = state_size_goal
        self.stack_size = stack_size
        self.hidden_dim = hidden_size
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(ICMModel, self).__init__()

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
            nn.Conv2d(in_channels=stack_size*4, out_channels=32, kernel_size=5, stride=1, padding=2),
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

        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=1)

        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )

        self.residual = [nn.Sequential(
            nn.Linear(num_outputs + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device)]* 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(num_outputs + hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(num_outputs + hidden_size, hidden_size),
        )

#         for p in self.modules():
#             if isinstance(p, nn.Conv2d):
#                 init.kaiming_uniform_(p.weight)
#                 p.bias.data.zero_()
#
#             if isinstance(p, nn.Linear):
#                 init.kaiming_uniform_(p.weight, a=1.0)
#                 p.bias.data.zero_()

    def init_weights(self,m):
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


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(1, batch_size, self.hidden_dim, device=self.device))

    def forward(self, map_state, depth_state , goal_state, hidden_h, hidden_c, next_map_state, next_depth_state , next_goal_state, next_hidden_h, next_hidden_c, action):
        self.hidden = (hidden_h, hidden_c)
        self.next_hidden = (next_hidden_h, next_hidden_c)

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

        encode_state = lstm_out.view(-1, lstm_out.shape[2])

        #33333333333333333333333333333333###33333333333333
        next_map_state = next_map_state.view(-1, next_map_state.shape[1], next_map_state.shape[2], next_map_state.shape[3])

        map = self.cnn_map(next_map_state)


        next_goal_state = next_goal_state.view(-1, next_goal_state.shape[1]* next_goal_state.shape[2])

        goal = self.cnn_goal(next_goal_state)

        goal = goal.view(-1, map.shape[1], map.shape[2], map.shape[3])


        map_and_goal = map.add(goal)

        map_goal_out = self.cnn_map_goal(map_and_goal)

        map_goal_out = map_goal_out.view(-1, map_goal_out.shape[1] * map_goal_out.shape[2] * map_goal_out.shape[3])

        next_depth_state = next_depth_state.view(-1, next_depth_state.shape[1], next_depth_state.shape[2], next_depth_state.shape[2])

        depth_out = self.cnn_depth(next_depth_state)
        depth_out = depth_out.view(-1, depth_out.shape[1] * depth_out.shape[2] * depth_out.shape[3])

        map_goal_depth = torch.cat((map_goal_out, depth_out), 1)

        map_goal_depth = map_goal_depth.view(1, -1, map_goal_depth.shape[1])

        lstm_out, self.next_hidden = self.lstm(map_goal_depth, self.next_hidden)

        encode_next_state = lstm_out.view(-1, lstm_out.shape[2])


        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------
        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)

        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = torch.cat((pred_next_state_feature_orig, action), 1)
            pred_next_state_feature = self.residual[i * 2](pred_next_state_feature)
            pred_next_state_feature_orig = self.residual[i * 2 + 1](torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig


        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action
