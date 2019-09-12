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
    def __init__(self,state_size_map, state_size_depth , state_size_goal, hidden_size, stack_size, std=0.0):
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

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=3)

        #self.hidden = self.init_hidden()
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
            #for name, p in self.named_parameters():
             #   if 'weight' in name:
             #       init.orthogonal_(p)
              #  elif 'bias' in name:
               #     init.constant_(p, 0)


            for name, p in self.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(p)
                    elif 'bias' in name:
                        init.constant_(p, 0)
        elif isinstance(m, nn.LSTMCell):

            #for name, p in self.named_parameters():
             #   if 'weight' in name:
              #      init.orthogonal_(p)
               # elif 'bias' in name:
                #    init.constant_(p, 0)

            for name, p in self.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(p)
                    elif 'bias' in name:
                        init.constant_(p, 0)

           # for param in m.parameters():
               # if len(param.shape) >= 2:
                    #torch.nn.init.orthogonal_(param.data)
               # for name, p in self.named_parameters():
                 #   if 'weight' in name:
                  #      init.orthogonal_(p)
                  #  elif 'bias' in name:
                   #     init.constant_(p, 0)
              #  else:
                    #torch.nn.init.normal_(param.data)
                 #   for name, p in self.named_parameters():
                 #       if 'weight' in name:
                #            init.normal_(p)
                   #     elif 'bias' in name:
                  #          init.constant_(p, 0)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(3, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(3, batch_size, self.hidden_dim, device=self.device))

    def forward(self, map_state, depth_state ,goal_state, hidden_h, hidden_c):

        self.hidden = (hidden_h, hidden_c)

       # plt.imshow(map_state[0][0].cpu().numpy() ,cmap="gray")
        #plt.show()
        #map_state= map_state.view(-1,map_state.shape[2], map_state.shape[3], map_state.shape[4])
        #plt.imshow(map_state[0][0].cpu().numpy(),cmap="gray")
        #plt.show()
        #plt.imshow(map_state[0][1].cpu().numpy(),cmap="gray")
        #plt.show()
        map = self.cnn_map(map_state)
        # print("goal_state"+str(goal_state))
        # print("goal_state[0][0]"+str(goal_state[0][0]))
        #
        # orientation = goal_state[0][0].cpu().numpy()
        # position = orientation[:3]
        # print("position"+str(position))
        #
        # orientation = orientation[3:]
        # print("orientation"+str(orientation))
        # position = np.multiply(position,10)
        # print("position"+str(position))
        #
        # bla = np.multiply(orientation,math.pi)
        # print("orientation_state"+str(bla/math.pi*180.0))
        # plt.imshow(map_state[0][0].cpu().numpy(),cmap="gray")
        # plt.show()
        #robotGroundMap =  np.multiply(map_state[0][0].cpu().numpy(), 2e8) #255


        goal_state = goal_state.view(-1, goal_state.shape[1]* goal_state.shape[2])

        goal = self.cnn_goal(goal_state)

        goal = goal.view(-1, map.shape[1], map.shape[2], map.shape[3])


        map_and_goal = map.add(goal)

        map_goal_out = self.cnn_map_goal(map_and_goal)

        map_goal_out = map_goal_out.view(-1, map_goal_out.shape[1] * map_goal_out.shape[2] * map_goal_out.shape[3])


        #plt.imshow(depth_state[0][0].cpu().numpy() ,cmap="gray")
        #plt.show()
        #plt.imshow(depth_state[0][1].cpu().numpy() ,cmap="gray")
        #plt.show()

        depth_out = self.cnn_depth(depth_state)

        #print('depth_out.shape' + str(depth_out.shape))
        #plt.imshow(depth_out[0][0].cpu().numpy() ,cmap="gray")
        #plt.show()
        depth_out = depth_out.view(-1, depth_out.shape[1] * depth_out.shape[2] * depth_out.shape[3])

        map_goal_depth = torch.cat((map_goal_out, depth_out), 1)

        map_goal_depth = map_goal_depth.view(1, -1, map_goal_depth.shape[1])

        lstm_out, self.hidden = self.lstm(map_goal_depth, self.hidden)

       # print("lstm_out.shape" + str(lstm_out.shape))
        lstm_out = lstm_out.view(-1, lstm_out.shape[2])
       # print("lstm_out.shape" + str(lstm_out.shape))

        hidden_h = self.hidden[0]
        hidden_c = self.hidden[1]
        #map_goal_depth = map_goal_depth.view(-1, map_goal_depth.shape[2])

        #return map_goal_depth, hidden_h.detach(), hidden_c.detach()
        return lstm_out, hidden_h, hidden_c


class ActorCritic(nn.Module):
    def __init__(self, num_outputs,hidden_size, std=0.0):
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            #nn.init.normal_(m.weight, mean=0., std=0.05)
            torch.nn.init.orthogonal_(m.weight.data)
            #torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
            #nn.init.constant_(m.bias, 0.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
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

   # def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
  #      return (torch.zeros(1, batch_size, self.hidden_dim, device=self.device),
  #              torch.zeros(1, batch_size, self.hidden_dim, device=self.device))

    def forward(self, lstm_out):
        value = self.critic(lstm_out)
        #print('value' + str(value))

        mu = self.actor(lstm_out)

        std = self.log_std.exp().expand_as(mu)
        #print('mu' + str(mu))
        #print('std' + str(std))
        dist = Normal(mu, std)

        return dist, value, std
