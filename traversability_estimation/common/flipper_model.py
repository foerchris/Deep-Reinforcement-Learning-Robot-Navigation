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
from nipy.modalities.fmri.fmristat.tests.FIACdesigns import dtype
import cv2
from inspect import currentframe, getframeinfo

class FeatureNetwork(nn.Module):
    def __init__(self,state_size_map, state_size_orientation, hidden_size, stack_size, std=0.0):
        self.state_size_map = state_size_map
        self.state_size_orientation = state_size_orientation
        self.stack_size = stack_size
        self.hidden_dim = hidden_size

        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(FeatureNetwork, self).__init__()

        #This is the ElevMap part
        self.cnn_map = nn.Sequential(
            nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )

        # This is the orientation pose part
        self.cnn_orientation = nn.Sequential(
            nn.Linear(state_size_orientation, 4608),
            nn.ReLU()
        )

        self.cnn_map_orientation = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers=1)

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

    def forward(self, map_state, orientation_state, hidden_h, hidden_c):

        self.hidden = (hidden_h, hidden_c)

        #plt.imshow(map_state[0][0].cpu().numpy(),cmap="gray")
        #plt.show()
        #robotGroundMap =  np.multiply(map_state[0][0].cpu().numpy(), 2e8) #255
        #bla = np.multiply(orientation_state[0][0],math.pi)
        #print("orientation_state"+str(bla/math.pi*180.0))
        #cv2.imshow('image',robotGroundMap)
        #cv2.waitKey(2)
        map = self.cnn_map(map_state)

        #print("orientation_state" + str(orientation_state))
        orientation_state = orientation_state.view(-1, orientation_state.shape[1]* orientation_state.shape[2])

        orientation = self.cnn_orientation(orientation_state)

        orientation = orientation.view(-1, map.shape[1], map.shape[2], map.shape[3])

        map_and_orientation = map.add(orientation)

        map_orientation_out = self.cnn_map_orientation(map_and_orientation)


        map_orientation_out = map_orientation_out.view(1, map_state.shape[0], -1)


        lstm_out, self.hidden = self.lstm(map_orientation_out, self.hidden)

        lstm_out = lstm_out.view(-1, lstm_out.shape[2])

        hidden_h = self.hidden[0]
        hidden_c = self.hidden[1]

        return lstm_out, hidden_h.detach(), hidden_c.detach()

class ActorCritic(nn.Module):
    def __init__(self, num_outputs,hidden_size, std=0.0):
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()

        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

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
