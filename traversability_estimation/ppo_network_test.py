#!/usr/bin/env python2
# -*- coding: utf-8

import math
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

from tensorboardX import SummaryWriter

import os
#from hgext.histedit import action
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, 'common'))
from flipper_model2 import FeatureNetwork, ActorCritic
from inspect import currentframe, getframeinfo

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()

num_envs = 5

state_size_map = 28*28
state_size_orientation = 7
stack_size = 1
hidden_size = 288
num_outputs = 2

feature_net = FeatureNetwork(state_size_map*stack_size, state_size_orientation * stack_size , hidden_size, stack_size).to(device)

ac_model = ActorCritic(num_outputs, hidden_size).to(device)

feature_net.apply(feature_net.init_weights)
ac_model.apply(ac_model.init_weights)

feature_net.hidden = feature_net.init_hidden(num_envs)
(hidden_state_h, hidden_state_c) = feature_net.hidden

map_state = np.zeros( (5,1, 28, 28), np.float32 )
orientation_state = np.zeros( (5,1, 7), np.float32 )
map_state = torch.FloatTensor(map_state).to(device)
orientation_state = torch.FloatTensor(orientation_state).to(device)
features, next_hidden_state_h, next_hidden_state_c = feature_net(map_state, orientation_state, hidden_state_h, hidden_state_c)

dist, value, std  = ac_model( features)
