#!/usr/bin/python
import tensorflow as tf
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
im = ax.imshow(depth_state[0][0])
ax.axis('off')
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(depth_state[0][1])
ax.axis('off')
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(depth_state[0][2])
ax.axis('off')
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(depth_state[0][3])
ax.axis('off')
plt.show()
