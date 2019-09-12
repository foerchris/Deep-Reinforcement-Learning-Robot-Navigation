import matplotlib.pyplot as plt
import numpy as np
import math as m
import torch
import cv2


import torch
import torch.nn as nn
from numpy import number
from torch.distributions import Normal

std = 0.5
log_std = nn.Parameter(torch.ones(1, 2) * std)

print(torch.ones(1, 2).shape)

print(torch.ones(1, 2))

print(log_std.shape)

print(log_std)
print(log_std.exp())
print(log_std.exp().expand_as(torch.zeros(1,2)))