#!/usr/bin/python
import tensorflow as tf
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
import matplotlib.pyplot as plt
from torch.autograd import Variable

map_states = torch.zeros((400,5,200,200))
#state = np.zeros((200,200))

mini_batch_size = 4

batch_size = 5
for _ in range(batch_size // mini_batch_size):
    rand_ids = np.random.randint(0, batch_size, mini_batch_size)
    bla = map_states[rand_ids, :]

action = torch.zeros((5,2)).long()

action = torch.LongTensor(action).to(device)
#action_onehot = torch.FloatTensor(
#            len(action), action.size(1)).to(
#            device)
#action_onehot.zero_()
#action_onehot.scatter_(1, action.view(len(action).to(
#            device), -1), 1)
output_size = 2
action_onehot = torch.FloatTensor(
            len(action), output_size).to(
            device)
action_onehot.zero_()


action_onehot.scatter_(1, action.view(len(action), -1), 1)

criterion = nn.CrossEntropyLoss()

output = Variable(torch.randn(5, 4).float())
target = Variable(torch.FloatTensor(5).uniform_(0, 4).long())



loss = criterion(output, target)


y_batch = torch.randn(5, 4).float()

y_batch = torch.FloatTensor(F.softmax(y_batch, dim=-1).data.cpu().numpy())


pred_action = Variable(torch.FloatTensor(abs(torch.randn(5))).long())


loss = criterion(y_batch, pred_action)


batch_size = 1
c, h, w = 1, 10, 10
nb_classes = 3
x = torch.randn(batch_size, c, h, w)
print(x)

target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)
print(target)

model = nn.Conv2d(c, nb_classes, 3, 1, 1)
criterion = nn.CrossEntropyLoss()

output = model(x)
print(output)
print(x.shape)
print(target.shape)
print(output.shape)

loss = criterion(output, target)

w, h = 512, 512
data = np.zeros((h, w), dtype=np.uint8)
img = Image.fromarray(data, 'P')
img.save('my.png')
img.show()
