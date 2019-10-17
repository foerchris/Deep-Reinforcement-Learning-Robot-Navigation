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
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, 'common'))
from model_lstm import FeatureNetwork, ActorCritic

class Agent():
    def __init__(self,state_size_map, state_size_depth , state_size_goal, num_outputs, hidden_size, stack_size, lstm_layers,load_model, MODELPATH, learning_rate, mini_batch_size, worker_number, lr_decay_epoch, init_lr, eta = 0.01):
        self.eta = eta
        self.lr_decay_epoch = lr_decay_epoch
        self.init_lr = init_lr
        self.final_lr = 1e-5
        self.lstm_layers = lstm_layers
        self.mini_batch_size = mini_batch_size
        self.worker_number = worker_number
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        torch.cuda.empty_cache()
        self.summary_writer = tf.summary.FileWriter("train_getjag/ppo/Tensorboard")

        self.feature_net = FeatureNetwork(state_size_map*stack_size, state_size_depth * stack_size, state_size_goal * stack_size, hidden_size, stack_size, lstm_layers).to(self.device)
        self.ac_model = ActorCritic(num_outputs, hidden_size).to(self.device)

        if(load_model):
            self.feature_net.load_state_dict(torch.load(MODELPATH + '/save_ppo_feature_net.dat'))
            self.ac_model.load_state_dict(torch.load(MODELPATH + '/save_ppo_ac_model.dat'))
        else:
            self.feature_net.apply(self.feature_net.init_weights)
            self.ac_model.apply(self.ac_model.init_weights)

        self.optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.ac_model.parameters()), lr=learning_rate)


    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns


    def ppo_iter(self, map_states, depth_states, goal_states, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c, actions, log_probs, returns, advantage, value):
        batch_size = map_states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size-self.worker_number, self.mini_batch_size)
            #print('map_states[rand_ids, :].shape' + str(map_states[rand_ids, :].shape))
            yield map_states[rand_ids, :], depth_states[rand_ids, :], goal_states[rand_ids, :], actor_hidden_h[rand_ids, :], actor_hidden_c[rand_ids, :], critic_hidden_h[rand_ids, :], critic_hidden_c[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], value[rand_ids, :]


    def ppo_update(self, frame_idx, ppo_epochs, map_states, depth_states, goal_states, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c, actions, log_probs, returns, advantages, values, epoch, clip_param=0.2, discount=0.5, beta=0.001, max_grad_norm =0.5):
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for update in range(1 ,ppo_epochs +1 ):
            for  map_state, depth_state, goal_state, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c, action, old_log_probs, return_, advantage, old_value in self.ppo_iter(map_states, depth_states, goal_states, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c, actions, log_probs,
                                                                            returns, advantages, values):

                actor_hidden_h = actor_hidden_h.view(1, -1, actor_hidden_h.shape[2])
                actor_hidden_c = actor_hidden_c.view(1, -1, actor_hidden_c.shape[2])

                critic_hidden_h = critic_hidden_h.view(1, -1, critic_hidden_h.shape[2])
                critic_hidden_c = critic_hidden_c.view(1, -1, critic_hidden_c.shape[2])

                frac = 1.0 - (update -1.0) / ppo_epochs

                lrnow = self.init_lr * frac

                self.optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.ac_model.parameters()), lr=lrnow)

                features = self.feature_net(map_state, depth_state, goal_state)
                dist, value, _, _, _ ,_ ,_ = self.ac_model(features, actor_hidden_h, actor_hidden_c,  critic_hidden_h, critic_hidden_c)

                vpredclipped = old_value + torch.clamp(value - old_value , - clip_param, clip_param)

                #Unclipped value
                vf_losses1 = - (value - return_).pow(2)

                vf_losses2 = - (vpredclipped - return_).pow(2)


                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()

                surr1 = ratio * advantage

                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()

                #critic_loss = (return_ - value).pow(2).mean()

                critic_loss =  .5 * (-torch.min(vf_losses1, vf_losses2).mean())

                loss = discount * critic_loss + actor_loss - beta * entropy

                self.optimizer.zero_grad()


                loss.backward()

                nn.utils.clip_grad_norm_(self.feature_net.parameters(),max_grad_norm)

                nn.utils.clip_grad_norm_(self.ac_model.parameters(),max_grad_norm)

                #for p in model.parameters():
                    #p.data.add_(-lr, p.grad.data)
                self.optimizer.step()

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
        self.summary_writer.add_summary(summary, frame_idx)
