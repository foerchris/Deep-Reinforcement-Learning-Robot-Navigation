#!/usr/bin/env python2
# -*- coding: utf-8

import threading
import multiprocessing
import rospy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from collections import deque
from matplotlib import pyplot as plt
from robotEnvWrapper_local import robotEnv

from helper import *

from time import sleep

stack_images = False


def all_actions():

    # Here our possible actions
    hard_left = [1, 0, 0, 0, 0]
    left = [0, 1, 0, 0, 0]
    forward = [0, 0, 1, 0, 0]
    right = [0, 0, 0, 1, 0]
    hard_right = [0, 0, 0, 0, 1]
    possible_actions = [hard_left, left, forward, right, hard_right]

    return  possible_actions


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder




stack_size = 1  # We stack 4 frames





# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



global_Network = True

class Global_Network():
    def __init__(self, state_size_depth, state_size_map, state_size_goal, a_size, scope, trainer, stack_size):
        self.state_size_depth = state_size_depth
        self.state_size_map = state_size_map
        self.state_size_goal = state_size_goal
        self.a_size = a_size
        self.stack_size = stack_size
        with tf.variable_scope(scope):

            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [4 * 7056]
            self.depthImages = tf.placeholder(tf.float32, [None, state_size_depth*stack_size], name="depthImages")
            # [84, 84, 4]
            self.imageIn = tf.reshape(self.depthImages, shape=[-1, 84, 84, stack_size])
            # [4 * 40000]
            self.eleviationMaps = tf.placeholder(tf.float32, [None, state_size_map*stack_size], name="eleviationMaps")
            # [200, 200, 4]
            self.mapIn = tf.reshape(self.eleviationMaps, shape=[-1, 200, 200, stack_size])
            # [4*6]
            self.goal = tf.placeholder(tf.float32, [None, state_size_goal*stack_size], name="goal")
            # [6, 4]
            self.goalIn = tf.reshape(self.goal, shape=[-1, 6, stack_size])
            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, a_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            ## This is the depthImage part
            self.padded_imageIn = tf.pad(self.imageIn, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.padded_imageIn, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='VALID')

            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

            self.padded_conv1 = tf.pad(self.pool1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.padded_conv1, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='VALID')

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)

            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.pool2, num_outputs=64,
                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=2)

            self.conv4 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.pool3, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], strides=2)
            self.flatten1 = slim.flatten(self.pool4)
            #self.hidden1 = slim.fully_connected(self.flatten1, 576, activation_fn=tf.nn.relu)

            # This is the ElevMap part
            self.convMap1 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=self.mapIn, num_outputs=32,
                                        kernel_size=[7, 7], stride=[1, 1], padding='VALID')
            self.poolMap1 = tf.layers.max_pooling2d(inputs=self.convMap1, pool_size=[2, 2], strides=2)

            self.convMap2 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=self.poolMap1, num_outputs=32,
                                        kernel_size=[6, 6], stride=[1, 1], padding='VALID')
            self.poolMap2 = tf.layers.max_pooling2d(inputs=self.convMap2, pool_size=[2, 2], strides=2)

            self.convMap3 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=self.poolMap2, num_outputs=64,
                                        kernel_size=[5, 5], stride=[1, 1], padding='VALID')
            self.poolMap3 = tf.layers.max_pooling2d(inputs=self.convMap3, pool_size=[2, 2], strides=2)

            self.convMap4 = slim.conv2d(activation_fn=tf.nn.relu,
                                        inputs=self.poolMap3, num_outputs=64,
                                        kernel_size=[4, 4], stride=[1, 1], padding='VALID')
            self.poolMap4 = tf.layers.max_pooling2d(inputs=self.convMap4, pool_size=[2, 2], strides=2)

            # This is the orientation part
            # self.fconOrientation = slim.fully_connected(slim.flatten(self.orientationIn), 3, activation_fn=tf.nn.elu)
            # self.reshapedOriantation = tf.reshape(self.fconOrientation, shape=[9, 9, 64])

            # This is the goal part
            self.flatten2 = slim.flatten(self.goalIn)

            self.fconGoal = slim.fully_connected(self.flatten2 , 5184, activation_fn=tf.nn.relu)

            self.reshapedGoal = tf.reshape(self.fconGoal, shape=[-1, 9, 9, 64])

            self.addGoal = tf.add(
                self.poolMap4,
                self.reshapedGoal,
                name=None)

            self.convMapAndOrientation = slim.conv2d(activation_fn=tf.nn.relu,
                                                     inputs=self.addGoal, num_outputs=64,
                                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')

            self.poolMapAndOrientation = tf.layers.max_pooling2d(inputs=self.convMapAndOrientation, pool_size=[2, 2],
                                                                 strides=2)
            self.flatten3 = slim.flatten(self.poolMapAndOrientation)

            self.concateLayer = tf.concat([self.flatten1, self.flatten3], -1)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.LSTMCell(1152, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(self.concateLayer, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 1152])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))




class Worker():
    def __init__(self, number,state_size_depth, state_size_map, state_size_goal, a_size, trainer, model_path, global_episodes, stack_size):
        self.name = "worker_" + str(number)
        self.number = number
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.robotEnv = robotEnv(self.number+1)
        self.actions = all_actions()
        self.state_size_depth = state_size_depth
        self.state_size_map = state_size_map
        self.state_size_goal = state_size_goal
        self.stack_size = stack_size
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = Global_Network(state_size_depth,state_size_map,state_size_goal, a_size, self.name, trainer,stack_size)
        self.summary_writer = tf.summary.FileWriter("train_getjag/train_" + str(self.number))

        self.update_local_ops = update_target_graph('global', self.name)
        # Initialize deque with zero-images one array for each image
        self.stacked_depth_frames = deque([np.zeros((state_size_depth), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        self.stacked_map_frames = deque([np.zeros((state_size_map), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        self.stacked_goal_frames = deque([np.zeros((state_size_goal), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

    def preprocess_state(self, state, state_size):
        return np.reshape(state, [state_size])

    def stack_frames(self, stacked_frames, state, state_size, is_new_episode):
        # Preprocess frame
        # because of different functions this as to be done beforhand
        # frame = preprocess_frame(state)
        state = self.preprocess_state(state, state_size)

        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((state_size), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

            # Because we're in a new episode, copy the same frame 4x
            for i in range(1, stack_size, 1):
                stacked_frames.append(state)

            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=0)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=0)

        stacked_state = self.preprocess_state(stacked_state, state_size * stack_size)

        return stacked_state, stacked_frames

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations1 = np.array(rollout[:, 0])
        observations2 = np.array(rollout[:, 1])
        observations3 = np.array(rollout[:, 2])
        actions = np.array(rollout[:, 3])
        rewards = np.array(rollout[:, 4])
        next_observations1 = np.array(rollout[:, 5])
        next_observations2 = np.array(rollout[:, 6])
        next_observations3 = np.array(rollout[:, 7])
        values = np.array(rollout[:, 8])
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.depthImages: np.vstack(observations1),
                     self.local_AC.eleviationMaps: np.vstack(observations2),
                     self.local_AC.goal: np.vstack(observations3),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, buffer_rollout):
        episode_count = sess.run(self.global_episodes)
        self.robotEnv.set_episode_length(max_episode_length)
        episode = 0
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            if(self.number == 0):
                #train_writer = tf.summary.FileWriter("train_" + str(0), sess.graph)
                test_writer = tf.summary.FileWriter("train_getjag/train_" + str(0), sess.graph)
                allTrainVars = tf.trainable_variables()
                weight_name_mapping = lambda x: x.replace(":", "_")

                var = [v for v in allTrainVars if v.name == "global/Conv/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_1/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_1/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_2/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_2/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_3/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_3/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_4/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_4/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_5/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_5/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_6/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_6/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_7/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_7/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/fully_connected/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/fully_connected/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/Conv_8/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/Conv_8/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/rnn/lstm_cell/kernel:0"][0]
                tf.summary.histogram(weight_name_mapping("global/rnn/lstm_cell/kernel/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/fully_connected_1/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/fully_connected_1/weights:0"), var)
                var = [v for v in allTrainVars if v.name == "global/fully_connected_2/weights:0"][0]
                tf.summary.histogram(weight_name_mapping("global/fully_connected_2/weights:0"), var)

            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_depth_frames = []
                episode_map_frames = []
                episode_goal_frames = []

                episode_reward = 0
                episode_step_count = 0
                d = False
                reward = 0
                self.robotEnv.reset()

                depth_state, map_state, goal_state = self.robotEnv.getState()

                episode_depth_frames.append(depth_state)
                episode_map_frames.append(map_state)
                episode_goal_frames.append(goal_state)

                depth_state, self.stacked_depth_frames = self.stack_frames(self.stacked_depth_frames, depth_state, state_size_depth, True)
                map_state, self.stacked_map_frames = self.stack_frames(self.stacked_map_frames, map_state, state_size_map, True)
                goal_state, self.stacked_goal_frames = self.stack_frames(self.stacked_goal_frames, goal_state, state_size_goal, True)

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while d == False:

                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state, depthImages, eleviationMaps,goal_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out,self.local_AC.depthImages,self.local_AC.eleviationMaps, self.local_AC.goal],
                        feed_dict={self.local_AC.depthImages: [depth_state],
                                   self.local_AC.eleviationMaps: [map_state],
                                   self.local_AC.goal: [goal_state],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    if (self.number == 0):
                        print("a_dist" + str(a_dist))

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    if (self.number == 0):
                        print("a_dist" + str(a_dist))
                    a = np.argmax(a_dist == a)
                    reward, d = self.robotEnv.takeStep(a)

                    depth_state, map_state, goal_state = self.robotEnv.getState()
                    depth_state, self.stacked_depth_frames = self.stack_frames(self.stacked_depth_frames, depth_state, state_size_depth, False)
                    map_state, self.stacked_map_frames = self.stack_frames(self.stacked_map_frames, map_state, state_size_map, False)
                    goal_state, self.stacked_goal_frames = self.stack_frames(self.stacked_goal_frames, goal_state, state_size_goal, False)

                    if d == False:
                        next_depth_state, next_map_state, next_goal_state = self.robotEnv.getState()

                        episode_depth_frames.append(next_depth_state)
                        episode_map_frames.append(next_map_state)
                        episode_goal_frames.append(next_goal_state)


                        next_depth_state, self.stacked_depth_frames = self.stack_frames(self.stacked_depth_frames, next_depth_state, state_size_depth, False)
                        next_map_state, self.stacked_map_frames = self.stack_frames(self.stacked_map_frames, next_map_state, state_size_map, False)
                        next_goal_state, self.stacked_goal_frames = self.stack_frames(self.stacked_goal_frames, next_goal_state, state_size_goal, False)

                    else:
                        next_depth_state = depth_state
                        next_map_state = map_state
                        next_goal_state = goal_state

                    episode_buffer.append([depth_state, map_state, goal_state, a, reward, next_depth_state, next_map_state, next_goal_state, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += reward
                    depth_state = next_depth_state
                    map_state = next_map_state
                    goal_state = next_goal_state

                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == buffer_rollout and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        self.robotEnv.stopSteps()
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.depthImages: [depth_state],
                                                 self.local_AC.eleviationMaps: [map_state],
                                                 self.local_AC.goal: [goal_state],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        self.robotEnv.stopSteps()
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)


                episode +=1


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:

                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        text_file = open("a3c_results.txt", "a")
                        text_file.write("Saved Model" + "\n")
                        text_file.close()
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")


                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_reward_long = 0;
                    if(self.episode_rewards.__sizeof__()>20):
                        mean_reward_long = np.mean(self.episode_rewards[-20:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    print('worker:' + str(self.name) +
                          'episode: ' + str(episode) +
                          'Total reward: ' + str(episode_reward) +
                          'episode_step_count: ' + str(episode_step_count) + "\n" +
                          'mean_reward: ' + str(mean_reward) +
                          'mean_length: ' + str(mean_length) +
                          'mean_value: ' + str(mean_value) +
                          'mean_reward_long: ' + str(mean_reward_long) + "\n" +
                          'Losses/Value Loss: ' + str(v_l) +
                          'Losses/Policy Loss: ' + str(p_l) +
                          'Losses/Entropy: ' + str(e_l) +
                          'Losses/Grad Norm: ' + str(g_n) +
                          'Losses/Var Norm: ' + str(v_n) + "\n")

                    text_file = open("a3c_results.txt", "a")
                    text_file.write('worker:    ' + str(self.name) +
                                    'episode:   ' + str(episode) +
                                    'Total reward:      ' + str(episode_reward) +
                                    'episode_step_count:    ' + str(episode_step_count) + "\n" +
                                    'mean_reward:     ' + str(mean_reward) +
                                    'mean_length:     ' + str(mean_length) +
                                    'mean_value:     ' + str(mean_value) +
                                    'mean_reward_long:     ' + str(mean_reward_long) + "\n" +
                                    'Losses/Value Loss:     ' + str(v_l) +
                                    'Losses/Policy Loss:     ' + str(p_l) +
                                    'Losses/Entropy:     ' + str(e_l) +
                                    'Losses/Grad Norm:     ' + str(g_n) +
                                    'Losses/Var Norm:     ' + str(v_n) + "\n")
                    text_file.close()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)

                    merged = tf.summary.merge_all()
                    m = sess.run(merged)
                    test_writer.add_summary(m)

                episode_count += 1


##############################################
state_size_depth = 7056
state_size_map = 40000
state_size_goal = 6


a_size = 5 # Agent can move Left, Right, or Fire


max_episode_length = 200
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
model_path = './model'
adam_learning_rate=0.0001
buffer_rollout = 30
tf.reset_default_graph()

text_file = open("a3c_results.txt", "w")
text_file.write("start training" + "\n"
                "max_episode_length" + str(max_episode_length) + "\n" +
                "gamma" + str(gamma) + "\n" +
                "adam_learning_rate" + str(adam_learning_rate) + "\n" +
                "max_episode_length" + str(max_episode_length) + "\n" )

text_file.close()


if not os.path.exists(model_path):
    os.makedirs(model_path)


with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(adam_learning_rate)

    master_network = Global_Network(state_size_depth, state_size_map, state_size_goal, a_size, 'global', None, stack_size)  # Generate global network

    num_workers_possible = multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    num_workers = 0
    for i in range(num_workers_possible):
        if(rospy.has_param("/GETjag" +str(i) + "/worker_ready")):
            if(rospy.get_param("/GETjag" + str(i) + "/worker_ready")):
                num_workers+=1
                print("worker_",num_workers)
    workers = []
    # Create worker classes
    for i in range(num_workers):
        print("worker_", i)
        workers.append(Worker( i, state_size_depth, state_size_map, state_size_goal, a_size, trainer, model_path, global_episodes,stack_size))

    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        #train_writer = tf.summary.FileWriter("train_" + str(0), sess.graph)
        #test_writer = tf.summary.FileWriter("train_" + str(0))
        sess.run(tf.global_variables_initializer())


    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        print("worker_")
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver,buffer_rollout)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
