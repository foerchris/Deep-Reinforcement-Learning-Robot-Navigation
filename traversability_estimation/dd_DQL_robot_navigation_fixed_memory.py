#!/usr/bin/env python2
# -*- coding: utf-8

from robotEnvWrapper_local import robotEnv
import tensorflow as tf
import numpy as np           # Handle matrices

import random                # Handling random number generation
import tensorflow.contrib.slim as slim

from collections import deque# Ordered collection with ends

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

from time import  sleep

def all_actions():

    # Here our possible actions
    hard_left = [1, 0, 0, 0, 0]
    left = [0, 1, 0, 0, 0]
    forward = [0, 0, 1, 0, 0]
    right = [0, 0, 0, 1, 0]
    hard_right = [0, 0, 0, 0, 1]
    possible_actions = [hard_left, left, forward, right, hard_right]

    return  possible_actions

robotEnv = robotEnv("1")

possible_actions = all_actions()


# def preprocess_frame(frame):
#     # Greyscale frame already done in our vizdoom config
#     # x = np.mean(frame,-1)
#
#     # Crop the screen (remove the roof because it contains no information)
#     cropped_frame = frame[30:-10, 30:-30]
#
#     # Normalize Pixel Values
#     normalized_frame = cropped_frame / 255.0
#
#     # Resize
#     preprocessed_frame = transform.resize(normalized_frame, [84, 84])
#
#     return preprocessed_frame
# Since python version < 3.5 need to be faltend

def preprocess_state(state, state_size):
    return np.reshape(state,[state_size])


stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_depth_frames = deque([np.zeros((7056), dtype=np.int) for i in range(stack_size)], maxlen=4)
stacked_map_frames = deque([np.zeros((40000), dtype=np.int) for i in range(stack_size)], maxlen=4)
stacked_orientation_frames = deque([np.zeros((3), dtype=np.int) for i in range(stack_size)], maxlen=4)
stacked_goal_frames = deque([np.zeros((6), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, state_size , is_new_episode):
    # Preprocess frame
    # because of different functions this as to be done beforhand
    #frame = preprocess_frame(state)
    state = preprocess_state(state,state_size)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((state_size), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)
        stacked_frames.append(state)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(state)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)

    stacked_state = preprocess_state(stacked_state,state_size*stack_size)
    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
###state_size = [100,120,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)

state_size_depth = 7056
state_size_elev = 40000
state_size_goal = 6

state_size_depth_stack = state_size_depth*stack_size
state_size_elev_stack = state_size_elev*stack_size
state_size_goal_stack = state_size_goal*stack_size

h_size = 1152

action_size = 5              # 5 possible actions: hard left, left, forward, right, hard_right
learning_rate =  0.001      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000        # Total episodes for training
max_steps = 200
batch_size = 40

lstm_state_size = 576


# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 5000 #Tau is the C step where we update our target network

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00005            # exponential decay rate for exploration prob

# Q learning hyperparameters

gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = 10000   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


class DDDQNNet:
    def __init__(self,  state_size_depth , state_size_elev, state_size_goal, action_size, h_size,learning_rate, name):
        self.state_size_depth = state_size_depth
        self.state_size_elev = state_size_elev
        #self.state_size_orientation = state_size_orientation
        self.action_size = action_size
        self.h_size = h_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [7056]
            self.depthImages = tf.placeholder(tf.float32, [None, state_size_depth], name="depthImages")
            # [84, 84, 1]
            self.imageIn = tf.reshape(self.depthImages, shape=[-1, 84, 84, 4])
            # [40000]
            self.eleviationMaps = tf.placeholder(tf.float32, [None, state_size_elev], name="eleviationMaps")
            # [200, 200, 1]
            self.mapIn = tf.reshape(self.eleviationMaps, shape=[-1, 200, 200, 4])
            # [3]
            #self.orientations = tf.placeholder(tf.float32, [None, state_size_orientation], name="orientations")
            # [3, 1]
            #self.orientationIn = tf.reshape(self.orientations, shape=[-1, 3, 4])

            # [6]
            self.goal = tf.placeholder(tf.float32, [None, state_size_goal], name="goal")
            # [6, 4]
            self.goalIn = tf.reshape(self.goal, shape=[-1, 6, 4])
            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            ## This is the depthImage part
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='VALID')

            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.pool1, num_outputs=32,
                                     kernel_size=[5, 5], stride=[1, 1], padding='VALID')

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.pool2, num_outputs=64,
                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=2)
            self.conv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.pool3, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='VALID')

            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], strides=2)
            hidden1 = slim.fully_connected(slim.flatten(self.pool4), 576, activation_fn=tf.nn.elu)

            # This is the ElevMap part
            self.convMap1 = slim.conv2d(activation_fn=tf.nn.elu,
                                        inputs=self.mapIn, num_outputs=32,
                                        kernel_size=[7, 7], stride=[1, 1], padding='VALID')

            self.poolMap1 = tf.layers.max_pooling2d(inputs=self.convMap1, pool_size=[2, 2], strides=2)
            self.convMap2 = slim.conv2d(activation_fn=tf.nn.elu,
                                        inputs=self.poolMap1, num_outputs=32,
                                        kernel_size=[6, 6], stride=[1, 1], padding='VALID')
            self.poolMap2 = tf.layers.max_pooling2d(inputs=self.convMap2, pool_size=[2, 2], strides=2)
            self.convMap3 = slim.conv2d(activation_fn=tf.nn.elu,
                                        inputs=self.poolMap2, num_outputs=64,
                                        kernel_size=[5, 5], stride=[1, 1], padding='VALID')
            self.poolMap3 = tf.layers.max_pooling2d(inputs=self.convMap3, pool_size=[2, 2], strides=2)
            self.convMap4 = slim.conv2d(activation_fn=tf.nn.elu,
                                        inputs=self.poolMap3, num_outputs=64,
                                        kernel_size=[4, 4], stride=[1, 1], padding='VALID')
            self.poolMap4 = tf.layers.max_pooling2d(inputs=self.convMap4, pool_size=[2, 2], strides=2)

            # This is the orientation part
            #self.fconOrientation = slim.fully_connected(slim.flatten(self.orientationIn), 3, activation_fn=tf.nn.elu)
            #self.reshapedOriantation = tf.reshape(self.fconOrientation, shape=[9, 9, 64])

            # This is the goal part
            self.fconGoal = slim.fully_connected(slim.flatten(self.goalIn), 6, activation_fn=tf.nn.elu)
            self.reshapedGoal = tf.reshape(self.fconGoal, shape=[9, 9, 64])

            self.addGoal = tf.add(
                self.poolMap4,
                self.reshapedGoal,
                name=None)


            self.convMapAndOrientation = slim.conv2d(activation_fn=tf.nn.elu,
                                                     inputs=self.poolMap3, num_outputs=64,
                                                     kernel_size=[4, 4], stride=[1, 1], padding='VALID')

            self.poolMapAndOrientation = tf.layers.max_pooling2d(inputs=self.convMapAndOrientation, pool_size=[2, 2],
                                                                 strides=2)

            hidden2 = slim.fully_connected(slim.flatten(self.poolMapAndOrientation), 576, activation_fn=tf.nn.elu)

            self.concateLayer = tf.concat([hidden1, hidden2],0)


            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=576, state_is_tuple=True)
            # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            # self.state_init = [c_init, h_init]
            #
            # c_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.c])
            # h_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.h])
            # self.state_in = (c_in, h_in)
            #
            # state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            #
            # step_size = tf.shape(self.imageIn)[:1]

            rnn_in = tf.expand_dims(self.concateLayer, [0])
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, dtype=tf.float32)

            # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            #      lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
            #      time_major=False)


            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 1152])

            #self.state_out = self.state_in
            #rnn_out = tf.reshape(self.concateLayer, [-1, 1152])

            # self.conv4 = slim.conv2d( \
            #     inputs=self.poolMapAndOrientation, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1],
            #     padding='VALID',
            #     biases_initializer=None)
            # self.final = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], strides=2)

            self.flatten = tf.layers.flatten(rnn_out)
            ## Here we separate into two streams
            # The one that calculate V(s)

            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree


            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size_depth_stack, state_size_elev_stack, state_size_goal_stack, action_size, h_size, learning_rate, name="DQNetwork")

TargetNetwork = DDDQNNet(state_size_depth_stack, state_size_elev_stack, state_size_goal_stack, action_size,h_size, learning_rate, name="TargetNetwork")


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.7  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

memory = Memory(memory_size)

robotEnv.set_episode_length(max_steps)

print("pre Taining")
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # Start a new episode and get state
        robotEnv.reset()
        depth_state, map_state, goal_state = robotEnv.getState()

        depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, state_size_depth, True)
        map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, state_size_elev, True)
        goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, state_size_goal, True)

    # Random action
    action = random.choice(possible_actions)
    #print("action", action)
    # Get the rewards, Look if the episode is finished
    reward, done  = robotEnv.takeStep(action)
    # If we're dead
    if done:
        # We finished the episode
        next_depth_state = np.zeros(state_size_depth)
        next_map_state = np.zeros(state_size_elev)
        next_goal_state = np.zeros(state_size_goal)

        next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, next_depth_state, state_size_depth,
                                                              False)
        next_map_state, stacked_map_frames = stack_frames(stacked_map_frames, next_map_state, state_size_elev, False)
        next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, next_goal_state, state_size_goal,
                                                            False)
        # Add experience to memory
        experience = depth_state, map_state, goal_state,\
                     action, reward, next_depth_state, next_map_state, next_goal_state, \
                     done
        memory.store(experience)

        # Start a new episode and get state
        robotEnv.reset()
        depth_state, map_state, goal_state = robotEnv.getState()

        depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, state_size_depth, True)
        map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, state_size_elev, True)
        goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, state_size_goal, True)

    else:
        # Get the next state
        next_depth_state, next_map_state, next_goal_state = robotEnv.getState()


        next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, next_depth_state, state_size_depth, False)
        next_map_state, stacked_map_frames = stack_frames(stacked_map_frames, next_map_state, state_size_elev, False)
        next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, next_goal_state, state_size_goal, False)


        # Add experience to memory
        experience = depth_state, map_state, goal_state,\
                     action, reward, next_depth_state, next_map_state, next_goal_state,\
                     done
        memory.store(experience)

        # Our state is now the next_state
        sleep(0.24)
        depth_state = next_depth_state
        map_state = next_map_state
        goal_state = next_goal_state



    if i % 100 == 0:
        print("pre Training Step" + str(i))
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""


def predict_action(explore_start, explore_stop, decay_rate, decay_step, depth_state, map_state,
                   goal_state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()


    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)




    # Get action from Q-network (exploitation)
    # Estimate the Qs values state
    feed_dict={DQNetwork.depthImages: [depth_state],
                DQNetwork.eleviationMaps: [map_state],
                DQNetwork.goal: [goal_state]}

    Qs = sess.run([DQNetwork.output], feed_dict=feed_dict)

    # Take the biggest Q value (= the best action)
    choice = np.argmax(Qs)
    action = possible_actions[int(choice)]

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    return action, explore_probability


# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Saver will help us to save our model
saver = tf.train.Saver()

print("Training")

text_file = open("results.txt", "w")
text_file.write("start training" + "\n")
text_file.close()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(total_episodes):

            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state and faltten them
            robotEnv.reset()
            depth_state, map_state, goal_state = robotEnv.getState()

            # Remember that stack frame function also call our preprocess function.
            depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, state_size_depth, True)
            map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, state_size_elev, True)
            goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, state_size_goal, True)

            while step < max_steps:
                step += 1

                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                    depth_state, map_state, goal_state,
                                                                    possible_actions)
                # Get the rewards, Look if the episode is finished
                reward, done = robotEnv.takeStep(action)

                # Add the reward to total reward
                episode_rewards.append(reward)
                total_reward = np.sum(episode_rewards)
                print('Total reward: {}'.format(np.sum(episode_rewards)))
                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_depth_state = np.zeros((state_size_depth), dtype=np.int)
                    next_map_state = np.zeros((state_size_elev), dtype=np.int)
                    next_goal_state = np.zeros((state_size_goal), dtype=np.int)

                    next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, next_depth_state, state_size_depth, False)
                    next_map_state, stacked_map_frames = stack_frames(stacked_map_frames, next_map_state, state_size_elev, False)
                    next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, next_goal_state, state_size_goal, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    text_file = open("results.txt", "a")
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(np.sum(episode_rewards)),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                    text_file.write('Episode: ' + str(episode) +
                          ', Total reward: ' + str(total_reward) +
                          ', Training loss: ' + str(loss) +
                          ', Explore P: ' + str(explore_probability) + "\n")
                    text_file.close()

                    # Add experience to memory
                    experience = depth_state, map_state, goal_state,\
                                 action, reward, next_depth_state, next_map_state, next_goal_state,\
                                 done

                    memory.store(experience)


                else:
                    # Get the next state
                    next_depth_state, next_map_state, next_goal_state= robotEnv.getState()

                    # Stack the frame of the next_state
                    next_depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, next_depth_state, state_size_depth, False)
                    next_map_state, stacked_map_frames = stack_frames(stacked_map_frames, next_map_state, state_size_elev, False)
                    next_goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, next_goal_state, state_size_goal, False)

                    experience = depth_state, map_state, goal_state,\
                                 action, reward, next_depth_state, next_map_state, next_goal_state,\
                                 done
                    memory.store(experience)

                    # st+1 is now our current state
                    depth_state = next_depth_state
                    map_state = next_map_state
                    goal_state = next_goal_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                depth_states_mb = np.array([each[0][0] for each in batch], ndmin=1)
                map_states_mb = np.array([each[0][1] for each in batch], ndmin=1)
                goal_states_mb = np.array([each[0][2] for each in batch], ndmin=1)
                actions_mb = np.array([each[0][3] for each in batch])
                rewards_mb = np.array([each[0][4] for each in batch])
                next_depth_states_mb = np.array([each[0][5] for each in batch], ndmin=1)
                next_map_states_mb = np.array([each[0][6] for each in batch], ndmin=1)
                next_goal_states_mb = np.array([each[0][7] for each in batch], ndmin=1)
                dones_mb = np.array([each[0][8] for each in batch])




                target_Qs_batch = []

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                # Get Q values for next_state


                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.depthImages: next_depth_states_mb,
                                                                     DQNetwork.eleviationMaps: next_map_states_mb,
                                                                     DQNetwork.goal: next_goal_states_mb})

                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.depthImages: next_depth_states_mb,
                                                                                TargetNetwork.eleviationMaps: next_map_states_mb,
                                                                                TargetNetwork.goal: next_goal_states_mb})


                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])

                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.depthImages: depth_states_mb,
                                                               DQNetwork.eleviationMaps: map_states_mb,
                                                               DQNetwork.goal: goal_states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb,
                                                               DQNetwork.ISWeights_: ISWeights_mb})

                # Update priority

                memory.batch_update(tree_idx, absolute_errors)

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.depthImages: depth_states_mb,
                                                        DQNetwork.eleviationMaps: map_states_mb,
                                                        DQNetwork.goal: goal_states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb,
                                                        DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")
            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
        print("endEnv")
        robotEnv.endEnv()



else:
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize the variables
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, './models/model.ckpt')

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 5000

        # Set tau = 0
        tau = 50000

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        while (1):
            robotEnv.reset()
            depth_state, map_state, goal_state = robotEnv.getState()

            # Remember that stack frame function also call our preprocess function.
            depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, state_size_depth, True)
            map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, state_size_elev, True)
           # orientation_state, stacked_orientation_frames = stack_frames(stacked_orientation_frames, orientation_state,
           #                                                              state_size_orientation, True)
            goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, state_size_goal, True)

            done  = False
            rnn_state = DQNetwork.state_init
            episode_rewards = []

            while(done==False):

                # Predict the action to take and take it
                action, explore_probability, rnn_state = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                       depth_state, map_state, goal_state,
                                                                       rnn_state, possible_actions)
                # Get the rewards, Look if the episode is finished
                reward, done = robotEnv.takeStep(action)
                # Add the reward to total reward
                episode_rewards.append(reward)

                depth_state, map_state, goal_state = robotEnv.getState()

                # Remember that stack frame function also call our preprocess function.
                depth_state, stacked_depth_frames = stack_frames(stacked_depth_frames, depth_state, state_size_depth, True)
                map_state, stacked_map_frames = stack_frames(stacked_map_frames, map_state, state_size_elev, True)
                # orientation_state, stacked_orientation_frames = stack_frames(stacked_orientation_frames, orientation_state,
                #                                                              state_size_orientation, True)
                goal_state, stacked_goal_frames = stack_frames(stacked_goal_frames, goal_state, state_size_goal, True)
            np.sum(episode_rewards)
            print('Total reward: {}'.format(np.sum(episode_rewards)))
