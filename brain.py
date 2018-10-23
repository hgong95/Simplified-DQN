#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:20:10 2018

@author: hgong
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from collections import namedtuple
import utils

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self._memory) < self._capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        if batch_size <= len(self._memory):
            samples = random.sample(self._memory, batch_size)
        else:
            samples = random.sample(self._memory, len(self._memory))
        
        states      = [data.state      for data in samples]
        actions     = [data.action     for data in samples]
        next_states = [data.next_state for data in samples]
        rewards     = [data.reward     for data in samples]
        dones       = [data.done       for data in samples]
        
        return Transition(np.array(states), 
                          np.array(actions), 
                          np.array(next_states),
                          np.array(rewards),
                          np.array(dones))

    def __len__(self):
        return len(self._memory)


class DQN(object):
    def __init__(self, **kwargs):
    
        self.n_features = kwargs['n_features']
        self.n_actions = kwargs['n_actions']
        self.gamma = 0.99
        
        self._build_net()
        
        self.update_gradient_op = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.initialize_op = tf.global_variables_initializer()
        
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'eval_net')
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]
          
    def _build_net(self):    
        self.s = tf.placeholder(tf.float32, 
                                shape = [None, self.n_features],
                                name = 's')
        
        self.s_next = tf.placeholder(tf.float32,
                                     shape = [None, self.n_features],
                                     name = 's_next')
        
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        
        self.done = tf.placeholder(tf.bool, [None, ], name='done') # input Dones, boolean values
        
        
        with tf.variable_scope('eval_net'):
            out = utils.forward_net(self.s)
            self.q_eval = layers.fully_connected(out, 
                                                 num_outputs = self.n_actions,
                                                 activation_fn = None)
        
        with tf.variable_scope('target_net'):
            out = utils.forward_net(self.s_next)
            self.q_next = layers.fully_connected(out, 
                                                 num_outputs = self.n_actions,
                                                 activation_fn = None)
        
        with tf.variable_scope('q_target'):
            self.q_next_wrt_a = tf.reduce_max(self.q_next, axis=1, name='Qmax_s_next') # shape = (None, )
            mask = 1- tf.cast(self.done, tf.float32)
            self.q_target = self.r + self.gamma * self.q_next_wrt_a * mask
        
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                                 axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )        
        
        with tf.variable_scope('loss'):
            td_errors = self.q_eval_wrt_a - tf.stop_gradient(self.q_target)
            self.loss = tf.reduce_mean(tf.square(td_errors))
        

class GeneralWorker(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.name = kwargs['worker_name']
        #self.initial_learning_rate = kwargs['learning_rate']
        #self.algo_name = kwargs['algo_name']

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.local_timesteps = 0


class WorkerDQN(GeneralWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.replay_memory_capacity = 10000
        self.minibatch_size = 32
        self.epsilon_final = 0.05
        self.epsilon_start = 1
        self.n_exploration_steps = 10000
        self.target_net_update_freq = 2000
        self.plot_learning_curve_freq = 50
        
        self.exploration_eps = np.linspace(self.epsilon_start, 
                                           self.epsilon_final, 
                                           num = self.n_exploration_steps)
        
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        
        self.dqn = DQN(n_features = self.n_features, n_actions = self.n_actions)
        
        self.replay_memory = ReplayMemory(self.replay_memory_capacity)

    def reset_network(self, sess):
        sess.run(self.dqn.initialize_op)
        sess.run(self.dqn.target_replace_op)

        self.learning_step_counter = 0
        self.final_returns = []
        
    def work(self, sess, n_episodes = 2000, plot_learning_curve = False):      
        try: 
            self.final_returns
        except AttributeError:
            self.reset_network(sess)
        
        with tf.variable_scope(self.name), sess.as_default(), sess.graph.as_default():                
            for episode in range(n_episodes):
                last_state = self.env.reset()
                
                # plot the learning curve
                if plot_learning_curve:
                    if episode % self.plot_learning_curve_freq == 0:
                        utils.plot_learning_curve(self.final_returns, 
                                                  self.plot_learning_curve_freq)
                        
                for i in range(1000):
                    self.env.render()
                    
                    # get curreny epsilon
                    if self.learning_step_counter < self.n_exploration_steps:
                        epsilon = self.exploration_eps[self.learning_step_counter]
                    else:
                        epsilon = self.epsilon_final
                    
                    # choose an action according to epsilon-greedy policy
                    random_helper = np.random.rand()
                    if random_helper >= epsilon:
                        q_vals = sess.run(self.dqn.q_eval ,
                                          feed_dict = {self.dqn.s: [last_state]})
                        action = np.argmax(q_vals)
                    else:
                        action = self.env.action_space.sample()
                    
                    # get and store the transition
                    next_state, reward, done, info = self.env.step(action)
                    
                    self.replay_memory.push(last_state, action, next_state, reward, done)
                    
                    states, actions, next_states, rewards, dones = self.replay_memory.sample(self.minibatch_size)
                    
                    sess.run(self.dqn.update_gradient_op, 
                             feed_dict = {self.dqn.s      :states, 
                                          self.dqn.s_next :next_states, 
                                          self.dqn.a      :actions,
                                          self.dqn.r      :rewards,
                                          self.dqn.done   :dones})

                    self.learning_step_counter += 1
                    
                    if self.learning_step_counter % self.target_net_update_freq == 0:
                        sess.run(self.dqn.target_replace_op)
                    
                    if done:
                        self.final_returns.append(i)
                        break
                    else:
                        last_state = next_state    
                    
    def test(self, sess, n_episodes = 100):
        self.test_memory = ReplayMemory(n_episodes * 200)
        self.test_final_returns = []
        
        for episode in range(100):
            last_state = self.env.reset()
            for i in range(10000):
                self.env.render()
                
                q_vals = sess.run(self.dqn.q_eval ,feed_dict = {self.dqn.s: [last_state]})
                action = np.argmax(q_vals)
                
                next_state, reward, done, info = self.env.step(action)

                self.test_memory.push(last_state, action, next_state, reward, done)
                
                if done:
                    self.test_final_returns.append(i)
                    break
                else:
                    last_state = next_state
        
        
        
        




