#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:38:44 2018

@author: hgong
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import brain
import kernel
import matplotlib.pyplot as plt


'''
    Env Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

'''

FEATURE_NAMES = ['0Cart Position', '1Cart Velocity', '2Pole Angle', '3Pole Velocity At Tip']

env = gym.make('CartPole-v0')
env.reset()

tf.reset_default_graph()
sess = tf.Session()

worker = brain.WorkerDQN(env = env, worker_name = 'CartPole_DQN')

worker.work(sess, plot_learning_curve=True)

worker.test(sess)











