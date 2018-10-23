#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:57:02 2018

@author: hgong
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt


def forward_net(inpt):
    out = inpt

    for hidden in [32, 64]:
        out = layers.fully_connected(out, 
                                     num_outputs = hidden,
                                     activation_fn = tf.nn.relu)
        # out = tf.nn.dropout(out, keep_prob = 0.7)
    return out


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
      tf.abs(x) < delta,  # condition
      tf.square(x) * 0.5,  # if satisfied use this
      delta * (tf.abs(x) - 0.5 * delta)  # else use this
    )
    
    
def plot_learning_curve(returns, freq):    
    if len(returns) == 0:
        return
    
    _y = np.reshape(np.array(returns), (-1, freq))
    _y = _y.mean(axis = 1)
    
    _x = (np.array(range(len(_y))) + 1) * freq
    
    plt.plot(_x, _y)    
    plt.show()

        
def gaussian_kernel(S, T):
    ''' Compute the gaussian kernel matrix between S and T.
        S, T.shape = (n_samples, n_features)
        K_ij = distance(S[i,:], T[j,:])
    '''
    if S.shape != T.shape:
        raise ValueError
    
    n, p = S.shape
    
    temp = (S * S).sum(axis = 1)
    Si_Si_t = np.repeat(temp, n).reshape(-1,n)
    
    temp = (T * T).sum(axis = 1)
    Tj_Tj_t = np.repeat(temp, n).reshape(-1,n).T
    
    Si_Tj_t = S.dot(T.T)
    
    exp_term = Si_Si_t - 2*Si_Tj_t + Tj_Tj_t
    
    K = 1/np.power(2*np.pi, p/2) * np.exp(-0.5 * exp_term)
    
    return K
    
def discrete_kernel(S, T, grid_shape = [10, 10, 10, 10]):
    ''' K_ij = 1 if |S[i,:] - T[j,:]| < range(S) / grid_shape
    '''    
    if S.shape != T.shape or S.shape[1] != len(grid_shape):
        raise ValueError
    
    thrshhlds = (S.max(axis=0) - S.min(axis=0)) / np.array(grid_shape)
    
    n, p = S.shape
    
    K = []
    for i in range(n):
        temp = S[i] - T <= thrshhlds
        K.append(np.prod(temp, axis=1))
        
    return np.array(K)    

      


    
    
    