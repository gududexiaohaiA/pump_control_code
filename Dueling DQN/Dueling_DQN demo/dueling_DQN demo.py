# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:29:10 2019

@author: Administrator
"""

import tensorflow as tf

class DDQN:
    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # network weights
        with tf.variable_scope('current_net'):
            W1 = self.weight_variable([self.state_dim,20])
            b1 = self.bias_variable([20])
    
            # hidden layer 1
            h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    
            # hidden layer  for state value
            with tf.variable_scope('Value'):
              W21= self.weight_variable([20,1])
              b21 = self.bias_variable([1])
              self.V = tf.matmul(h_layer_1, W21) + b21
    
            # hidden layer  for action value
            with tf.variable_scope('Advantage'):
              W22 = self.weight_variable([20,self.action_dim])
              b22 = self.bias_variable([self.action_dim])
              self.A = tf.matmul(h_layer_1, W22) + b22
    
              # Q Value layer
              self.Q_value = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
    
        with tf.variable_scope('target_net'):
            W1t = self.weight_variable([self.state_dim,20])
            b1t = self.bias_variable([20])
    
            # hidden layer 1
            h_layer_1t = tf.nn.relu(tf.matmul(self.state_input,W1t) + b1t)
    
            # hidden layer  for state value
            with tf.variable_scope('Value'):
              W2v = self.weight_variable([20,1])
              b2v = self.bias_variable([1])
              self.VT = tf.matmul(h_layer_1t, W2v) + b2v
    
            # hidden layer  for action value
            with tf.variable_scope('Advantage'):
              W2a = self.weight_variable([20,self.action_dim])
              b2a = self.bias_variable([self.action_dim])
              self.AT = tf.matmul(h_layer_1t, W2a) + b2a
    
              # Q Value layer
              self.target_Q_value = self.VT + (self.AT - tf.reduce_mean(self.AT, axis=1, keep_dims=True))
              
              
if __name__=='__main__':
    a=DDQN()
    a.create_Q_network()