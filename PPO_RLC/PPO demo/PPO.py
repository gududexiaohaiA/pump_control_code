# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:30:53 2019

@author: chong

PPO algorithm
"""

import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf


class PPO:
    def __init__(self, ep, batch, t='ppo2'):
        self.t = t
        self.ep = ep
        self.batch = batch
        self.log = 'model/{}_log'.format(t)

        self.env = gym.make('Pendulum-v0')
        self.bound = self.env.action_space.high[0]

        self.gamma = 0.9
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.A_UPDATE_STEPS = 10
        self.C_UPDATE_STEPS = 10

        # KL penalty, d_target、β for ppo1
        self.kl_target = 0.01
        self.lam = 0.5
        # ε for ppo2
        self.epsilon = 0.2

        self.sess = tf.Session()
        self.build_model()

    def _build_critic(self):
        """critic model.
        """
        with tf.variable_scope('critic'):
            x = tf.layers.dense(self.states, 100, tf.nn.relu)

            self.v = tf.layers.dense(x, 1)
            self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        """actor model.
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(self.states, 100, tf.nn.relu, trainable=trainable)

            mu = self.bound * tf.layers.dense(x, 1, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(x, 1, tf.nn.softplus, trainable=trainable)

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return norm_dist, params

    def build_model(self):
        """build model with ppo loss.
        """
        # inputs
        self.states = tf.placeholder(tf.float32, [None, 3], 'states')
        self.action = tf.placeholder(tf.float32, [None, 1], 'action')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.dr = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # define ppo loss
        with tf.variable_scope('loss'):
            # critic loss
            self.closs = tf.reduce_mean(tf.square(self.advantage))

            # actor loss
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
                surr = ratio * self.adv

            if self.t == 'ppo1':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(old_nd, nd)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else: 
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.- self.epsilon, 1.+ self.epsilon) * self.adv))

        # define Optimizer
        with tf.variable_scope('optimize'):
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(nd.sample(1), axis=0)

        # update old actor
        with tf.variable_scope('update_old_actor'):
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        tf.summary.FileWriter(self.log, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        """choice continuous action from normal distributions.

        Arguments:
            state: state.

        Returns:
           action.
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.states: state})[0]

        return np.clip(action, -self.bound, self.bound)

    def get_value(self, state):
        """get q value.

        Arguments:
            state: state.

        Returns:
           q_value.
        """
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.sess.run(self.v, {self.states: state})

    def discount_reward(self, states, rewards, next_observation):
        """Compute target value.

        Arguments:
            states: state in episode.
            rewards: reward in episode.
            next_observation: state of last action.

        Returns:
            targets: q targets.
        """
        s = np.vstack([states, next_observation.reshape(-1, 3)])
        q_values = self.get_value(s).flatten()

        targets = rewards + self.gamma * q_values[1:]
        targets = targets.reshape(-1, 1)

        return targets

# not work.
#    def neglogp(self, mean, std, x):
#        """Gaussian likelihood
#        """
#        return 0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
#               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#               + tf.reduce_sum(tf.log(std), axis=-1)

    def update(self, states, action, dr):
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.sess.run(self.update_old_actor)

        adv = self.sess.run(self.advantage,
                            {self.states: states,
                             self.dr: dr})

        # update actor
        if self.t == 'ppo1':
            # run ppo1 loss
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.states: states,
                     self.action: action,
                     self.adv: adv,
                     self.tflam: self.lam})

            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            # run ppo2 loss
            for _ in range(self.A_UPDATE_STEPS):
                self.sess.run(self.atrain_op,
                              {self.states: states,
                               self.action: action,
                               self.adv: adv})

        # update critic
        for _ in range(self.C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op,
                          {self.states: states,
                           self.dr: dr})

    def train(self):
        """train method.
        """
        tf.reset_default_graph()

        history = {'episode': [], 'Episode_reward': []}

        for i in range(self.ep):
            observation = self.env.reset()

            states, actions, rewards = [], [], []
            episode_reward = 0
            j = 0

            while True:
                a = self.choose_action(observation)
                next_observation, reward, done, _ = self.env.step(a)
                states.append(observation)
                actions.append(a)

                episode_reward += reward
                rewards.append((reward + 8) / 8)

                observation = next_observation

                if (j + 1) % self.batch == 0:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.discount_reward(states, rewards, next_observation)

                    self.update(states, actions, d_reward)

                    states, actions, rewards = [], [], []

                if done:
                    break
                j += 1

            history['episode'].append(i)
            history['Episode_reward'].append(episode_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(i, episode_reward))

        return history

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


if __name__ == '__main__':
    '''
    model1 = PPO(1000, 32, 'ppo1')
    history = model1.train()
    model1.save_history(history, 'ppo1.csv')
    '''
    model2 = PPO(1000, 32, 'ppo2')
    history = model2.train()
    model2.save_history(history, 'ppo2.csv')