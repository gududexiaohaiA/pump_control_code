# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:43:29 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
#import gym
import os
import pandas as pd

import set_datetime
import change_rain


class Actor(object):                    #Policy net
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")   #state input
        self.a = tf.placeholder(tf.int32, None, "act")                  #action input
        self.td_error = tf.placeholder(tf.float32, None, "td_error")    # TD_error

        with tf.variable_scope('Actor',reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1',
                #reuse=tf.AUTO_REUSE
            )                   #fully connected layer 1

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob',
                #reuse=tf.AUTO_REUSE
            )               #output softmax

        with tf.variable_scope('exp_v',reuse=tf.AUTO_REUSE):
            log_prob = tf.log(self.acts_prob[0, self.a])        #selecte the action prob value
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
            

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}  #td temproal difference 由 critic net产生
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)  #优化
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int 以某种概率选择动作


class Critic(object):
    def __init__(self, sess, n_features, GAMMA, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")   #critic net input 当前状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")         #下一状态对应的value 值
        self.r = tf.placeholder(tf.float32, None, 'r')                  #当前状态执行动作后的奖励值
        self.GAMMA=GAMMA
        with tf.variable_scope('Critic',reuse=tf.AUTO_REUSE):                           #构建critic 网络，注意输出一个值表示当前value
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error',reuse=tf.AUTO_REUSE):
            self.td_error = self.r + GAMMA * self.v_ - self.v   #贝尔慢迭代公式 求TD-error
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):      #critic net的学习算法
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})    #下一状态输入到网络获取下一状态的q值
        td_error, _ = self.sess.run([self.td_error, self.train_op],    
                                          {self.s: s, self.v_: v_, self.r: r})  #当前状态下输入获取当前状态q值
        return td_error




class A2C(object):
    
    def __init__(self,OUTPUT_GRAPH,MAX_EPISODE,MAX_EP_STEPS,GAMMA,LR_A,LR_C,env):
        np.random.seed(2)
        tf.set_random_seed(2)  # reproducible
        
        '''
        # Superparameters
        OUTPUT_GRAPH = False # 是否保存模型（网络结构）
        MAX_EPISODE = 3000
        DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
        MAX_EP_STEPS = 100000   # maximum time step in one episode
        RENDER = True  # rendering wastes time
        GAMMA = 0.9     # reward discount in TD error
        LR_A = 0.1    # learning rate for actor
        LR_C = 0.1     # learning rate for critic
        
        env = gym.make('MountainCar-v0')
        env.seed(1)  # reproducible
        env = env.unwrapped
        
        N_F = env.observation_space.shape[0]   #状态空间的维度
        N_A = env.action_space.n                #动作空间的维度
        '''
        # Superparameters
        self.OUTPUT_GRAPH = OUTPUT_GRAPH # 是否保存模型（网络结构）
        self.MAX_EPISODE = MAX_EPISODE
        #self.DISPLAY_REWARD_THRESHOLD = DISPLAY_REWARD_THRESHOLD  # renders environment if total episode reward is greater then this threshold
        self.MAX_EP_STEPS = MAX_EP_STEPS   # maximum time step in one episode
        self.GAMMA = GAMMA     # reward discount in TD error
        self.LR_A = LR_A    # learning rate for actor
        self.LR_C = LR_C     # learning rate for critic
        
        self.env = env#gym.make('MountainCar-v0')
        #env.seed(1)  # reproducible
        #env = env.unwrapped
        
        self.N_F = env.observation_space#.shape[0]   #状态空间的维度
        self.N_A = 1#env.action_space[1]                #动作空间的维度
        
        self.sess = tf.Session()
        #tf.reset_default_graph() 
        self.actor = Actor(self.sess, n_features=self.N_F, n_actions=self.N_A, lr=self.LR_A)
        self.critic = Critic(self.sess, n_features=self.N_F, GAMMA=self.GAMMA, lr=self.LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        
        history = {'episode': [], 'Episode_reward': []}
        
        if self.OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)
        
        for i in range(self.MAX_EPISODE):
            s =self.env.reset(self.rainData[i])
            t = 0
            track_r = []
            #episode_reward=0
            
            while True:
                a = self.actor.choose_action(s)
        
                s_, r, done, info = self.env.step(a,self.rainData[i])
        
                #if done: r = -20
                #episode_reward = r-10*np.exp(-0.09*i)
                
                track_r.append(r)
        
                td_error = self.critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                self.actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
        
                s = s_
                t += 1
        
                if done or t >= self.MAX_EP_STEPS:
                    ep_rs_sum = sum(track_r)
        
                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                    #if running_reward > self.DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                    #episode_reward =running_reward
                    #running_reward+= r-np.exp(-0.9*i)
                    #print("episode:", i, "  reward:", int(running_reward))
                    break
            
            history['episode'].append(i)
            history['Episode_reward'].append(running_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(i, running_reward))
        #保存模型    
        saver=tf.train.Saver()   
        sp=saver.save(self.sess,"./save/model.ckpt")
        print("model saved:",sp)
            
        return history


    def test(self,test_num):
        """train method.
        """
        #加载模型
        saver=tf.train.Saver()
        saver.restore(self.sess,"./save/model.ckpt")
        
        dr=[]
        for i in range(test_num):
            print('test'+str(i))
            s =self.env.reset(self.testRainData[i])
            t = 0
            track_r = []
            while True:
                a = self.actor.choose_action(s)
        
                s_, r, done, info = self.env.step(a,self.testRainData[i])
        
                #if done: r = -20
        
                track_r.append(r)
        
                td_error = self.critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
                self.actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
        
                s = s_
                t += 1
                if done:
                    dr.append(track_r)
                    break
                
            #对比HC
            change_rain.copy_result('./test_result/HC/compare_tem_HC'+str(i)+'.inp',self.env.orf_rain+'.inp')#还原
            tem_etime=self.env.date_time[self.env.iten]
            set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./test_result/HC/compare_tem_HC'+str(i)+'.inp')
            self.env.simulation('./test_result/HC/compare_tem_HC'+str(i)+'.inp')

            #history['episode'].append(i)
            #history['Episode_reward'].append(episode_reward)
            #print('Episode: {} | Episode reward: {:.2f}'.format(i, episode_reward))
            sout='./test_result/'+str(i)+'.rpt'
            sin=self.env.staf+'.rpt'
            change_rain.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)

        return dr
    
    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
            
            

if __name__=='__main__':
    # Superparameters
    OUTPUT_GRAPH = False # 是否保存模型（网络结构）
    MAX_EPISODE = 100
    DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 100000   # maximum time step in one episode
    RENDER = True  # rendering wastes time
    GAMMA = 0.9     # reward discount in TD error
    LR_A = 0.1    # learning rate for actor
    LR_C = 0.1     # learning rate for critic
    
    env = gym.make('MountainCar-v0')
    model=A2C(OUTPUT_GRAPH,MAX_EPISODE,MAX_EP_STEPS,GAMMA,LR_A,LR_C,env)
    model.train()
