# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:45:18 2019

@author: Administrator

The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import os
import pandas as pd


import change_rain
import set_datetime

np.random.seed(1)
tf.set_random_seed(1)


class DDQN:
    def __init__(self,n_actions,n_features,step=200,learning_rate=0.001,reward_decay=0.9,e_greedy=0.9,replace_target_iter=10,memory_size=50,batch_size=10,e_greedy_increment=None,output_graph=False,dueling=True,env=gym.make('Pendulum-v0').unwrapped):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.traing_step=step

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        #self.memory = np.zeros((self.memory_size, n_features*2+2))
        self.memory=[]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        self.env=env
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1',reuse=tf.AUTO_REUSE):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value',reuse=tf.AUTO_REUSE):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage',reuse=tf.AUTO_REUSE):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q',reuse=tf.AUTO_REUSE):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q',reuse=tf.AUTO_REUSE):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net',reuse=tf.AUTO_REUSE):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net',reuse=tf.AUTO_REUSE):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r], s_))
        '''
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        #print('s=',s)
        #print('[a,r=]',a,r)
        #print('s_=',s_)
        '''
        
        self.memory.append(transition)
        self.memory_counter += 1
        

    def choose_action(self, observation):
        #observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self,total_step):
        '''
        if self.learn_step_counter >=total_step/10:#% self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        '''
        self.sess.run(self.replace_target_op)
        print('\ntarget_params_replaced\n')
        
        #print(self.learn_step_counter % self.replace_target_iter)
        sample_index = np.random.choice(total_step, size=self.batch_size)
        batch_memory=[]
        #print(self.memory)
        #print(batch_memory[:, -self.n_features:])
        for i in sample_index:
            #print(self.memory[int(i)])
            batch_memory.append(list(self.memory[int(i)]))
        #batch_memory=list(batch_memory)
        batch_memory=np.array(batch_memory)
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:,-self.n_features:]}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:,:self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        #print(q_target[[1,2]])

        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target[batch_index] = np.reshape(reward + self.gamma * np.max(q_next, axis=1),(self.batch_size,1))
        
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
    def train(self):
        
        history = {'episode': [], 'Episode_reward': []}
        for i in range(self.traing_step):
            acc_r = [0]
            total_steps = 0
            print(self.rainData[i])
            observation = self.env.reset(self.rainData[i])
            
            episode_reward=0
            
            while True:
                # if total_steps-MEMORY_SIZE > 9000: env.render()
                #print('ob=',observation)
                action = self.choose_action(observation)#action为概率
                #print('a=',action)
                f_action = (action-(self.n_actions-1)/2)/((self.n_actions)/1)   # [-2 ~ 2] float actions
                print(f_action)
                observation_, reward, done, info = self.env.step(np.array([f_action]),self.rainData[i])
                #print(observation_, reward, done, info)
                reward /= 10      # normalize to a range of (-1, 0)
                
                episode_reward = reward-10*np.exp(-self.gamma*i)#为什么，为了画训练步数收敛图
                
                acc_r.append(reward + acc_r[-1])  # accumulated reward 这里reward是指每一步的reward，acc_r是每场降雨叠加的
        
                self.store_transition(observation, action, reward, observation_)
     
                observation = observation_
                total_steps += 1
                
                #print(total_steps%self.env.T)
                if total_steps%self.env.T>=self.env.T-3:
                    print('start learning')
                    
                    self.learn(total_steps)
        
                if done:
                    break
    
                if total_steps-self.memory_size > 15000:
                    break
            
            history['episode'].append(i)
            history['Episode_reward'].append(episode_reward)
            print('Episode: {} | Episode reward: {:.2f}'.format(i, episode_reward))
 
        saver=tf.train.Saver()   
        sp=saver.save(self.sess,"./save/model.ckpt")
        print("model saved:",sp)
        
        return self.cost_his, acc_r,history
    
    def test(self,test_num):
        """train method.
        """
        saver=tf.train.Saver()
        saver.restore(self.sess,"./save/model.ckpt")

        dr=[]
        for i in range(test_num):
            acc_r = [0]
            
            observation = self.env.reset(self.testRainData[i])
            #print('obtest=',observation)
            while True:
                # if total_steps-MEMORY_SIZE > 9000: env.render()
        
                action = self.choose_action(observation)
        
                f_action = (action-(self.n_actions-1)/2)/((self.n_actions)/4)   # [-2 ~ 2] float actions
                observation_, reward, done, info = self.env.step(np.array([f_action]),self.testRainData[i])
                #print(observation_, reward, done, info)
                reward /= 10      # normalize to a range of (-1, 0)
                acc_r.append(reward + acc_r[-1])  # accumulated reward
        
                #self.store_transition(observation, action, reward, observation_)
     
                observation = observation_
                #print('obtest=',observation)
                if done:
                    break
                    dr.append(acc_r)
                
            #对比HC
            change_rain.copy_result('./test_result/HC/compare_tem_HC'+str(i)+'.inp',self.env.orf_rain+'.inp')#还原
            tem_etime=self.env.date_time[self.env.iten]
            set_datetime.set_date(self.env.sdate,self.env.edate,self.env.stime,tem_etime,'./test_result/HC/compare_tem_HC'+str(i)+'.inp')
            self.env.simulation('./test_result/HC/compare_tem_HC'+str(i)+'.inp')
                        

            #history['episode'].append(i)
            #history['Episode_reward'].append(episode_reward)
            #print('Episode: {} | Episode reward: {:.2f}'.format(i, episode_reward))
            sout='./test_result/DDQN/DDQN_'+str(i)+'.rpt'
            sin=self.env.staf+'.rpt'
            change_rain.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)

        return dr
    
    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
    
    
if __name__=='__main__':
    
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    #env = gym.make('CartPole-v0')
    env.seed(1)
    MEMORY_SIZE = 3000
    ACTION_SPACE = 25
    
    #sess = tf.Session()
    with tf.variable_scope('natural'):
        natural_DQN = DDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=False)
    
    with tf.variable_scope('dueling'):
        dueling_DQN = DDQN(
            n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=True, output_graph=True)
    
    #sess.run(tf.global_variables_initializer())
    
    c_natural, r_natural = natural_DQN.train()
    c_dueling, r_dueling = dueling_DQN.train()
    
    plt.figure(1)
    plt.plot(np.array(c_natural), c='r', label='natural')
    plt.plot(np.array(c_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('cost')
    plt.xlabel('training steps')
    plt.grid()
    
    plt.figure(2)
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()
    
    plt.show()