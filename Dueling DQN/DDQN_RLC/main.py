# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:47:03 2019

@author: Administrator
"""

import DDQN
import env_SWMM


if __name__=='__main__':
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    
    env=env_SWMM.env_SWMM(date_time, date_t)
    #observation=env.reset()
    #print(env.action_space)
    '''
    print(observation)
    for t in range(len(date_t)-3):
        r=env.step([0.3,0.7])
        print(env.iten)
        print(r)
    '''
    # Superparameters
    MEMORY_SIZE = 15000
    ACTION_SPACE = 1
    step=200
    
    natural_DQN = DDQN.DDQN(
            n_actions=ACTION_SPACE, step=step, n_features=5, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=False, output_graph=True,env=env)
    _,_,history=natural_DQN.train()
    natural_DQN.save_history(history,'DQN.csv')
    '''
    dueling_DQN = DDQN.DDQN(
            n_actions=ACTION_SPACE, step=step, n_features=5, memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001, dueling=True, output_graph=True,env=env)
    dueling_DQN.train()
    '''
    #train完对三场降雨进行测试    #是四场降雨吧？？？
    test_num=4
    #r1=dueling_DQN.test(test_num)
    r2=natural_DQN.test(test_num)
    
    #print(r2)
    