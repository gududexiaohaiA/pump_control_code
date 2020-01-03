# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:02:42 2019

@author: chong
"""

import A2C
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
    OUTPUT_GRAPH = False # 是否保存模型（网络结构）
    MAX_EPISODE = 200#200场降雨
    DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 24   # maximum time step in one episode
    RENDER = True  # rendering wastes time
    GAMMA = 0.9     # reward discount in TD error
    LR_A = 0.1    # learning rate for actor
    LR_C = 0.1     # learning rate for critic
    
    #env = gym.make('MountainCar-v0')
    model=A2C.A2C(OUTPUT_GRAPH,MAX_EPISODE,MAX_EP_STEPS,GAMMA,LR_A,LR_C,env)
    history=model.train()
    model.save_history(history, 'a2c.csv')
    #model.test(2)
    
    #train完对三场降雨进行测试
    test_num=4
    r=model.test(test_num)
    
    print(r)
    