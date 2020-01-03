# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:02:42 2019

@author: chong
"""

import PPO
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
    model2 = PPO.PPO(env,200, 24, 'ppo2')
    history = model2.train()
    model2.save_history(history, 'ppo2.csv')
    
    #train完对三场降雨进行测试
    test_num=4
    r=model2.test(test_num)
    
    #print(r)
