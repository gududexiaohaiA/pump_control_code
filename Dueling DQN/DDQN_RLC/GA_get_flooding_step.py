# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:46:00 2019

@author: admin
"""

import numpy as np
import change_rain#随机生成降雨序列
import set_pump
from pyswmm import Simulation
import get_rpt

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass 


def GA_get_flooding_step(action_seq,startflie,floodStepFile,date_time,pumps):
    action_step=[]
    flooding_step=[]
    for i in range(0,action_seq.shape[0]):
        
        if i==0:
            action_step.append(action_seq[0])
            action_step.append(action_seq[1])
            #print(action_step)
            #action_step_list=action_step.tolist()
            
            change_rain.copy_result(floodStepFile+'.inp',startflie+'.inp')
            #print(date_time[0:i+2])
            
            set_pump.set_pump(action_seq,date_time[0:2],pumps,floodStepFile+'.inp')   
            simulation(floodStepFile+'.inp')
            _,flooding,_,_,_,_=get_rpt.get_rpt(floodStepFile+'.rpt')
            flooding_step.append(flooding)
            #print(flooding)
    
            
        if i>1:
            action_step.append(action_seq[i])
            change_rain.copy_result(floodStepFile+'.inp',startflie+'.inp')
            set_pump.set_pump(action_seq,date_time[0:i+1],pumps,floodStepFile+'.inp')   
            simulation(floodStepFile+'.inp')
            _,flooding,_,_,_,_=get_rpt.get_rpt(floodStepFile+'.rpt')
            flooding_step.append(flooding)
            #print(flooding_step)
    return flooding_step
