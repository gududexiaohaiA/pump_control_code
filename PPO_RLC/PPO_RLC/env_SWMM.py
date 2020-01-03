# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:22:14 2019

@author: chong

env based on SWMM
"""

#import gym

import numpy as np
import get_rpt
import set_datetime

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列

import random
import GA_sim
import GA_get_flooding_step

from pyswmm import Simulation
    


class env_SWMM:
    def __init__(self, date_time, date_t):
        
        self.action_space=[1.0 , 0.0]
        
        self.orf='./sim/orf'#原始的inp文件，只含有管网信息
        self.orf_rain='./sim/orf_rain'#
        self.staf='./sim/staf'#用于sim的inp文件，在orf基础上修改了时间与降雨
        self.orftem='./sim/orf_tem'#最终模拟使用的file
        self.GA_tem='./sim/GA_tem'#GA临时inpfile
        self.GAfile='./sim/GAfile'#GA模拟使用的inpfile
        self.GAStepfile='./sim/GAStepfile'#GA分步模拟使用的inpfile
        
        change_rain.copy_result(self.staf+'.inp',self.orf+'.inp')
        change_rain.copy_result(self.orf_rain+'.inp',self.orf+'.inp')
        change_rain.copy_result(self.orftem+'.inp',self.orf+'.inp')
        
        
        self.date_time=date_time
        self.date_t=date_t
        self.T=len(self.date_t)
        
        
        self.deltt=1
        
        self.iten=0#当前模拟的时间步
        self.action_seq=[]
        
        self.sdate=self.edate='08/28/2015'
        #先sim10min
        self.stime=date_time[0]
        self.etime=date_time[1]
        
        self.pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2']}#,'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
        self.limit_level={'CC-storage':[0.9,3.02,4.08]}#,'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
        self.max_depth={'CC-storage':5.6}#,'JK-storage':4.8,'XR-storage':7.72}
        self.pool_list=['CC-storage']#,'JK-storage','XR-storage']
        
        self.rain=[]
        
        self.pool_d=[]
        
        self.crossRate=0.7
        self.mutationRate=0.02
        self.lifeCount=10
        
        self.GA_action_seq=[]
        self.GAStepNum=1
        self.GA_flooding_step=[]
        
        
    def simulation(self,filename):
        with Simulation(filename) as sim:
            #stand_reward=0
            for step in sim:
                pass    
    
    def copy_result(self,outfile,infile):
        output = open(outfile, 'wt')
        with open(infile, 'rt') as data:
            for line in data:
                output.write(line)
        output.close()
    
    def reset(self,raindata):
        #每一次batch都新生成一个新的降雨
        #每一次reset都赋予新的降雨，新的泵序列
        
#        set_datetime.set_date(self.sdate,self.edate,self.stime,self.etime,self.staf+'.inp')
#        A=random.randint(100,150)
#        C=random.randint(3,9)/10.00
#        P=random.randint(1,5)
#        b=12
#        n=0.77
#        R=random.randint(3,7)/10.00
#        self.rain=change_rain.gen_rain(self.date_t[-1],A,C,P,b,n,R,self.deltt)
        
        change_rain.change_rain(raindata,self.orf_rain+'.inp')#先修改self.orf_rain，再复制给staf
        change_rain.copy_result(self.staf+'.inp',self.orf_rain+'.inp')
        change_rain.copy_result(self.orftem+'.inp',self.orf_rain+'.inp')
        
        change_rain.copy_result(self.GA_tem+'.inp',self.orf_rain+'.inp')#将修改了降雨数据的self.orf_rain.inp复制给GAfile.inp文件
        
        self.iten=1
        self.action_seq=[]
        
        tem_etime=self.date_time[self.iten]
        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,self.staf+'.inp')
        
        pumps=[]
        for pool in self.pool_list:        
            for item in self.pump_list[pool]:
                pumps.append(item)
        
        '''
        #对比HC
        change_rain.copy_result('compare_tem_HC.inp',self.orf_rain+'.inp')#还原
        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,'compare_tem_HC.inp')
        self.simulation('compare_tem_HC.inp')
        '''
        #对比GA，生成GA算法在降雨时间内的策略和flooding数据，flooding数据作为DDQN每步模拟的基准数据
        self.GA_action_seq=GA_sim.GA_sim(self.GA_tem,self.GAfile,self.crossRate,self.mutationRate,self.lifeCount,self.date_time,pumps,self.GAStepNum)
        self.GA_flooding_step=GA_get_flooding_step.GA_get_flooding_step(self.GA_action_seq,self.GA_tem,self.GAStepfile,self.date_time,pumps)
        
        self.simulation(self.staf+'.inp')
     
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.staf+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        self.pool_d=get_output.depth(self.staf+'.out',self.pool_list,self.date_t[self.iten]-self.iten)
        rain_sum=sum(raindata[self.date_t[self.iten]:self.date_t[self.iten+1]])/max(raindata)
        
        for pool in self.pool_list:
            state=np.array([outflow/total_in,flooding/total_in,store/total_in,self.pool_d[pool],rain_sum])
        
        return state
    
    def step(self,a,raindata):
        #修改statf的date，根据iten逐步向前
        #加入action
        #开始模拟，存储结果

        self.iten+=1
        action=[]
        pumps=[]
        
        #print('a=',a)
        for pool in self.pool_list:
            #检测水位，根据水位决定泵启停
            flage=0
            if self.pool_d[pool]>(self.limit_level[pool][0]) and self.pool_d[pool]<(self.limit_level[pool][2]):
                flage=0
            elif self.pool_d[pool]<(self.limit_level[pool][0]):
                flage=-1
            else:
                flage=1
            
            #泵的启停策略
            if flage==0:
                if a<self.action_space[1]+0.1:
                    action.append(0)
                    action.append(0)
                elif a>=self.action_space[1]+0.1 and a<self.action_space[1]+0.6:
                    action.append(0)
                    action.append(1)
                else:
                    action.append(1)
                    action.append(1)               
            elif flage==-1:
                action.append(0)
                action.append(0)
            else:
                action.append(1)
                action.append(1)
                
                
            for item in self.pump_list[pool]:
                pumps.append(item)
                
            #设置pump并模拟之后才有reward
            self.action_seq.append(action)
            #print(self.action_seq)

        set_pump.set_pump(self.action_seq,self.date_time[1:self.iten],pumps,self.orftem+'.inp')
        
        
        tem_etime=self.date_time[self.iten]
        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,self.orftem+'.inp')
        #change_rain.copy_result(infile+'.inp',startfile+'.inp')
        
        #还原SWMM缓存inp
        change_rain.copy_result(self.staf+'.inp',self.orftem+'.inp')
        change_rain.copy_result(self.orftem+'.inp',self.orf_rain+'.inp')
        
        #step forward
        self.simulation(self.staf+'.inp')

        #从out和rpt文件读取sate值
        #如果iten==最终的时间步，模拟停止
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.staf+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        self.pool_d=get_output.depth(self.staf+'.out',self.pool_list,self.date_t[self.iten]-self.iten)
        rain_sum=sum(raindata[self.date_t[self.iten]:self.date_t[self.iten+1]])/max(raindata)
        
        for pool in self.pool_list:
            state=np.array([outflow/(0.001+total_in),flooding/(0.001+total_in),store/(0.001+total_in),self.pool_d[pool],rain_sum])
        
        self.simulation(self.staf+'.inp')
        
        #reward计算 当前reward只考虑溢流污染控制
        '''
        #reward1
        reward_sum=0
        for pool in self.pool_list:
            if flooding>total_in*0.1:
                reward_sum+=-1.0
            else:
                reward_sum+=1.0
        '''
        
        #try different reward
#        change_rain.copy_result('compare_tem_HC.inp',self.orf_rain+'.inp')
#        set_datetime.set_date(self.sdate,self.edate,self.stime,tem_etime,'compare_tem_HC.inp')
#         #reward2使用的标准比对baseline
#        self.simulation('compare_tem_HC.inp')
#        _,flooding_compare,_,_,_,_=get_rpt.get_rpt('compare_tem_HC.rpt')
        
        '''
        #reward2
        reward_sum=0
        for pool in self.pool_list:
            if flooding_compare!=0.0:
                reward_sum+=(flooding_compare-flooding)/flooding_compare
            else:
                reward_sum+=-flooding/(0.0001+total_in)
        '''
                
        #与GA算法计算的flooding进行比较        
        reward_sum=0
        for pool in self.pool_list:
            reward_sum+=(self.GA_flooding_step[self.iten]-flooding)/(0.0001+total_in)#GA算法的flooding数据作为DDQN每步模拟的基准数据
        
        if self.iten==self.T-2:
            done=True
        else:
            done=False

        return state,reward_sum,done,{}




if __name__=='__main__':

    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    
    print(date_t[24])
    env=env_SWMM(date_time, date_t)
    observation=env.reset()
    print(observation)
    for t in range(len(date_t)-3):
        r=env.step(0.3)
        print(env.iten)
        print(r)
    '''
    env = gym.make('Pendulum-v0')
    bound = env.action_space.high[0]
    
    print(bound)
    observation = env.reset()
    
    print(observation)
    
    r=env.step([2,3])
    print(r)
    r=env.step([3,4])
    print(r)
    r=env.step([3,4])
    print(r)
    '''