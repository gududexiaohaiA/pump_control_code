# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:53:14 2018

@author: chong
"""

import numpy as np
import tensorflow as tf
import get_rpt
import set_datetime

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列
import GA_sim#导入GA运算文件
import GA_get_flooding_step#导入GA每步运行文件，产生每步的flooding
import os

import datetime

from pyswmm import Simulation
    
'''
date_time=['08:00','08:30','09:00','09:30','10:00','10:30',\
           '11:00','11:30','12:00']
date_t=[0,30,60,90,120,150,180,210,240]

date_time=['08:00','08:20','08:40','09:00','09:20','09:40','10:00','10:20','10:40',\
           '11:00','11:20','11:40','12:00']
date_t=[0,20,40,60,80,100,120,140,160,180,200,220,240]

pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2']}
limit_level=[0.9,3.02,0.9,3.02,0.9,1.26,1.53,1.71]
pool_name=['CC-storage','CC-storage','JK-storage','JK-storage','XR-storage','XR-storage','XR-storage','XR-storage']
max_depth=[5.6,4.8,7.72]
'''
date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
           '09:00','09:10','09:20','09:30','09:40','09:50',\
           '10:00','10:10','10:20','10:30','10:40','10:50',\
           '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
date_t=[0,10,20,30,40,50,\
        60,70,80,90,100,110,\
        120,130,140,150,160,170,\
        180,190,200,210,220,230,240]

class env_SWMM:
    def __init__(self, date_time, date_t):

        self.pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2']}#,'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
        self.limit_level={'CC-storage':[0.9,3.02,4.08]}#,'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
        self.max_depth={'CC-storage':5.6}#,'JK-storage':4.8,'XR-storage':7.72}
        self.pool_list=['CC-storage']#,'JK-storage','XR-storage']
        self.pool_d=[]
        self.action_space=[1.0 , 0.0]
        
        self.infile='./sim/oti'#每次修改降雨数据，进行模拟的input文件
        self.startfile='./sim/ot'#最初的inp文件
        self.sdate='08/28/2015'
        self.edate='08/28/2015'
        self.date_time=date_time
        self.date_t=date_t
        self.stime=date_time[0]
        
        self.GA_tem='./sim/GA_tem'#GA临时inpfile
        self.GAfile='./sim/GAfile'#GA模拟使用的inpfile
        self.GAStepfile='./sim/GAStepfile'#GA分步模拟使用的inpfile
        self.crossRate=0.7
        self.mutationRate=0.02
        self.lifeCount=10
        self.GAStepNum=1
        
        
        self._build_net()
        
        self.sess=tf.Session()
        
        self.trainRainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取总的训练降雨数据
        self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        
    def simulation(self,filename):
        with Simulation(filename) as sim:
            #stand_reward=0
            for step in sim:
                pass    
    
    def discount_reward(self,r):
        gamma=0.5
        discounted_r=np.zeros_like(r)
        running_add=0
        for t in reversed(range(r.size)):
            running_add=running_add*gamma+r[t]
            discounted_r[t]=running_add
        return discounted_r
    
    def _build_net(self):
        #计算图
        #For agent
        self.D=5

        self.H=20
        self.batch_size=1
        learning_rate=1e-1
        
        tf.reset_default_graph()    
        self.observations=tf.placeholder(tf.float32,[None,self.D],name="input_x")#输入数据
        self.W1=tf.get_variable("W1",shape=[self.D,self.H],initializer=tf.contrib.layers.xavier_initializer())#第一层权重5行20列
        self.layer1=tf.nn.relu(tf.matmul(self.observations,self.W1))#非线性化
        self.W2=tf.get_variable("W2",shape=[self.H,1],initializer=tf.contrib.layers.xavier_initializer())#第二层权重20行一列
        
        self.score=tf.matmul(self.layer1,self.W2)#分值
        self.probability=tf.nn.sigmoid(self.score)
        
        self.input_y=tf.placeholder(tf.float32,[None,1],name="input_y")
        self.advantages=tf.placeholder(tf.float32,name="reward_signal")
        self.loglik=tf.log(self.input_y*(self.input_y-self.probability)+(1-self.input_y)*(self.input_y+self.probability))
        self.loss=-tf.reduce_mean(self.loglik*self.advantages)
        
        self.tvars=tf.trainable_variables()
        self.newGrads=tf.gradients(self.loss,self.tvars)
        
        
        adam=tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.W1Grad=tf.placeholder(tf.float32,name="batch_grad1")
        self.W2Grad=tf.placeholder(tf.float32,name="batch_grad2")
        self.batchGrad=[self.W1Grad,self.W2Grad]
        self.updateGrads=adam.apply_gradients(zip(self.batchGrad,self.tvars))
        
                                                 
    def train(self): 
        xs,ys,drs=[],[],[]
        rendering=False
        init=tf.global_variables_initializer()
        self.sess.run(init)    
        gradBuffer=self.sess.run(self.tvars)

        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix]=grad*0
            
        etime=date_time[1]
        episode_number=0
        print(len(self.trainRainData))
        
        pumps=[]

        for pool in self.pool_list:#pool为前池
            for item in self.pump_list[pool]:
                pumps.append(item)

        
        while episode_number<len(self.trainRainData):
            reward_sum=0
            
            rainData=self.trainRainData[episode_number]

            s1= datetime.datetime.now()
            
            change_rain.copy_result(self.startfile+'.inp','arg-original.inp')#将最初输入文件arg-original.inp复制为start输入文件，每次reset对其修改降雨数据
            change_rain.copy_result(self.infile+'.inp','arg-original.inp')#初始化生成一个infile文件，infile为每个模拟所用input文件
            
            change_rain.change_rain(rainData,self.startfile+'.inp')#修改start inp文件中的降雨数据
            #print(A,C,P,b,n,R)
            
            change_rain.copy_result(self.GA_tem+'.inp',self.startfile+'.inp')#将修改了降雨数据的输入文件，复制为GA——tem文件
            #对比GA，生成GA算法在降雨时间内的策略和flooding数据，flooding数据作为DDQN每步模拟的基准数据
            self.GA_action_seq=GA_sim.GA_sim(self.GA_tem,self.GAfile,self.crossRate,self.mutationRate,self.lifeCount,self.date_time,pumps,self.GAStepNum)
            self.GA_flooding_step=GA_get_flooding_step.GA_get_flooding_step(self.GA_action_seq,self.GA_tem,self.GAStepfile,self.date_time,pumps)

            
            #先sim10min
            set_datetime.set_date(self.sdate,self.edate,self.stime,etime,self.startfile+'.inp')#修改start input文件中的时间数据
            
            change_rain.copy_result(self.infile+'.inp',self.startfile+'.inp')#将修改降雨数据后的start inp文件复制为infile文件
            self.simulation(self.infile+'.inp')
            
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.infile+'.rpt')

            self.pool_d=get_output.depth(self.infile+'.out',self.pool_list,self.date_t[1])
                
            
            action_seq=[]
            
            for i in range(1,len(self.date_t)-1):#对于一场雨中的每一步进行迭代运算
                rain_sum=sum(rainData[self.date_t[i]:self.date_t[i+1]])/max(rainData)
    
                action=[]    
                for pool in self.pool_list:#pool为前池 
                    
                    observation=[outflow/(0.001+total_in),flooding/(0.001+total_in),store/(0.001+total_in),self.pool_d[pool],rain_sum]
                    x_in=observation
                    x=np.reshape(x_in,[1,self.D])
                    tfprob=self.sess.run(self.probability,feed_dict={self.observations:x}) #开关泵概率   
                    
                    #对flage初始化为0，针对不同的情况设置flage
                    flage=0
                    
                    if self.pool_d[pool]>(self.limit_level[pool][0]) and self.pool_d[pool]<(self.limit_level[pool][2]):#判断是否大于最低水位，小于最高水位时，设置flag为0
                        flage=0
                    
                    elif self.pool_d[pool]<(self.limit_level[pool][0]):#当前池水位小于最低开泵水位时，flage设置为-1
                        flage=-1
                    else:#否则为1
                        flage=1
                            
                    #泵的启停策略
                    if flage==0:
                        if tfprob<self.action_space[1]+0.1:
                            #概率小于0.1的时候，不开泵
                            action.append(0)
                            action.append(0)
                            a=0
                        elif tfprob>=self.action_space[1]+0.1 and tfprob<self.action_space[1]+0.6:
                            #概率在0.1~0.6的时候，开一个泵
                            action.append(0)
                            action.append(1)
                            a=1
                        else:#开两个泵
                            action.append(1)
                            action.append(1) 
                            a=1
                    elif flage==-1:#小于最低水位不开泵
                        action.append(0)
                        action.append(0)
                        a=0
                    else:#flag为1的时候大于最高水位，泵全开
                        action.append(1)
                        action.append(1)
                        a=1
                    
                    xs.append(x)
                    y=1-a
                    ys.append(y)
                    
                #添加所有泵的action
                action_seq.append(action)
                #print(action_seq)
    
                #stime=date_time[i]
                etime=date_time[i+1]
                set_datetime.set_date(self.sdate,self.edate,self.stime,etime,self.startfile+'.inp')
                change_rain.copy_result(self.infile+'.inp',self.startfile+'.inp')
                set_pump.set_pump(action_seq,date_time[1:i+1],pumps,self.infile+'.inp')
                
                self.simulation(self.infile+'.inp')
                
                #获取rpt内信息，当前时刻的flooding，产生新的action
                total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.infile+'.rpt')
                #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
                self.pool_d=get_output.depth(self.infile+'.out',self.pool_list,self.date_t[i]-i)
                                
                #reward计算 当前reward只考虑溢流污染控制
                
                for pool in self.pool_list:
                    reward_sum+=(self.GA_flooding_step[i]-flooding)/(0.0001+total_in)#GA算法的flooding数据作为DDQN每步模拟的基准数据
                    drs.append((self.GA_flooding_step[i]-flooding)/(0.0001+total_in))
                   
            episode_number+=1
            #完成一场降雨模拟，更新agent
            #when the game over, which means the stick fall; means the time is out in the new situation
            #记录一场降雨的reward
            epx=np.vstack(xs)
            epy=np.vstack(ys)
            epr=np.vstack(drs)
            xs,ys,drs=[],[],[]
            discounted_epr=self.discount_reward(epr)
            discounted_epr-=np.mean(discounted_epr)
            discounted_epr/=np.std(discounted_epr)
    
            tGrad=self.sess.run(self.newGrads,feed_dict={self.observations:epx,self.input_y:epy,self.advantages:discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix]+=grad
            
            #若已有一个batch的reward值，用于更新agent
            if episode_number%self.batch_size==0:
                #print("train")
                self.sess.run(self.updateGrads,feed_dict={self.W1Grad:gradBuffer[0],self.W2Grad:gradBuffer[1]})
                print('Average reward for %d:%f.'%(episode_number,reward_sum/self.batch_size))
                #reward_sum=0
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix]=grad*0
                    
                #if abs(old_reward-reward_sum/self.batch_size)/abs(old_reward)<=1e-15:
                    #print("Task soveld in", episode_number)
                    #break
                #old_reward=reward_sum/self.batch_size
                
            #observation=env.reset()
        
    
            s2= datetime.datetime.now()
            print(s2-s1)
        print("training done")
        saver=tf.train.Saver()   
        sp=saver.save(self.sess,"./save/model.ckpt")
        print("model saved:",sp)
        return drs

        
    def test(self):    
        #反复多次进行模拟
        saver=tf.train.Saver()
        saver.restore(self.sess,"./save/model.ckpt")
        etime=date_time[1]
        xs,ys,drs=[],[],[]
        pumps=[]
        for pool in self.pool_list:#pool为前池
            for item in self.pump_list[pool]:
                pumps.append(item)

        for iten in range(len(self.testRainData)):
            reward_sum=0
                    
            change_rain.copy_result(self.startfile+'.inp','arg-original.inp')
            change_rain.copy_result(self.infile+'.inp','arg-original.inp')
            
            rainData=self.testRainData[iten]
                       
            change_rain.change_rain(rainData,self.startfile+'.inp')
            self.simulation(self.infile+'.inp')

            change_rain.copy_result(self.GA_tem+'.inp',self.startfile+'.inp')#将最初输入文件arg-original.inp复制为start输入文件，每次reset对其修改降雨数据
            #对比GA，生成GA算法在降雨时间内的策略和flooding数据，flooding数据作为DDQN每步模拟的基准数据
            self.GA_action_seq=GA_sim.GA_sim(self.GA_tem,self.GAfile,self.crossRate,self.mutationRate,self.lifeCount,self.date_time,self.pumps,self.GAStepNum)
            self.GA_flooding_step=GA_get_flooding_step.GA_get_flooding_step(self.GA_action_seq,self.GA_tem,self.GAStepfile,self.date_time,self.pumps)
            
            begin = datetime.datetime.now()

            change_rain.copy_result('./sim/test/result/en/inp/'+str(iten)+'.inp',self.startfile+'.inp')
            change_rain.copy_result('./sim/test/result/en/rpt/'+str(iten)+'.rpt',self.startfile+'.rpt')
            
            
            set_datetime.set_date(self.sdate,self.edate,self.stime,etime,self.startfile+'.inp')
            
            change_rain.copy_result(self.infile+'.inp',self.startfile+'.inp')
            
            self.simulation(self.infile+'.inp')
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.infile+'.rpt')
            #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
            self.pool_d=get_output.depth(self.infile+'.out',self.pool_list,self.date_t[1])
            
            action_seq=[]
            log_reward=''
            
            for i in range(1,len(self.date_t)-1):#用1场降雨生成结果,随机生成batch_size场降雨的inp
                rain_sum=sum(rainData[self.date_t[i]:self.date_t[i+1]])/max(rainData)
     
                action=[]

                for pool in self.pool_list:
                    observation=[outflow/total_in,flooding/total_in,store/total_in,self.pool_d[pool],rain_sum]
                    x_in=observation
                    x=np.reshape(x_in,[1,self.D])
                    tfprob=self.sess.run(self.probability,feed_dict={self.observations:x})    
                    
                    #对flage初始化为0，针对不同的情况设置flage
                    flage=0
                    
                    if self.pool_d[pool]>(self.limit_level[pool][0]) and self.pool_d[pool]<(self.limit_level[pool][2]):#判断是否到达最低水位，当到达最低水位时，设置flag为True
                        flage=0
                    
                    elif self.pool_d[pool]<(self.limit_level[pool][0]):#当前池水位小于最低开泵水位时，flage设置为-1
                        flage=-1
                    else:
                        flage=1
                            
                    #泵的启停策略
                    if flage==0:
                        if tfprob<self.action_space[1]+0.1:#概率小于0的时候，不开泵
                            action.append(0)
                            action.append(0)
                            a=0
                        elif tfprob>=self.action_space[1]+0.1 and tfprob<self.action_space[1]+0.6:#概率在0.1~0.6的时候，开一个泵
                            action.append(0)
                            action.append(1)
                            a=1
                        else:#开两个泵
                            action.append(1)
                            action.append(1) 
                            a=1
                    elif flage==-1:
                        action.append(0)
                        action.append(0)
                        a=0
                    else:#flag为1的时候全开
                        action.append(1)
                        action.append(1)
                        a=1
                    
                    xs.append(x)
                    y=1-a
                    ys.append(y)
                    
                #设置pump并模拟之后才有reward
                action_seq.append(action)
                
                #stime=date_time[i]
                etime=self.date_time[i+1]
                set_datetime.set_date(self.sdate,self.edate,self.stime,etime,self.startfile+'.inp')
                change_rain.copy_result(self.infile+'.inp',self.startfile+'.inp')
                set_pump.set_pump(action_seq,self.date_time[1:i+1],pumps,self.infile+'.inp')
                                
                self.simulation(self.infile+'.inp')
                #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
                #获取rpt内信息，产生新的action
                total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(self.infile+'.rpt')
                #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
                self.pool_d=get_output.depth(self.infile+'.out',self.pool_list,date_t[i]-i)
                                
                #reward计算 当前reward只考虑溢流污染控制
                for pool in self.pool_list:
                    reward_sum+=(self.GA_flooding_step[i]-flooding)/(0.0001+total_in)#GA算法的flooding数据作为DDQN每步模拟的基准数据
                    drs.append((self.GA_flooding_step[i]-flooding)/(0.0001+total_in))
                        
                log_reward+=str(reward_sum)+'\n'
                
                
            end = datetime.datetime.now()
            print(iten,'  ',end-begin)
            f=open('reward'+str(iten)+'.txt','w')
            f.write(log_reward)
            #保存inp与rpt文件
            change_rain.copy_result('./sim/test/result/ai/inp/'+str(iten)+'.inp','./sim/test/oti.inp')
            change_rain.copy_result('./sim/test/result/ai/rpt/'+str(iten)+'.rpt','./sim/test/oti.rpt')
            print("操控序列：",action_seq)
            print("得分：",reward_sum)


if __name__ == '__main__':
    
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    env=env_SWMM(date_time, date_t)
    if os.path.exists("./save/model.ckpt"):
        env.test()     
    else:    
        env.train()
        env.test()     
    
