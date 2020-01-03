# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:12:54 2019

@author: chong


Applying FCA on model-free RL for pump control
Good luck!!!! 
"""
import tensorflow as tf
import numpy as np
import xlrd
import pandas as pd
import set_datetime
import get_output#从out文件读取水动力初值
import change_rain#随机生成降雨序列
import get_rpt
import set_pump
import random

#import datetime
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler

from pyswmm import Simulation

def max_data(data1):
    amax=np.array([])
    [n,m]=data1.shape
    for i in range(m):
        tmax=data1[0][i]
        for j in range(n):
            if  data1[j][i]>tmax:
                tmax=data1[j][i]
        T=np.append(amax,tmax) 
        amax=T
    return amax
    
def min_data(data1):
    amin=np.array([])
    [n,m]=data1.shape
    for i in range(m):
        tmin=data1[0][i]
        for j in range(n):
            if data1[j][i]<tmin:
                tmin=data1[j][i]
        T=np.append(amin,tmin)
        amin=T
    return amin

def normalization(data1):
    amin=min_data(data1)
    amax=max_data(data1)
    [n,m]=data1.shape
    for i in range(m):
        for j in range(n):
            if(amax[i]==amin[i]):
                data1[j][i]=0
            else:
                data1[j][i]=2*(data1[j][i]-amin[i])/(amax[i]-amin[i])-1
        
    return data1

def re_normalization(data1,amin,amax):
    [n,m]=data1.shape
    redata=[]
    for i in range(n):
        tem=[]
        for j in range(m):
            tem.append(((1+data1[i][j])*(amax[j]-amin[j]))/2+amin[j])
        redata.append(tem)
        
    return redata

def normalization_ver1(data1,amin,amax):
    [n,m]=data1.shape
    for i in range(m):
        for j in range(n):
            data1[j][i]=2*(data1[j][i]-amin[i])/(0.001+amax[i]-amin[i])-1
        
    return data1

def read_data(st,sheet):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[sheet]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(1,nrows):
        tem=[]
        #print(i)
        for j in range(ncols):
            s=table.cell(i,j).value
            tem.append(float(s)) 
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows-1,ncols)
    print(t_data.shape)
    return t_data

def save_xls_file(data,name): 
    csv_pd = pd.DataFrame(data)  
    csv_pd.to_csv(name+".csv", sep=',', header=False, index=False)

def discount_reward(r):
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(r.size)):
        running_add=running_add*gamma+r[t]
        discounted_r[t]=running_add
    return discounted_r

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass    

def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()

def bc_data(node,label):
    T=len(node)
    N=len(node[0])
    #M=len(node[0][0])
    #print(T,N,M)
    n=0
    if label=='inflow':
        n=4
    else:
        n=5
    data=[]
    for t in range(T):
        text=[]
        for iten in range(N):
            text.append(node[t][iten][n][0])
        data.append(text)
    return data

def ic_data(node,link):
    data=[]
    #T=len(node)
    N=len(node[0])
    #M=len(node[0][0])
    for i in range(2):
        for iten in range(N):
            data.append(node[0][iten][i][0])
    N=len(link[0])
    for i in range(2):
        for iten in range(N):
            data.append(link[0][iten][i][0])
    return data



tf.reset_default_graph() 
'''
构建水力水质估计模型FCA
基于sklearn
'''
#training data
str_data='./NN_mf/od.xlsx'
print(str_data)
data0=read_data(str_data,0)
maxdata=max_data(data0)
mindata=min_data(data0)

tdata0=normalization(data0)
[tnum,xnum]=data0.shape
training_num=int(tnum)
#test data
test_datain1=data0[0:tnum-1,:]
inflow_data=read_data(str_data,1)
flooding_data=read_data(str_data,2)
pump_data=read_data(str_data,3)


dataMat = np.array(tdata0)
X=dataMat[0:training_num-1,:]
y = dataMat[1:training_num,:]
scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集
FCA_env = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 10), random_state=1)
FCA_env.fit(X, y)
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络

dataMatt = np.array(test_datain1)
Xt=dataMatt[0:training_num-1,:]
yt = dataMatt[1:training_num,:]
scalert = StandardScaler() # 标准化转换
scalert.fit(Xt)  # 训练标准化对象
Xt = scaler.transform(Xt)   # 转换数据集
result=FCA_env.predict(Xt)
result=re_normalization(result,mindata,maxdata)
print('training done', len(result))  # 预测某个输入对象
save_xls_file(result,'sktest')

'''
#构建policy network的control NN
'''
#计算图
#For agent
deltt=1
H=20
batch_size=3
learning_rate=1e-1
D=5
gamma=0.99
xs,ys,drs=[],[],[]
reward_sum=0
episode_number=0
total_episodes=100  
observations=tf.placeholder(tf.float32,[None,D],name="input_x")
W1=tf.get_variable("W1",shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
layer1=tf.nn.relu(tf.matmul(observations,W1))
W2=tf.get_variable("W2",shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
score=tf.matmul(layer1,W2)
probability=tf.nn.sigmoid(score)
input_y=tf.placeholder(tf.float32,[None,1],name="input_y")
advantages=tf.placeholder(tf.float32,name="reward_signal")
loglik=tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability))
loss=-tf.reduce_mean(loglik*advantages)

tvars=tf.trainable_variables()
newGrads=tf.gradients(loss,tvars)

adam=tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad=tf.placeholder(tf.float32,name="batch_grad1")
W2Grad=tf.placeholder(tf.float32,name="batch_grad2")
batchGrad=[W1Grad,W2Grad]
updateGrads=adam.apply_gradients(zip(batchGrad,tvars))


'''
#以一个session运行policy network实现采样训练控制 使用sklearn作为env
'''
    

with tf.Session() as sess:    

    #policy network采样训练

    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
           '09:00','09:10','09:20','09:30','09:40','09:50',\
           '10:00','10:10','10:20','10:30','10:40','10:50',\
           '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,239]
    pump_list={'CC-storage':['CC-Pump-1']}#,'CC-Pump-2']},'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
    limit_level={'CC-storage':[0.9,4.08]}#,'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
    max_depth={'CC-storage':5.6}#,'JK-storage':4.8,'XR-storage':7.72}
    pool_list=['CC-storage']#,'JK-storage','XR-storage']

    rendering=False
    init=tf.global_variables_initializer()
    sess.run(init)    
    gradBuffer=sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix]=grad*0

    old_reward=1000000
    while episode_number<=total_episodes:

        #用不同降雨带入一个临时的SWMM生成对应的边界条件inflow 从node和link的值也有初始条件

        #每一次batch都新生成一个新的降雨
        A=random.randint(50,150)
        C=random.randint(3,9)/10.00
        #P=random.randint(1,5)
        #b=random.randint(1,3)
        #n=random.random()
        P=random.randint(1,5)
        b=12
        n=0.77
        R=random.randint(2,8)/10.00
        rain=change_rain.gen_rain(date_t[-1],A,C,P,b,n,R,deltt)
        
        
        bcfile='./boundary condition/otbc'
        bcstartfile='./boundary condition/ot'
        change_rain.copy_result(bcstartfile+'.inp',bcfile+'.inp')
  
        change_rain.change_rain(rain,bcfile+'.inp')
        #print(A,C,P,b,n,R)
        
        sdate=edate='08/28/2015'
        stime=date_time[0]
        etime=date_time[-1]
        set_datetime.set_date(sdate,edate,stime,etime,bcfile+'.inp')
        
        simulation(bcfile+'.inp')
        filename=bcfile+'.out'
        sub,node,link,sub_name,node_name,link_name=get_output.read_out(filename)
        #边界条件在node中
        inflow_data=bc_data(node,'inflow')
        flooding_data=bc_data(node,'flooding')
        #初始条件
        init_data=ic_data(node,link)
        #print(len(init_data))
        
        
        mf_result=[]
        mf_result.append(init_data)
        

        #借助inflow与上述FCA进行一个时间步的模拟并获取reward与state 先进行第一个时间步的控制

        #获取policy network的输入
        total_in=sum(inflow_data[0])
        flooding=sum(flooding_data[0])
        store=total_in-flooding
        outflow=10*(init_data[4]+init_data[5])
        upflow=10*init_data[6]
        downflow=10*sum(init_data)-10*init_data[4]-10*init_data[5]-28.26*init_data[6]
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        pool_d={}
        for item in pool_list:
            pool_d[item]=init_data[6]
        rain_sum=sum(rain[date_t[0]:date_t[1]])/max(rain)
        
        action_seq=[]
        for t in range(1,len(date_t)-1):#用1场降雨生成结果,随机生成batch_size场降雨的inp
            #利用state和reward调用policy network得到下一步的action进行逐时间步控制，控制后生成新的reward和state 
            action=[]
            pumps=[]
            for pool in pool_list:
                observation=[outflow/total_in,flooding/total_in,store/total_in,pool_d[pool],rain_sum]
                x_in=observation
                x=np.reshape(x_in,[1,D])
                tfprob=sess.run(probability,feed_dict={observations:x})    
                
                #初始化pump的action序列
                temaction=[]
                for pump in pump_list[pool]:
                    temaction.append(0)
                    
                flage=True
                if pool_d[pool]>(limit_level[pool][0]):
                    flage=True
                else:
                    flage=False

                if np.random.uniform()+0.5>tfprob and flage:
                    a=1
                    #设置pump
                    temaction[-1]=1
                    for j in range(len(limit_level[pool])-1):
                        if pool_d[pool]>=limit_level[pool][j+1]:
                            temaction[j]=1
                else:
                    a=0
                
                for item in temaction:
                    action.append(item)
                for item in pump_list[pool]:
                    pumps.append(item)
                xs.append(x)
                y=1-a
                #y=np.mean(a)
                ys.append(y)
                #设置pump并模拟之后才有reward
            

            #给到pump的操控策略进行一个时间步的模拟

            action_seq.append(action)
            for i in range(date_t[t-1],date_t[t]):
                tem=[]
                for j in range(len(init_data)):
                    if j<len(inflow_data[0]):
                        if j !=len(inflow_data[0])-1:
                            tem.append(init_data[j]-flooding_data[i][j]/100+inflow_data[i][j]/100)#+lateral_inflow_data[i,j])#
                        else:
                            tem.append(init_data[j]-flooding_data[i][j]/282.6+inflow_data[i][j]/282.6-action[0]*17.28/282.6)#只有一个泵，对应最后一位流量，所以这样做
                    else:
                        tem.append(init_data[j])
                    #tem.append(result[0][j])

                tem=np.array([tem])
                #env里模拟需要二维list
                '''
                scalert = StandardScaler() # 标准化转换
                scalert.fit(tem)  # 训练标准化对象
                tem = scaler.transform(tem)   # 转换数据集
                '''
                tem=normalization_ver1(tem,mindata,maxdata)
                
                init_data=FCA_env.predict(tem)
                #回到control的时候需要一维list
                
                init_data=re_normalization(init_data,mindata,maxdata)
                #init_data=init_data.tolist()[0]
                init_data=init_data[0]
                mf_result.append(init_data)
            #获取policy network的输入
            total_in=sum(inflow_data[date_t[t]-1])
            flooding=sum(flooding_data[date_t[t]-1])
            store=total_in-flooding
            outflow=10*(init_data[4]+init_data[5])
            upflow=10*init_data[6]
            downflow=10*sum(init_data)-10*init_data[4]-10*init_data[5]-28.26*init_data[6]
            #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
            pool_d={}
            for item in pool_list:
                pool_d[item]=init_data[6]
            rain_sum=sum(rain[date_t[0]:date_t[1]])/max(rain)
            
            
            for pool in pool_list:
                reward=0
                if flooding/total_in>=0.5:
                    reward-=1.0
                else:
                    reward-=10
                
                if store/total_in>=0.0001:
                    reward-=1.0
                    
                if upflow<downflow:
                    reward+=10.0
                else:
                    reward-=100
                    
                
                if pool_d[pool]<=limit_level[pool][0] or pool_d[pool]>=max_depth[pool]:
                    reward-=10.0
            
                reward_sum+=reward
                drs.append(reward)
            #print(len(xs),len(ys),len(drs))
        
            print(action_seq)
            #plt.plot(range(239),rain)
        episode_number+=1
        save_xls_file(mf_result,'mfresult'+str(episode_number))
        
        #when the game over, which means the stick fall; means the time is out in the new situation
        #记录一场降雨的reward
        epx=np.vstack(xs)
        epy=np.vstack(ys)
        epr=np.vstack(drs)
        xs,ys,drs=[],[],[]
        discounted_epr=discount_reward(epr)
        discounted_epr-=np.mean(discounted_epr)
        discounted_epr/=np.std(discounted_epr)

        tGrad=sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discounted_epr})
        for ix,grad in enumerate(tGrad):
            gradBuffer[ix]+=grad
        
        #若已有一个batch的reward值，用于更新agent
        if episode_number%batch_size==0:
            print("train")
            sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix]=grad*0
                print('Average reward for %d:%f.'%(episode_number,reward_sum/batch_size))
            #if reward_sum/batch_size>370:#reward标准待定
            if abs(old_reward-reward_sum/batch_size)<=0.001:
                print("Task soveld in", episode_number)
                break
            old_reward=reward_sum/batch_size
            reward_sum=0
        #observation=env.reset()
        
    print("training done")
    
    

    #用准备好的10个SWMM测试训练的policy network

    for iten in range(10):
        infile='./test SWMM/ot'+str(iten)
        
        #perpare inp
        copy_result('./test SWMM/en/ot'+str(iten)+'.inp',infile+'.inp')
        copy_result('./test SWMM/ai/ot'+str(iten)+'.inp',infile+'.inp')
        copy_result('./test SWMM/ai/tem/ot'+str(iten)+'.inp',infile+'.inp')
        
        #human
        eninfile='./test SWMM/en/ot'+str(iten)
        simulation(eninfile+'.inp')
        
        #AI
        aiinfile='./test SWMM/ai/ot'+str(iten)
        aiinfile_tem='./test SWMM/ai/tem/ot'+str(iten)
        sdate=edate='08/28/2015'
        #先sim10min
        stime=date_time[0]
        etime=date_time[1]
        set_datetime.set_date(sdate,edate,stime,etime,aiinfile_tem+'.inp')
        
        simulation(aiinfile_tem+'.inp')
        #获取rpt内信息，产生新的action
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(aiinfile_tem+'.rpt')
        #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
        pool_d=get_output.depth(aiinfile_tem+'.out',pool_list,date_t[1])
        rain=[]
        #获取降雨数据
        with open(aiinfile_tem+'.inp', 'rt') as data:
            for line in data:
                line = line.rstrip('\n')
                node = line.split()
                if node!=[]:
                    if node[0].find('Oneyear-2h')>=0:
                        rain.append(float(node[-1]))
        
        rain_sum=sum(rain[date_t[0]:date_t[1]])/max(rain)
        
        action_seq=[]
        reward_sum=0
        for i in range(1,len(date_t)-1):#用1场降雨生成结果,随机生成batch_size场降雨的inp
 
            action=[]
            pumps=[]
            for pool in pool_list:
                observation=[outflow/total_in,flooding/total_in,store/total_in,pool_d[pool],rain_sum]
                x_in=observation
                x=np.reshape(x_in,[1,D])
                tfprob=sess.run(probability,feed_dict={observations:x})    
                
                #初始化pump的action序列
                temaction=[]
                for pump in pump_list[pool]:
                    temaction.append(0)
                    
                flage=True
                if pool_d[pool]>(limit_level[pool][0]):
                    flage=True
                else:
                    flage=False

                if np.random.uniform()+0.5>tfprob and flage:
                    a=1
                    #设置pump
                    temaction[-1]=1
                    for j in range(len(limit_level[pool])-1):
                        if pool_d[pool]>=limit_level[pool][j+1]:
                            temaction[j]=1
                else:
                    a=0
                
                for item in temaction:
                    action.append(item)
                for item in pump_list[pool]:
                    pumps.append(item)
                xs.append(x)
                y=1-a
                #y=np.mean(a)
                ys.append(y)
                
            #设置pump并模拟之后才有reward
            action_seq.append(action)
            
            #stime=date_time[i]
            etime=date_time[i+1]
            copy_result(aiinfile_tem+'.inp',aiinfile+'.inp')
            set_datetime.set_date(sdate,edate,stime,etime,aiinfile_tem+'.inp')
            set_pump.set_pump(action_seq,date_time[1:i+1],pumps,aiinfile_tem+'.inp')
            
            
            simulation(aiinfile_tem+'.inp')
            #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
            #获取rpt内信息，产生新的action
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(aiinfile_tem+'.rpt')
            #在确定泵开关之前确定最末时刻（当前）前池水位，水位过低时不开启
            pool_d=get_output.depth(aiinfile_tem+'.out',pool_list,date_t[i]-i)
            
            rain_sum=sum(rain[date_t[i]:date_t[i+1]])/max(rain)
            
            for pool in pool_list:
                reward=0
                if flooding/total_in>=0.5:
                    reward-=1.0
                else:
                    reward-=100
                
                if store/total_in>=0.0001:
                    reward-=1.0
                    
                if upflow<downflow:
                    reward+=10.0
                else:
                    reward-=100
                    
                
                if pool_d[pool]<=limit_level[pool][0] or pool_d[pool]>=max_depth[pool]:
                    reward-=10.0
            
                reward_sum+=reward

        #保存inp与rpt文件
        #copy_result('./sim/test/result/ai/inp/'+str(iten)+'.inp','./sim/test/oti.inp')
        #copy_result('./sim/test/result/ai/rpt/'+str(iten)+'.rpt','./sim/test/ot.rpt')
        print("操控序列：",action_seq)
        print("得分：",reward_sum)   