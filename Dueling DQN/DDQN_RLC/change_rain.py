# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:47:58 2018

@author: chong
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:23:08 2018

@author: chong
"""

from yaml import load
import math
import matplotlib.pyplot as plt
import numpy as np
import random

constants = load(open('./constants.yml', 'r', encoding='utf-8'))


def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def replace_line(line, title,rain,t):

    node=line.split()
    if(node[0]=='Oneyear-2h'):
        t=t+1
        tem=node[0]+' '*8+node[1]+' '+node[2]+' '*6+str(rain)
        #print(tem)
        line=tem
        return t,line
    else:
        return t,line


def handle_line(line, flag, title,rain,t):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    elif line.find(';') == -1 and flag:
        t,line = replace_line(line, title,rain,t)
    return t,line, flag


def change_rain(rain,infile):
    temfile=infile+'tem_rain.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        rain_flag =  False
        t=0
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            t,line, flag = handle_line(line, rain_flag, '[TIMESERIES]',rain[t],t)
            rain_flag = flag
            output.write(line + '\n')
    output.close()
    copy_result(infile,temfile)


def gen_rain(t,A,C,P,b,n,R,deltt):
    '''
    t是生成雨量时间序列步数上限,t是总的时间
    delt是时间间隔，取1
    '''
    rain=[]
    for i in range(t):
        if i <int(t*R):
            #rain.append(A*(1+C*math.log(P))*((1-n)*i/R+b)/167/math.pow((i/R+b),n+1))
            rain.append(A*(1+C*math.log(P))/math.pow(((t*R-i)+b),n))            
            
        else:
            #rain.append(A*(1+C*math.log(P))*((1-n)*(i-t*R)/(1-R)+b)/167/math.pow(((i-t*R)/(1-R)+b),n+1))
            rain.append(A*(1+C*math.log(P))/math.pow(((i-t*R)+b),n))
    
    return rain

def generate_rain_loop(rainFile,num,totalNum,t,minA,maxA,minC,maxC,CRatio,minP,maxP,b,n,minR,maxR,RRatio,deltt):#RNum为雨峰个数,num为一个峰降雨的数量
    rainData=[]
    
    for RNum in range(3):
        if RNum==0:
            pass

        elif RNum==1:#当为一个峰的降雨时
            for i in range(num):
                
                A=random.randint(minA,maxA)
                C=random.randint(minC,maxC)/CRatio
                P=random.randint(minP,maxP)
                
                R=random.randint(minR,maxR)/RRatio
                rain=gen_rain(t[-1],A,C,P,b,n,R,deltt)
                rainData.append(rain)
                #print(len(rainData))        
                
        else:#当为两个峰的降雨时
            #print(RNum)
    
            for j in range(totalNum-num):
                R_t1=random.randint(0,t[-1])
                #print(R_t1)
                if R_t1==t[-1]:
                    pass
                else:                                        
                    A=random.randint(minA,maxA)
                    C=random.randint(minC,maxC)/CRatio
                    P=random.randint(minP,maxP)
                    
                    R=random.randint(minR,maxR)/RRatio
                    rain1=gen_rain(R_t1,A,C,P,b,n,R,deltt)
                    
                    R_t2=t[-1]-R_t1
                    #print(R_t2)
                    rain2=gen_rain(R_t2,A,C,P,b,n,R,deltt)
                    rainData2=np.hstack((rain1,rain2))
                    rainData.append(rainData2)
                    #print(len(rainData))  
    #plt.plot(range(240),rainData2)
    print(len(rainData))

    np.savetxt(rainFile,rainData,fmt='%f',delimiter=',')        
       
        
    #print(rainData)
    return(rainData)
    
        


if __name__ == '__main__':
    infile='ot.inp'
    outfile='tem.inp'
    A=10#random.randint(5,15)
    C=13#random.randint(5,20)
    P=2#random.randint(1,5)
    b=1#random.randint(1,3)
    n=0.5#random.random()
    R=0.5#random.random()
    deltt=1
    t=240
    #change_rain(A,C,P,b,n,infile,outfile)
    rain=gen_rain(t,A,C,P,b,n,R,deltt)
    #plt.plot(range(240),rain)
    #copy_result(infile,'arg-original.inp')
    change_rain(rain,infile)
    
    date_t=[0,10,20,30,40,50,\
        60,70,80,90,100,110,\
        120,130,140,150,160,170,\
        180,190,200,210,220,230,240]
    trainRainData=generate_rain_loop('./sim/trainRainFile.txt',100,200,date_t,100,150,3,9,10,1,5,12,0.77,3,7,10,1)#生成200场降雨，其中100场为单峰，100场双峰。每场降雨240分值，一分钟一个降雨数据
    testRainData=generate_rain_loop('./sim/testRainFile.txt',2,4,date_t,100,150,3,9,10,1,5,12,0.77,3,7,10,1)#生成4场降雨，其中2场为单峰，2场双峰
    
    