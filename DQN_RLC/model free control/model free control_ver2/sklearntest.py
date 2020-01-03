# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:01:22 2019

@author: chong
"""

import numpy as np
import xlrd
import pandas as pd
#import datetime
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler


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



'''
构建水力水质估计模型FCA
基于sklearn
'''
#training data
str_data='./NN_mf/od.xlsx'
print(str_data)
data0=read_data(str_data,0)
[tnum,xnum]=data0.shape
training_num=int(tnum)
'''
training_datain=data0[0:training_num-1,:]
training_dataout=data0[1:training_num,:]
'''
#test data
test_datain1=data0[0:tnum-1,:]
inflow_data=read_data(str_data,1)
flooding_data=read_data(str_data,2)
pump_data=read_data(str_data,3)


dataMat = np.array(data0)
X=dataMat[0:training_num-1,:]
y = dataMat[1:training_num,:]
print(len(X),len(X[0]))
scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集
clf = MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 10), random_state=1)
#clf.fit(X, y)
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
dataMatt = np.array(test_datain1)
Xt=dataMatt[0:training_num-1,:]
yt = dataMatt[1:training_num,:]
scalert = StandardScaler() # 标准化转换
scalert.fit(Xt)  # 训练标准化对象
Xt = scaler.transform(Xt)   # 转换数据集
#result=clf.predict(Xt)
#print('预测结果：', len(result))  # 预测某个输入对象

print(len(Xt),len(Xt[0]))
for i in range(training_num-2):#int(training_num/100)):
    tem=[]
    for j in range(len(Xt[i])):
        if j<len(inflow_data[i]):
            if j !=len(inflow_data[0])-1:
                tem.append(Xt[i][j]-flooding_data[i][j]/10+inflow_data[i][j]/10)#+lateral_inflow_data[i,j])#
            else:
                tem.append(Xt[i][j]-flooding_data[i][j]/28.26+inflow_data[i][j]/28.26-pump_data[i]*17.4/28.6)
        else:
            tem.append(Xt[i][j])
    tem=[tem]
    scalert.fit(tem)
    tem = scaler.transform(tem)
    clf.fit(tem, [y[i+1]])




X0=[dataMatt[0,:]]
scalert=StandardScaler()
scalert.fit(X0)
X0 = scaler.transform(X0)
init_data=clf.predict(X0)
print(len(init_data),len(init_data[0]))

result=[]
result.append(init_data)

text=''
for i in range(training_num-2):#int(training_num/100)):
    if i>=training_num/3 and i <=training_num*3/4:
        action=1
    else:
        action=0
    tem=[]
    for j in range(len(init_data[0])):
        if j<len(inflow_data[0]):
            if j !=len(inflow_data[0])-1:
                tem.append(init_data[0][j]-flooding_data[i][j]/10+inflow_data[i][j]/10)#+lateral_inflow_data[i,j])#
            else:
                tem.append(init_data[0][j]-flooding_data[i][j]/28.26+inflow_data[i][j]/28.26-pump_data[i]*17.28/28.26)#只有一个泵，对应最后一位流量，所以这样做
        else:
            tem.append(init_data[0][j])
    tem=[tem]
    scalert.fit(tem)
    tem = scaler.transform(tem)
    init_data=clf.predict(tem)
    result.append(init_data[0])
    for j in range(len(init_data[0])):
        text+=str(init_data[0][j])+'\t'
    text+='\n'
    
f=open('sklearntest.txt','w')
f.write(text)
f.close()
print(len(result),len(result[0]))

#save_xls_file(result,'sktest')