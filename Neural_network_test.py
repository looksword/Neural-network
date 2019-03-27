# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:59:08 2018

@author: Administrator
"""

#神经网络程序（参考）

#1
#建立单层神经网络，训练四个样本，
import numpy as np
def nonlin(x,deriv=False): #deriv为False计算前向传播值，为True时计算反向偏导
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #输入样本，四个样本，每个样本三个特征向量
y = np.array([[0,0,1,1]]).T #期望输出
np.random.seed(1)#设定随机种子

w = 2*np.random.random((3,1))-1 #使用高斯变量初始化权值，E(x)=0,D(x)=1,w的值在[-1，+1]之间；

for iter in range(10000): #迭代一万次
    l0 = X  #输入给l0
    l1 = nonlin(np.dot(l0,w)) #计算经过第一层后的得分函数
    l1_error = y-l1 #计算Loss值，相当于损失函数的偏导
    l1_grad = l1_error*nonlin(l1,True) #Loss值带入梯度公式计算梯度
    w += np.dot(l0.T,l1_grad) #最终的权重梯度
print (l1)

#2
#两层神经网络
import numpy as np
def nonlin(x,deriv=False): #deriv为False时计算前向传播，为True计算反向偏导，激活函数为sigmoid函数
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #输入样本
y = np.array([[0,0,1,1]]).T #期望输出
w0 = 2*np.random.random((3,4))-1 #第一层权重
w1 = 2*np.random.random((4,1))-1 #第二层权重

for iter in range(10000): #迭代一万次
    l0 = X
    l1 = nonlin(np.dot(l0,w0)) #计算第一层后的得分
    l2 = nonlin(np.dot(l1,w1)) #经过第二层后的得分
    l2_error = y-l2  #计算Loss值，损失函数的偏导
    l2_grad = l2_error*nonlin(l2,deriv=True)#第二层梯度，l2_error越大，第二层的梯度也越大
    l1_error = l2_grad.dot(w1.T)#l1_error由l2_error迭代进来
    l1_grad = l1_error*nonlin(l1,deriv=True)
    w1+=l1.T.dot(l2_grad)
    w0+=l0.T.dot(l1_grad)
print(l2)

