# -*- coding:utf-8 -*-
__author__ = 'ZFJ'
__time__ = '2019/4/9'

import numpy as np
import matplotlib.pyplot as plt


# initialize parameters(w,b)
# 这里的权重采用随机初始化，偏置采用零值初始化
# 需要注意的是，在logistic回归中，可以将权重全部初始化为0，但是在本次的神经网络中是不可以全部初始化为0的
# 因为这样会使得梯度下降无效，理由是：无论哪个初始化的输入和零权重相乘得到的结果都是零，激活函数也是如此，那么最终导致反向传播的结果也是一样的
def initialize_parameters(in_n, h_n, out_n):
    """
    初始化权重和偏置
    :param in_n: 输入层的节点数
    :param h_n: 隐藏层的节点数
    :param out_n: 输出层的节点数
    :return:
    params --
              W1 --  权重矩阵，维度为(h_n, in_n)
              b1 -- 偏置向量，维度为(h_n, 1)
              W2 -- 权重矩阵，维度为 (out_n, h_n)
              b2 -- 偏置向量，维度为 (out_n, 1)
    """
    # Note:这里一定要注意要*0.01，目的是：假设不*0.01在参数初始化的时候可能会得到一个较大的数，那么会导致z=w^T * X + b比较大
    # 这样的话，如果使用sigmod或者是tanH作为激活函数的话，得到的梯度会非常小，这样的参数更新的速度会非常的慢
    W1 = np.random.randn(h_n, in_n) * 0.01
    b1 = np.zeros((h_n, 1))
    W2 = np.random.randn(out_n, h_n) * 0.01
    b2 = np.zeros((out_n, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# 激活函数
# Note:因为我们这里有隐藏层，所以我选用了两个不同的激活函数，隐藏层的激活函数是使用的tanh()函数，输出层还是使用sigmod()函数
def sigmod(z):
    """
    sigmod激活函数
    :param z: 输入
    :return: sigmod(z)
    """
    return 1 / (1 + np.exp((-z)))


def tanh(z):
    """
    tanh激活函数
    :param z: 输入
    :return: tanh(z)
    """
    return 2 * sigmod(2 * z) - 1


# BP算法
# 因为含有隐藏层，所以BP算法来的复杂一些，这是因为我们需要一层一层的正向传播，然后在反向传播回来。
# 有无隐藏层的正向传播都是一样的，都是从输入层开始依次计算每一层的输入，上一次的输出作为当前层的输入
# 方向传播则是从输出层开始依次进行梯度的下降
# 前向传播
def forward_propagation(X, parameters):
    """
    前向传播
    :param X:输入数据
    :param parameters:参数(包含W1,b1,W2,b2)
    :return:
    A2 -- 网络输出
    temp -- 包含Z1,A1,Z2,A2的字典，用于BP算法中
    """
    # 取回参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmod(Z2)

    temp = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

    return A2, temp


# 反向传播
def backward_propagation(parameters, temp, X, Y):
    """
    反向传播
    :param parameters:参数(包含W1,b1,W2,b2)
    :param temp: 包含Z1,A1,Z2,A2的字典，用于BP算法
    :param X: 输入数据
    :param Y: 输入数据标签
    :return:
    grads -- 返回不同参数的梯度
    cost -- 损失函数
    """
    # 取回参数
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = temp["A1"]
    A2 = temp["A2"]

    # 样本的数目，为了后面计算交叉熵的时候使用
    num = Y.shape[0]
    # 交叉熵损失函数
    cost = -1 / num * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

    # 反向传播
    dZ2 = A2 - Y
    dW2 = 1 / num * np.dot(dZ2, A1.T)
    # Note:axis的取值有三种情况，分别是None(默认)、整数和整数元组
    # axis = 0表示压缩行，就是把每一列的元素相加，将矩阵压缩为一行
    # axis = 1表示压缩列，就是把每一行的元素相加，将矩阵压缩为一列
    # axis = -1相当于axis=1；axis = -2相当于axis = 0的效果
    # keepdims是用来保持矩阵的二维特性
    db2 = 1 / num * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / num * np.dot(dZ1, X.T)
    db1 = 1 / num * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

    return cost, gradients


# 参数更新
def update_parameters(parameters, gradients, learning_rate):
    """
    梯度下降更新参数
    :param parameters: 参数(包含W1,b1,W2,b2)
    :param gradients: 不同参数的梯度
    :param learning_rate: 学习率
    :return:
    :parameters -- 更新后的参数
    """
    # 取回参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    # 更新参数
    # Note:New_W = Old_W - learning_rate*dW
    # New_b = Old_b - learning_rate*db
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * W2
    b2 -= learning_rate * b2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 预测
def predict(parameters, X):
    """
    使用学习好的参数进行预测
    :param parameters: 学习好的参数
    :param X: 输入的数据
    :return:
    predictions -- 预测结果(0/1)
    """
    A2, temp = forward_propagation(X, parameters)
    predictions = np.round(A2)


if __name__ == "__main__":
    # X为横坐标，Y为纵坐标
    X = [0, 0, 1, 1]
    Y = [0, 1, 0, 1]
    label = [1, 0, 0, 1]
    # 第一类为蓝色，第二类为红色
    label_color = ['blue', 'red']
    color = []
    for i in label:
        if i == 1:
            color.append(label_color[0])
        else:
            color.append(label_color[1])

    # 数据的归一化处理
    X = np.array(X)
    Y = np.array(Y)
    X = (X - np.average((X)))
    Y = (Y - np.average((Y)))
    X = X / X.max()
    Y = Y / Y.max()
    for i in range(len(X)):
        plt.scatter(X[i], Y[i], c=color[i])
    plt.title("Normalization Data")
    plt.show()

    data_X = np.vstack((X, Y))
    data_label = np.array([label])

    # 参数设置
    in_n = 2
    h_n = 3
    out_n = 1
    costs = []
    Y_prediction = []
    iters = 20000
    learning_rate = 2
    parameters = initialize_parameters(in_n, h_n, out_n)

    # 开始训练
    for i in range(iters):
        # 前向传播
        A2, temp = forward_propagation(data_X, parameters)
        # 反向传播
        costTemp, gradients = backward_propagation(parameters, temp, data_X, data_label)
        costs.append(costTemp)
        # 参数更新
        parameters = update_parameters(parameters, gradients, learning_rate)

    # 预测
    Y_prediction = predict(parameters, data_X)

    plot_decision_boundary = (lambda x: predict(parameters, x.T), data_X, data_label)
    plt.show()

    # 画图
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()
