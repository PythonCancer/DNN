# -*- coding:utf-8 -*-
__author__ = 'ZFJ'
__time__ = '2019/4/18'

import numpy as np
import matplotlib.pyplot as plt
# 此处选取skleran的datasets数据集中的威斯康辛州乳腺癌数据(该数据集特使适合于分类问题)
# 这个数据集包含了威斯康辛州记录的569个病人的乳腺癌恶性/良性(1/0)类别型数据(训练目标)，
# 以及与之对应的30个维度的生理指标数据；因此这是个非常标准的二类判别数据集
# 在这里只需要使用load_breast_cancer(return_X_y)来导出数据即可
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# initialize parameters(w,b)
# 这里的权重采用随机初始化，偏置采用零值初始化
# 需要注意的是，在logistic回归中，可以将权重全部初始化为0，但是在本次的神经网络中是不可以全部初始化为0的
# 因为这样会使得梯度下降无效，理由是：无论哪个初始化的输入和零权重相乘得到的结果都是零，激活函数也是如此，那么最终导致反向传播的结果也是一样的
def initialize_parameters(layer_dims):
    """
    初始化权重和偏置
    :param layer_dims: list,每一层单元的个数(维度)
    :return: dictionary,存储w1,w2,...,wL,b1,b2,..,bL
    """
    # 设置种子数为3，保证了每次产生的值都是固定的
    np.random.seed(3)
    # 每一层中神经元的个数
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        # Note:W的维度是(当前神经元的个数*前一层神经元的个数)
        # 在这里随机初始化参数w时，之所以要*0.01，是因为，如果我们不乘这个0.1，那么参数初始化会可能得到一个较大的数
        # 那么z=w^T * X + b 得到的Z值就会比较大，再使用sigmod或者是tanH作为激活函数时，得到的剃度会非常的小，这样会使得参数的更新速度很慢
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# 实现前项传播
def liner_forward(A_pre, W, b):
    """
    前项传播
    :param A_pre:上一层的激活值,shape:(size of previous layer,m)
    :param W: 权重矩阵,shape:(size of current layer,size of previous layer)
    :param b: 偏置向量,shpae:(size of current layer,1)
    :return:
    Z:激活函数的输入值(就是通过线性相加得到的和)
    cache:因为bp的时候要用到w,b和a，所以我将每一层的都存储起来，以便后面使用
    """
    Z = np.dot(W, A_pre) + b
    return Z


# sigmod实现前向传播
def sigmod(Z):
    """
    sigmod激活函数
    :param Z: Output of the linear layer
    :return:
    A:output of the activation
    """
    A = 1 / (1 + np.exp(-Z))
    return A


# 整个Forward propagation的过程
def forward_propagation(X, parameters):
    """
    整个前向传播的过程
    :param X:input dataset,shape(input size,number of examples)
    :param parameters:python dictionary containing your parameters"W1","b1","W2","b2",..."WL","bL"
                      W -- 权重矩阵 ,shape(size of current layer,size of previous layer)
                      b -- 偏置向量,shape(size of current layer,1)
    :return:
    AL -- the output of the last Layer(y_predict)
    caches -- list,every element is a tuple:(W,b,z,A_pre)
    """
    # number of layer
    L = len(parameters) // 2
    A = X
    caches = []
    # calculate from 1 to L-1 layer
    for l in range(1, L):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        # 整个前向传播的过程:linear forward -> relu ->linear forward....
        z = liner_forward(A, W, b)
        # 需要注意：以激活函数为分割，到z认为是这一层的，激活函数的输出值A认为是下一层的输入，划归到下一层。
        # 注意cache的位置，要放在sigmod前面。
        caches.append((A, W, b, z))
        # 调用sigmod激活函数
        A = sigmod(z)

    # caculate Lth layer
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    zL = liner_forward(A, WL, bL)
    caches.append((A, WL, bL, zL))
    AL = sigmod(zL)
    return AL, caches


# caculate cost function
def compute_cost(AL, Y):
    """
    计算损失值
    :param AL:最后一层的激活值，就是预测值,shape:(1,number of examples)
    :param Y: 真实值,shape:(1,number of examples)
    :return:
    """
    m = Y.shape[1]
    cost = 1. / m * np.nansum(np.multiply(-np.log(AL), Y) +
                              np.multiply(-np.log(1 - AL), 1 - Y))
    # 从数组的形状中删除单维度条目，就是把shape中为1的维度去掉，例如把[[[2]]]变成2
    cost = np.squeeze(cost)
    return cost


# derivation of sigmod
def sigmoid_backward(dA, Z):
    """
    :param dA:
    :param Z:
    :return:
    """
    a = 1 / (1 + np.exp(-Z))
    dZ = dA * a * (1 - a)
    return dZ


# derivation of linear
def linear_backward(dZ, cache):
    """
    :param dZ: Upstream derivative, the shape (n^[l+1],m)
    :param A: input of this layer
    :return:
    """
    A, W, b, z = cache
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    da = np.dot(W.T, dZ)
    return da, dW, db


# BP propagation
def backward_propagation(AL, Y, caches):
    """
    反向传播
    :param AL:
    :param Y:实际值的向量组
    :param caches:caches output from forward_propagation(),即(W,b,z,pre_A)
    :return:
    gradients -- Adictionary with the gradients with respect to dW,db
    """
    m = Y.shape[1]
    L = len(caches) - 1
    # caculate the Lth layer gredients
    dz = 1. / m * (AL - Y)
    da, dWL, dbL = linear_backward(dz, caches[L])
    gradients = {"dW" + str(L + 1): dWL, "db" + str(L + 1): dbL}

    # calculate from L-1 to 1 layer gradients
    # 反转迭代器，即L-1,L-3,...,0
    for l in reversed(range(0, L)):
        A, W, b, z = caches[l]
        # sigmod backward -> linear backward
        # sigmod backward
        dout = sigmoid_backward(da, z)
        # linear backward
        da, dW, db = linear_backward(dout, caches[l])
        # print("========dW" + str(l+1) + "================")
        # print(dW.shape)
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db
    return gradients


# 更新参数w和b
def update_parameters(parameters, grads, learning_rate):
    """
    参数更新
    :param parameters:dictionary,W,b
    :param grads:dW,db
    :param learning_rate:alpha
    :return:
    """
    L = len(parameters) // 2
    for l in range(L):
        # 参数更新方式就是：W-=学习率*梯度更新；b-=学习率*梯度更新
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, mini_batch_size=64):
    """
    把fp和bp串联起来，实现整个网络的功能
    :param X: input
    :param Y: 带有标签的真实值
    :param layer_dims: list containing the input size and each layer size
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :return:
    parameters -- final parameters:(W,b)
    """
    costs = []
    # 初始化参数
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        # foward propagation
        AL, caches = forward_propagation(X, parameters)
        # calculate the cost
        cost = compute_cost(AL, Y)
        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
        # backward propagation
        grads = backward_propagation(AL, Y, caches)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

    print('length of cost is: ', len(costs))
    # print(len(costs))
    # plt.cla()清除轴，当前活动轴在当前的图中，它保持其它轴不变
    # plt.clf()清除整个当前的数字，与所有的轴，但离开窗口打开，这样它就可以再用在其他的plots上了
    # plt.close()关上窗户，如果未另指定，则该窗口将是当前窗口
    plt.clf()
    # 画图
    plt.plot(costs)
    # 横坐标名字
    plt.xlabel("iterations(thousand)")
    # 横坐标名字
    plt.ylabel("cost")
    # 显示学习率
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()
    return parameters


# predict function
def predict(X_test, y_test, parameters):
    """
    根据神经网络的输出值来进行预测
    :param X_test:
    :param y_test:
    :param parameters:
    :return:
    """
    m = y_test.shape[1]
    Y_prediction = np.zeros((1, m))
    prob, caches = forward_propagation(X_test, parameters)
    for i in range(prob.shape[1]):
        # 转换概率A[0,i]到实际的预测概率p[0,i]
        if prob[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    accuracy = 1 - np.mean(np.abs(Y_prediction - y_test))
    return accuracy


# 给出参数，跑跑看效果
# DNN模型
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.01, num_iterations=100000, mini_batch_size=256):
    parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations, mini_batch_size)
    train_accuracy = predict(X_train, y_train, parameters)
    test_accuracy = predict(X_test, y_test, parameters)
    return train_accuracy, test_accuracy


# Today is to kill me, I will not change this Bug
if __name__ == "__main__":
    X_data, y_data = load_breast_cancer(return_X_y=True)
    # train_test_split是交叉验证中常用的函数，功能是从样本中随机的按照比例选取train data和test data
    # random_state:随机数的种子，随机数种子其实就是改组随机数的编号
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.5, test_size=0.5, random_state=28)
    X_train = X_train.T
    y_train = y_train.reshape(y_train.shape[0], -1).T
    X_test = X_test.T
    y_test = y_test.reshape(y_test.shape[0], -1).T
    train_accuracy, test_accuracy = DNN(X_train, y_train, X_test, y_test, [X_train.shape[0], 10, 5, 1])
    print('train accuracy is : ', train_accuracy)
    print('test accuracy is : ', test_accuracy)
