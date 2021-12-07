import math
import numpy as np


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数求导
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# 图像转化为单维度向量
def image2vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)


# 矩阵行标准化
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / x_norm


# softmax函数
def softmax(x):
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp/sum_exp


# 求导函数
def derivative(func, param, value):
    d = func.diff(param)
    return d.subs({param: value})
