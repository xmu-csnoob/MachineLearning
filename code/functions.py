import numpy as np


# 简单通过|yhat-y|来衡量损失值
def lostFunction01(y,yhat):
    return np.sum(np.abs(y-yhat))


# 通过差值平方衡量损失值
def lostFunction02(y,yhat):
    return np.dot(y-yhat,(y-yhat).T)
