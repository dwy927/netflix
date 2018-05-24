#!/usr/bin/env python3  
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import gc
import time

data_root = '../Project2-data'


def prepareData():
    # X_ij means the score from user i to movie j
    # the score range from 1 to 5, 0 means no score
    X_train = np.zeros((10000, 10000))
    X_test = np.zeros((10000, 10000))

    users = {}
    with open(data_root + '/users.txt') as f:
        for index, line in enumerate(f):
            line = line.strip()
            users[line] = index
        print("prepare user list finish")

    with open(data_root + '/netflix_train.txt') as f:
        for line in f.readlines():
            line = line.split()
            user = int(users[line[0]])
            movie = int(line[1]) - 1
            score = int(line[2])
            if X_train[user][movie] != 0:
                print("X_train[%d][%d] not 0" % (user, movie))
                continue
            X_train[user][movie] = score
        print("prepare train data finish")

    with open(data_root + '/netflix_test.txt') as f:
        for line in f.readlines():
            line = line.split()
            user = int(users[line[0]])
            movie = int(line[1]) - 1
            score = int(line[2])
            if X_test[user][movie] != 0:
                print("X_test[%d][%d] not 0" % (user, movie))
                continue
            X_test[user][movie] = score
        print("prepare test data finish")

    del users
    gc.collect()
    return X_train, X_test

'''
k: 隐空间数
lam: 控制正则项
step: 迭代次数
'''
def gradientDescent(k, lam, step, isplot):
    m = 10000  # 用户数
    n = 10000  # 电影数
    alpha = 0.00001  # 学习率


    U = np.random.rand(m, k) * 0.01
    V = np.random.rand(n, k) * 0.01
    A = X_train > 0
    N = np.sum(X_test > 0)

    RMSE = []
    J = []
    t = []

    time1 = time.time()
    for i in range(step):
        du = np.dot((np.dot(U, V.T) - X_train) * A, V) + 2 * lam * U
        dv = np.dot(((np.dot(U, V.T) - X_train) * A).T, U) + 2 * lam * V
        U = U - alpha * du
        V = V - alpha * dv
        if isplot:
            score = np.dot(U, V.T)
            rmse = np.sqrt(np.sum(np.square(score * (X_test > 0) - X_test)) / N)
            j = np.square(np.linalg.norm(A*(X_train - np.dot(U, V.T)), ord='fro')) / 2 + \
                lam * np.square(np.linalg.norm(U, ord='fro')) + \
                lam * np.square(np.linalg.norm(V, ord='fro'))
            if i % 50 == 0:
                t.append(i)
                RMSE.append(rmse)
                J.append(j)
        else:
            if i == step - 1:
                score = np.dot(U, V.T)
                rmse = np.sqrt(np.sum(np.square(score * (X_test > 0) - X_test)) / N)
                j = np.square(np.linalg.norm(A * (X_train - np.dot(U, V.T)), ord='fro')) / 2 + \
                    lam * np.square(np.linalg.norm(U, ord='fro')) + \
                    lam * np.square(np.linalg.norm(V, ord='fro'))
                RMSE.append(rmse)
                J.append(j)

    time2 = time.time()
    print("k=%d, lam=%f,梯度下降时间开销为: %s, RMSE=%f , J=%f\n" % (k, lam, str(time2 - time1), RMSE[-1], J[-1]))

    if isplot:
        fig, ax = plt.subplots()
        ax.plot(t, RMSE)
        title = "RMSE when k=" + str(k) + ", lam=" + str(lam)
        ax.set(xlabel='iter times', ylabel='RMSE', title=title)
        ax.grid()
        fileName = "output/RMSE_k" + str(k) + "lam" + str(lam) + str(".png")
        fig.savefig(fileName)
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(t, J)
        title = "J when k=" + str(k) + ", lam=" + str(lam)
        ax.set(xlabel='iter times', ylabel='J', title=title)
        ax.grid()
        fileName = "output/J_"+ str(step) + "k" + str(k) + "lam" + str(lam) + str(".png")
        fig.savefig(fileName)
        plt.show()



if __name__ == "__main__":
    X_train, X_test = prepareData()
    k_list = [50, 20, 10]  # 隐空间数
    lam_list = [0.001, 0.01, 0.1] # 控制正则项大小
    # gradientDescent(50, 0.01, 1000, True)
    for k in k_list:
        for lam in lam_list:
            gradientDescent(k, lam, 100, False)