#!/usr/bin/env python3  
# -*- coding: utf-8 -*-

import numpy as np
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
        # print("prepare user list finish")

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
        # print("prepare train data finish")

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
        # print("prepare test data finish")

    del users
    gc.collect()
    return X_train, X_test

time1 = time.time()
X_train, X_test = prepareData()
time2 = time.time()
print("预处理数据的时间为:", str(time2 - time1))

# 计算余弦相似度
# sim_users[i][j]存储了用户i和用户j之间的相似度
time3 = time.time()
norm = np.matrix(np.linalg.norm(X_train, axis=1))
sim_users = np.dot(X_train, X_train.T) / np.dot(norm.T, norm)
time4 = time.time()
print("用户相似度矩阵计算时间:", str(time4 - time3))


# 计算用户对电影打分的估计
X_train = np.array(X_train)
sim_users = np.array(sim_users)
time5 = time.time()
score = np.dot(sim_users, X_train)/np.dot(sim_users, X_train>0)
time6 = time.time()
print("协同过滤:", str(time6 - time5))


# 计算 RMSE
N = np.sum(X_test>0)
RMSE = np.sqrt(np.sum(np.square(score * (X_test > 0) - X_test)) / N)
print("RMSE: ", RMSE)


