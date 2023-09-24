# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 19:08:40 2021

@author: DELL
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn

np.random.seed(2018-10-1)
N = 1000

_x = np.linspace(0, np.pi, num=N)
x0 = np.array([_x, np.sin(_x)]).T
x1 = -1 * x0 + [np.pi / 2, 0]

scale = 0.15
x0 += np.random.normal(scale=scale, size=(N, 2))
x1 += np.random.normal(scale=scale, size=(N, 2))

X = np.vstack([x0, x1])

proto_0 = np.array([[0], [0]]).T # the single "labeled" 0
proto_1 = np.array([[np.pi / 2], [0]]).T # the single "labeled" 1

model = RandomForestClassifier()
model.fit(np.vstack([proto_0, proto_1]), np.array([0, 1]))
for itercount in range(100):
    labels = model.predict_proba(X)[:, 0]
    labels += (np.random.random(labels.size) - 0.5) / 10 # add some noise
    labels = labels > 0.5
    model = RandomForestClassifier()
    model.fit(X, labels)

f, axs = plt.subplots(1, 2, squeeze=True, figsize=(10, 5))

axs[0].plot(x0[:, 0], x0[:, 1], '.', alpha=0.25, label='unlabeled x0')
axs[0].plot(proto_0[:, 0], proto_0[:, 1], 'o', color='royalblue', markersize=10, label='labeled x0')
axs[0].plot(x1[:, 0], x1[:, 1], '.', alpha=0.25, label='unlabeled x1')
axs[0].plot(proto_1[:, 0], proto_1[:, 1], 'o', color='coral', markersize=10, label='labeled x1')
axs[0].legend()

axs[1].plot(X[~labels, 0], X[~labels, 1], '.', alpha=0.25, label='predicted class 0')
axs[1].plot(X[labels, 0], X[labels, 1], '.', alpha=0.25, label='predicted class 1')
axs[1].plot([np.pi / 4] * 2, [-1.5, 1.5], 'k--', label='halfway between labeled data')
axs[1].legend()
plt.show()