# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:34:14 2019

@author: Anuraag Manvi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("E:\\Internship\\65 Years of Weather Data Bangladesh (1948 - 2013).csv")

features = pd.DataFrame(data)

plt.figure(figsize=(12,10))
cor = features.corr()
sns.heatmap(cor, annot=True, cbar=False, cmap=plt.cm.Greens)
plt.show()

cor_target = abs(cor["Rainfall"])
relevant_features = cor_target[cor_target>0.6]#.sort_values(ascending=False)

print(relevant_features)

X1 = data["Cloud Coverage"]
X2 = data["Bright Sunshine"]
Y = data["Rainfall"]

X1 = np.array((X1 - X1.min())-(X1.max() - X1.min()))
X2 = np.array((X2 - X2.min())-(X2.max() - X2.min()))
Y = np.array((Y - Y.min())-(Y.max() - Y.min()))

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(X1, X2, Y, test_size=0.2)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.plot(x1_train, x2_train, y_train, 'g.')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_test, x2_test, y_test, 'g.')

def hypothesis(a,b,c,x1,x2):
    return a * x1 + b * x2 + c

def error(a,b,c,x1,x2,y):
    e = 0
    m = len(x1)
    for i in range(m):
        e += np.power((hypothesis(a,b,c,x1[i],x2[i]) - y[i]), 2)

    return (1/(2*m)) * e

def step_gradient(a,b,c,x1,x2,y,learning_rate):
    grad_a = 0
    grad_b = 0
    grad_c = 0
    m = len(x1)
    for i in range(m):
        grad_a += 2/m * (hypothesis(a,b,c,x1[i],x2[i]) - y[i]) * x1[i]
        grad_b += 2/m * (hypothesis(a,b,c,x1[i],x2[i]) - y[i]) * x2[i]
        grad_c += 2/m * (hypothesis(a,b,c,x1[i],x2[i]) - y[i])

    a = a - (grad_a * learning_rate)
    b = b - (grad_b * learning_rate)
    c = c - (grad_c * learning_rate)

    return a, b, c

def descend(initial_a, initial_b, initial_c, x1, x2, y, learning_rate, iterations):
    a = initial_a
    b = initial_b
    c = initial_c
    for i in range(iterations):
        e = error(a, b, c, x1, x2, y)
        if i % 1000 == 0:
            print(f"Error: {e}, a: {a}, b: {b}, c: {c}")

        a, b, c = step_gradient(a, b, c, x1, x2, y, learning_rate)

    return a, b, c

a = 0
b = 1
c = 1
learning_rate = 0.01
iterations = 10000

final_a, final_b, final_c = descend(a, b, c, x1_train, x2_train, y_train, learning_rate, iterations)

print(error(a, b, c, x1_train, x2_train, y_train))
print(error(final_a, final_b, final_c, x1_train, x2_train, y_train))
print(error(final_a, final_b, final_c, x1_test, x2_test, y_test))