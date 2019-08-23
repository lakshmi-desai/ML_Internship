# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:30:58 2019

@author: Anuraag Manvi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("E:\\Internship\\65 Years of Weather Data Bangladesh (1948 - 2013).csv")

features = pd.DataFrame(data)

plt.figure(figsize=(12,10))
cor = features.corr()
sns.heatmap(cor, annot=True, cbar=False, cmap=plt.cm.Greens)
plt.show()

cor_target = abs(cor["Rainfall"])
relevant_features = cor_target[cor_target>0.6]#.sort_values(ascending=False)

print(relevant_features)

X = data["Cloud Coverage"]
Y = data["Rainfall"]

X = np.array((X - X.min())/(X.max() - X.min()))
Y = np.array((Y - Y.min())/(Y.max() - Y.min()))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

plt.plot(x_train, y_train, 'r.')

sns.relplot(x=x_train, y=y_train, data=data)

sns.relplot(x=X, y=Y, hue="Cloud Coverage", data=data)

plt.plot(x_test, y_test, 'b.')

def hypothesis(a,b,x):
    return a * x  + b

def error(a,b,x,y):
    e = 0
    m = len(y)
    for i in range(m):
        e += np.power((hypothesis(a,b,x[i]) - y[i]), 2)

    return (1/(2 * m)) * e

def step_gradient(a,b,x,y,learning_rate):
    grad_a = 0
    grad_b = 0
    m = len(x)
    for i in range(m):
        grad_a += 1/m * (hypothesis(a,b,x[i]) - y[i]) * x[i]
        grad_b += 1/m * (hypothesis(a,b,x[i]) - y[i])

    a = a - (grad_a * learning_rate)
    b = b - (grad_b * learning_rate)

    return a, b

def descend(initial_a, initial_b, x, y, learning_rate, iterations):
    a = initial_a
    b = initial_b
    for i in range(iterations):
        e = error(a, b, x ,y)
        if i % 1000 == 0:
            print(f"Error: {e}, a: {a}, b: {b}")

        a, b = step_gradient(a, b, x, y, learning_rate)

    return a, b

a = 0
b = 1
learning_rate = 0.01
iterations = 10000

final_a, final_b = descend(a, b, x_train, y_train, learning_rate, iterations)

print(error(a,b,x_train,y_train))
print(error(final_a, final_b, x_train, y_train))
print(error(final_a, final_b, x_test, y_test))

plt.plot(x_train, y_train, 'r.', x_train, hypothesis(a, b, x_train), 'g', x_train, hypothesis(final_a, final_b, x_train), 'b', )

plt.plot(x_test, y_test, 'r.', x_test, hypothesis(final_a, final_b, x_test), 'g')

print(str((1-error(final_a, final_b, x_test, y_test))*100) + " %")