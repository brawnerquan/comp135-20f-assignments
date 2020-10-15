import os
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics

import matplotlib.pyplot as plt


df_y_train = pd.read_csv("data_digits_8_vs_9_noisy/y_train.csv")
df_y_valid = pd.read_csv("data_digits_8_vs_9_noisy/y_valid.csv")
df_x_train = pd.read_csv("data_digits_8_vs_9_noisy/x_train.csv")
df_x_valid = pd.read_csv("data_digits_8_vs_9_noisy/x_valid.csv")

train_count = df_y_train.describe().to_numpy()[0,0]
valid_count = df_y_valid.describe().to_numpy()[0,0]

pos_train = df_y_train.to_numpy().sum()
pos_valid = df_y_valid.to_numpy().sum()

frac_train = pos_train/train_count
frac_valid = pos_valid/valid_count

print("train count:",  train_count)
print("test count:",  valid_count)
print("frac train:", frac_train)
print("frac valid:", frac_valid)

print("pos train", pos_train)
print("pos valid", pos_valid)

print("============================")

x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy().flatten()
x_valid = df_x_valid.to_numpy()
y_valid = df_y_valid.to_numpy().flatten()

"""
iter_list = []
entropy_train_list = []
entropy_valid_list = []
zero_train_list = []
zero_valid_list = []


for i in range(1,41):
    
    model = sklearn.linear_model.LogisticRegression(C=1e6, solver='lbfgs', max_iter=i)

    model.fit(x_train, y_train)
    y_hat = model.predict(x_train).flatten()
    
    log_loss_train = sklearn.metrics.log_loss(y_train, y_hat)
    zero_loss_train = sklearn.metrics.zero_one_loss(y_train, y_hat)

    y_hat = model.predict(x_valid).flatten()

    log_loss_valid = sklearn.metrics.log_loss(y_valid, y_hat)
    zero_loss_valid = sklearn.metrics.zero_one_loss(y_valid, y_hat)
    
    iter_list.append(i)
    entropy_train_list.append(log_loss_train)
    entropy_valid_list.append(log_loss_valid)
    zero_train_list.append(zero_loss_train)
    zero_valid_list.append(zero_loss_valid)

plt.plot(iter_list, entropy_train_list, '.-', c='b', label="training set")
plt.plot(iter_list, entropy_valid_list, '.-', c='r', label="validation set")

plt.ylabel("log loss")
plt.xlabel("iterations")

plt.legend(loc='upper right')

plt.show()


plt.plot(iter_list, zero_train_list, '.-', c='b', label="training set")
plt.plot(iter_list, zero_valid_list, '.-', c='r', label="validation set")

plt.ylabel("error rate")
plt.xlabel("iterations")

plt.legend(loc='upper right')

plt.show()
    
"""

"""
C_grid = np.logspace(-9, 6, 31)
zero_train_list = []
zero_valid_list = []

for C in C_grid:
    # Build and evaluate model for this C value
    model = sklearn.linear_model.LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
    model.fit(x_train, y_train)

    # Record training and validation set error rate
    y_hat = model.predict(x_train).flatten()
    zero_loss = sklearn.metrics.zero_one_loss(y_train, y_hat)
    zero_train_list.append(zero_loss)

    y_hat = model.predict(x_valid).flatten()
    zero_loss = sklearn.metrics.zero_one_loss(y_valid, y_hat)
    zero_valid_list.append(zero_loss)


plt.plot(C_grid, zero_train_list, '.-', c='b', label="training set")
plt.plot(C_grid, zero_valid_list, '.-', c='r', label="validation set")

plt.ylabel("error rate")
plt.xlabel("C")

plt.legend(loc='upper right')

plt.show()


optimal_C = C_grid[np.argmin(zero_valid_list)]
print(optimal_C)
"""

optimal_C = 0.01


model = sklearn.linear_model.LogisticRegression(C=optimal_C, solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)

y_hat = model.predict(x_valid).flatten()

FN_id = np.logical_and(y_hat == 0, y_valid == 1).flatten()
FP_id = np.logical_and(y_hat == 1, y_valid == 0).flatten()

FN_id = np.arange(valid_count)[FN_id]
FP_id = np.arange(valid_count)[FP_id]


"""
import show_images
show_images.show_images(x_valid, y_hat, FN_id[:9].astype(int))

plt.show()

show_images.show_images(x_valid, y_hat, FP_id[:9].astype(int))

plt.show()
"""

weights = model.coef_.reshape((28,-1))

plt.imshow(weights, vmin=-0.5, vmax=0.5, cmap='RdYlBu')
plt.show()

