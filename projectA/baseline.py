import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics

import matplotlib.pyplot as plt


df_x_test = pd.read_csv("data_sneaker_vs_sandal/x_test.csv")
df_x_train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv")
df_y_train = pd.read_csv("data_sneaker_vs_sandal/y_train.csv")


x_test = df_x_test.to_numpy()
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()


"""
new_x_train = []

train_xs = np.repeat([np.arange(28)], 28, axis=0)
train_ys = train_xs.T



for row in x_train:
    x_pos = (row.reshape((28,28))*train_xs).sum()/(np.sum(row))
    y_pos = (row.reshape((28,28))*train_ys).sum()/(np.sum(row))

    shift_x = int(14 - x_pos) 
    shift_y = int(14 - y_pos)

    ori = row.reshape((28,28)).copy()

    out1 = np.zeros((28,28))

    if shift_x > 0:
        out1[:, shift_x:] = ori[:, :-shift_x]
    elif shift_x < 0:
        out1[:, :shift_x] = ori[:, -shift_x: ]

    out2 = np.zeros((28,28))

    if shift_y > 0:
        out2[shift_y:, :] = out1[:-shift_y, :]
    elif shift_y < 0:
        out2[:shift_y, :] = out1[-shift_y:, : ]

    new_x_train.append(out2.flatten())

train_x = np.array(new_x_train)
"""
"""

C = np.logspace(-8, 0, 31)
kf = KFold(n_splits=10)

C_lst_train = []
C_lst_valid = []


for c in C:
  model = sklearn.linear_model.LogisticRegression(C=c, solver='lbfgs', max_iter=100)

  avg_score_tr = []
  avg_score_va = []
  for (i, (tr_i, ts_i)) in enumerate(kf.split(x_train)):
      curr_train_x , curr_valid_x = x_train[tr_i], x_train[ts_i]
      curr_train_y , curr_valid_y = y_train[tr_i], y_train[ts_i]
      
      
      model.fit(curr_train_x, curr_train_y.flatten())
      
      
      y_hat = model.predict(curr_train_x).flatten()
      log_loss = sklearn.metrics.log_loss(curr_train_y, y_hat)
      avg_score_tr.append(log_loss)
      y_hat = model.predict(curr_valid_x).flatten()
      log_loss = sklearn.metrics.log_loss(curr_valid_y, y_hat)
      avg_score_va.append(log_loss)
      print(i)
  C_lst_train.append(np.mean(avg_score_tr))
  C_lst_valid.append(np.mean(avg_score_va))

print(C_lst_train)
print(C_lst_valid)

print(C[np.argmin(C_lst_valid)])

plt.plot(C, C_lst_train, '.-', c='b', label="training set")
plt.plot(C, C_lst_valid, '.-', c='r', label="validation set")

plt.title("Log Loss vs C Hyperparameter Selection")


plt.ylabel("log loss")
plt.xlabel("C")

plt.legend(loc='upper right')
plt.show()

"""



train_in, test_in, train_out, test_out = train_test_split(x_train, y_train, test_size=0.3)



model = sklearn.linear_model.LogisticRegression(C=0.292, solver='lbfgs', max_iter=1000)
model.fit(train_in, train_out.flatten())

plt.imshow(model.coef_.reshape((28,28)))
plt.show()

save = model.predict_proba(test_in)[:, 1]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_out.flatten(), save.flatten())

print(fpr)

plt.plot(fpr, tpr, '.-', c='b', label="validation set")

save = model.predict_proba(train_in)[:, 1]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_out.flatten(), save.flatten())

plt.plot(fpr, tpr,'.-', c='r', label="training set")

plt.title("ROC curve of Baseline")


plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

plt.legend(loc='lower right')
plt.show()

print(len(np.where(save == test_out.flatten())[0]))
print(test_out.shape)
print(sklearn.metrics.zero_one_loss(save, test_out))

